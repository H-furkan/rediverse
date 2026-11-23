"""Krea 2 loading, text contexts, schedules and the single-forward helper shared by every krea2 stage.

Text contexts. Krea2Pipeline pads every prompt to 512 text tokens ([prefix | prompt | PAD | suffix]) and masks the
padding. torch SDPA with a key-padding mask plus GQA (48 q / 12 kv heads) falls back to the math kernel, which
materialises (B, 48, L, L) scores and runs out of memory at batch 4 (1024^2). Every krea2 stage therefore drops the
padded rows once, at encoding time ("compaction"): exact for this architecture, since text tokens carry all-zero
rotary ids in the DiT, the text-fusion blocks have no positional encoding, and padded rows are masked as keys
everywhere and never read back. A compacted context needs no mask, so one prompt x many noises runs on flash SDPA.
Every stage (probe, target bank, directions, training, generation) uses this same path.

Contexts are the stacked text-encoder taps [1, L_valid, 12, 2560] (before text fusion, which lives inside each
transformer and differs between Raw and Turbo), so one cached context serves both models.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from krea2 import config as C

_CACHE: dict = {}


# ------------------------------------------------------------------ loading --
def snapshot(repo: str, revision: str) -> str:
    """Local snapshot directory of a pinned revision (loads never touch the Hub; HF_HUB_OFFLINE-safe)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, local_files_only=True)


def _pipe_cls():
    from diffusers import Krea2Pipeline
    return Krea2Pipeline


def load_text_pipe(device):
    """Krea2Pipeline with the text encoder and tokenizer only (identical blobs in Raw and Turbo)."""
    return _pipe_cls().from_pretrained(snapshot(C.STUDENT_ID, C.STUDENT_REVISION), transformer=None, vae=None,
                                       torch_dtype=C.DTYPE).to(device)


def load_transformer(arm: str, device):
    """The frozen Krea2Transformer2DModel of `arm` (teacher = Raw, student = Turbo), bf16, eval."""
    from diffusers import Krea2Transformer2DModel
    repo, rev = (C.TEACHER_ID, C.TEACHER_REVISION) if arm == "teacher" else (C.STUDENT_ID, C.STUDENT_REVISION)
    m = Krea2Transformer2DModel.from_pretrained(snapshot(repo, rev), subfolder="transformer", torch_dtype=C.DTYPE)
    assert len(m.transformer_blocks) == C.N_BLOCKS, len(m.transformer_blocks)
    m.requires_grad_(False).eval()
    # device only: from_pretrained(torch_dtype=bf16) keeps the RMSNorm scales (_keep_in_fp32_modules: norm, norm1,
    # norm2, norm_q, norm_k) in fp32, as Krea2Pipeline does; a dtype cast here would round 1 + weight to bf16.
    m = m.to(device)
    bad = [n for n, p in m.named_parameters() if n.split(".")[-2] in m._keep_in_fp32_modules and p.dtype != torch.float32]
    assert not bad, f"norm weights not fp32: {bad[:3]}"
    return m


def load_vae(device):
    from diffusers import AutoencoderKLQwenImage
    return AutoencoderKLQwenImage.from_pretrained(snapshot(C.STUDENT_ID, C.STUDENT_REVISION), subfolder="vae",
                                                  torch_dtype=C.DTYPE).to(device).eval()


# ------------------------------------------------------------------ text --
@torch.no_grad()
def encode(pipe, prompt: str, device) -> torch.Tensor:
    """The compacted context of one prompt: [1, L_valid, 12, 2560] bf16 (padded rows dropped)."""
    pe, mask = pipe.encode_prompt(prompt=prompt, device=device, max_sequence_length=C.MAX_SEQ_LEN)
    assert pe.shape[0] == 1 and mask.shape == pe.shape[:2], (pe.shape, mask.shape)
    return pe[:, mask[0].bool()].to(C.DTYPE).contiguous()


def encode_many(prompts: list[str], device, log=print) -> list[torch.Tensor]:
    """Contexts of several prompts on the CPU; the text encoder is loaded, used and freed."""
    pipe = load_text_pipe(device)
    out = [encode(pipe, p, device).cpu() for p in prompts]
    del pipe
    torch.cuda.empty_cache()
    log(f"[krea2] encoded {len(prompts)} prompts (text rows {min(o.shape[1] for o in out)}..{max(o.shape[1] for o in out)})")
    return out


# ------------------------------------------------------------------ geometry --
def image_ids(device) -> torch.Tensor:
    """(t, h, w) rotary coordinates of the image rows, as Krea2Pipeline.prepare_position_ids builds them."""
    key = ("ids", str(device))
    if key not in _CACHE:
        ids = torch.zeros(C.GRID, C.GRID, 3, device=device)
        ids[..., 1] = torch.arange(C.GRID, device=device)[:, None]
        ids[..., 2] = torch.arange(C.GRID, device=device)[None, :]
        _CACHE[key] = ids.reshape(C.N_IMG_TOKENS, 3)
    return _CACHE[key]


def position_ids(text_len: int, device) -> torch.Tensor:
    return torch.cat([torch.zeros(text_len, 3, device=device), image_ids(device)], 0)


def extract(name: str, out) -> torch.Tensor:
    """Image rows of a Krea2TransformerBlock output [B, L_txt + 4096, 6144] -> [B, 4096, 6144]."""
    assert torch.is_tensor(out) and out.shape[-1] == C.D_MODEL, (name, type(out), getattr(out, "shape", None))
    return out[:, -C.N_IMG_TOKENS:]


def forward(model, x: torch.Tensor, ctx: torch.Tensor, sigma: float | torch.Tensor):
    """One transformer forward: x [B, 4096, 64] packed latents, ctx [1 or B, L, 12, 2560], at noise level sigma
    (the transformer's `timestep` is sigma in [0, 1], cast to the latent dtype as the pipeline does)."""
    b = x.shape[0]
    t = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma), device=x.device)
    t = t.to(device=x.device, dtype=x.dtype).reshape(-1).expand(b) if t.numel() == 1 else t.to(x.device, x.dtype)
    ctx = ctx.to(x.device, x.dtype)
    if ctx.shape[0] == 1 and b > 1:
        ctx = ctx.expand(b, -1, -1, -1)
    return model(hidden_states=x, encoder_hidden_states=ctx, timestep=t,
                 position_ids=position_ids(ctx.shape[1], x.device), encoder_attention_mask=None,
                 return_dict=False)[0]


def forward_to_blocks(model, x: torch.Tensor, ctx: torch.Tensor, sigma, blocks, checkpoint: bool = False) -> dict:
    """{b: image rows [B, 4096, 6144]} of the requested blocks, running the transformer only up to max(blocks).

    Mirrors Krea2Transformer2DModel.forward (diffusers 0.39, transformer_krea2.py:486-515) without a text mask
    (compacted contexts); verified against hooks on the full forward by check_truncation()."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as ckpt
    blocks = sorted(set(int(b) for b in blocks))
    b = x.shape[0]
    t = torch.full((b,), float(sigma), device=x.device).to(x.dtype) if not torch.is_tensor(sigma) else \
        sigma.to(x.device, x.dtype).reshape(-1).expand(b)
    ctx = ctx.to(x.device, x.dtype)
    if ctx.shape[0] == 1 and b > 1:
        ctx = ctx.expand(b, -1, -1, -1)
    temb = model.time_embed(t, dtype=x.dtype)
    temb_mod = model.time_mod_proj(F.gelu(temb, approximate="tanh"))
    ehs = model.txt_in(model.text_fusion(ctx, attention_mask=None))
    h = torch.cat([ehs, model.img_in(x)], dim=1)
    rot = model.rotary_emb(position_ids(ctx.shape[1], x.device))
    out = {}
    for i, blk in enumerate(model.transformer_blocks[:blocks[-1] + 1]):
        if checkpoint and torch.is_grad_enabled():
            h = ckpt(blk, h, temb_mod, rot, None, use_reentrant=False)
        else:
            h = blk(h, temb_mod, rot, None)
        if i in blocks:
            out[i] = h[:, -C.N_IMG_TOKENS:]
    return out


@torch.no_grad()
def check_truncation(model, ctx, device, blocks=(0, 13, 27)) -> float:
    """Max relative difference between forward_to_blocks and hooks on the full forward (expect exactly 0)."""
    x = torch.randn(2, *C.LATENT_SHAPE, generator=torch.Generator("cpu").manual_seed(0)).to(device, C.DTYPE)
    got = {}
    hs = [model.transformer_blocks[bk].register_forward_hook(
        lambda _m, _i, o, _b=bk: got.__setitem__(_b, extract(f"b{_b}", o).clone())) for bk in blocks]
    try:
        forward(model, x, ctx, 0.9)
    finally:
        for hh in hs:
            hh.remove()
    mine = forward_to_blocks(model, x, ctx, 0.9, blocks)
    return max(float((mine[bk].float() - got[bk].float()).norm() / got[bk].float().norm()) for bk in blocks)


# ------------------------------------------------------------------ schedules --
def calculate_mu(image_seq_len: int = C.N_IMG_TOKENS) -> float:
    """Krea2Pipeline / FLUX calculate_shift with the scheduler config (base 256 / 0.5, max 6400 / 1.15)."""
    m = (1.15 - 0.5) / (6400 - 256)
    return image_seq_len * m + (0.5 - m * 256)


def schedule(arm: str) -> dict:
    """timesteps and sigmas of `arm`'s sampling schedule, computed exactly as Krea2Pipeline does."""
    key = ("sch", arm)
    if key not in _CACHE:
        from diffusers import FlowMatchEulerDiscreteScheduler
        sch = FlowMatchEulerDiscreteScheduler.from_pretrained(
            snapshot(C.STUDENT_ID, C.STUDENT_REVISION), subfolder="scheduler")
        n = C.STEPS[arm]
        mu = 1.15 if arm == "student" else calculate_mu()
        sch.set_timesteps(sigmas=np.linspace(1.0, 1 / n, n), mu=mu)
        _CACHE[key] = dict(mu=mu, timesteps=[float(t) for t in sch.timesteps],
                           sigmas=[float(s) for s in sch.sigmas])
    return _CACHE[key]


def teacher_target_sigmas(k: int = C.TEACHER_TARGET_STEP) -> dict:
    """Teacher forward k (1-based) reads sigmas[k-1] and its Euler update lands at sigmas[k]; compare with the
    student's sigma after step 1."""
    tsg = schedule("teacher")["sigmas"]
    ssg = schedule("student")["sigmas"]
    return dict(k=k, sigma_in=tsg[k - 1], sigma_out=tsg[k], student_sigma1=ssg[1],
                gap=tsg[k] - ssg[1],
                nearest_k=1 + int(np.argmin([abs(s - ssg[1]) for s in tsg[1:]])))


def euler_step(x, v, s_from: float, s_to: float):
    """FlowMatchEulerDiscreteScheduler.step: x + (sigma_next - sigma) v, in fp32, cast back."""
    return (x.float() + (s_to - s_from) * v.float()).to(x.dtype)


# ------------------------------------------------------------------ ReDiverse-FT LoRA --
FT_ADAPTER = "rediverse"
# Every linear of a single-stream Krea2TransformerBlock: attention q/k/v (k, v: 6144 -> 1536 under GQA), the sigmoid
# output gate, the output projection, and the SwiGLU gate/up/down. text_fusion reuses these names, hence the anchor.
LORA_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate", "attn.to_out.0", "ff.gate", "ff.up", "ff.down")


def lora_target_modules(model, last_block: int) -> list[str]:
    import re
    pat = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
    names = []
    for name, mod in model.named_modules():
        m = pat.match(name)
        if m and int(m.group(1)) <= last_block and m.group(2) in LORA_SUFFIXES and isinstance(mod, torch.nn.Linear):
            names.append(name)
    assert len(names) == (last_block + 1) * len(LORA_SUFFIXES), (len(names), last_block)
    return sorted(names)


def attach_lora(model, last_block: int, rank: int = 32, alpha: int = 32) -> dict:
    """Trainable ReDiverse-FT LoRA (B = 0) on blocks 0..last_block, fp32 on the bf16 base; everything else frozen."""
    from peft import LoraConfig
    targets = lora_target_modules(model, last_block)
    model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets,
                                 init_lora_weights=True), adapter_name=FT_ADAPTER)
    for n, p in model.named_parameters():
        is_ft = "lora_" in n and f".{FT_ADAPTER}." in n
        if is_ft and p.dtype != torch.float32:
            p.data = p.data.float()
        p.requires_grad_(is_ft)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    return dict(n_modules=len(targets), n_tensors=len(params), n_params=sum(p.numel() for p in params),
                blocks=f"0..{last_block}", rank=rank, alpha=alpha, suffixes=list(LORA_SUFFIXES))


def lora_state_dict(model) -> dict:
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()
            if "lora_" in n and f".{FT_ADAPTER}." in n}


def load_lora(model, path: str, last_block: int = None, rank: int = None, alpha: int = None):
    """Attach the adapter and load trained weights (a .safetensors written by krea2.train). rank, alpha (= rank) and
    the last LoRA block default to the values recorded in the file's metadata."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, "pt") as fh:
        md = fh.metadata() or {}
    rank = rank or int(md.get("rank", 32))
    alpha = alpha or rank
    last_block = int(md["lora_last"]) if last_block is None and "lora_last" in md else last_block
    assert last_block is not None and last_block >= 0, "unknown last LoRA block"
    if "lora_last" in md:
        assert int(md["lora_last"]) == last_block, (md, last_block)
    attach_lora(model, last_block, rank, alpha)
    sd = load_file(path)
    res = model.load_state_dict(sd, strict=False)
    own = {n for n, _ in model.named_parameters() if "lora_" in n and f".{FT_ADAPTER}." in n}
    assert not res.unexpected_keys and own == set(sd), (res.unexpected_keys[:3], len(own), len(sd))
    nzB = sum(float(v.abs().sum()) for k, v in sd.items() if "lora_B" in k)
    assert nzB > 0, "adapter has all-zero lora_B"
    model.requires_grad_(False)
    return dict(path=path, n_tensors=len(sd), sum_abs_B=nzB)
