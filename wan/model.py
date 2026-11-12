"""Wan 2.1 T2V-14B teacher and lightx2v 4-step student: loading, text contexts, samplers, decode.

Shared by every stage of the wan package. The training-side stages (target bank, ReDiverse-FT
training) run in the `train` environment, the rest (text contexts, generation, ReDiverse-TF
directions) in the `video` environment; see wan/README.md.

Dtype policy. `_keep_in_fp32_modules` is neutralised while loading, so every floating parameter
of the teacher and of the student is bfloat16. The model builds its RoPE tables as float32
buffers in __init__; they are left alone (never blanket-`.to(dtype)` this model). Latents are
carried in float32 and cast per call, and the timestep tensor stays float32.

Student. The lightx2v repository's config.json describes the I2V model (in_channels 36); the
student is therefore built from the teacher's T2V transformer config and loaded from
distill_models/distill_model.safetensors, whose native Wan keys diffusers'
convert_wan_transformer_to_diffusers renames one to one (the key set is asserted).
"""
from __future__ import annotations

import re

import torch

from wan import config as C

DTYPE = torch.bfloat16

# Negative prompt of the evaluation teacher's CFG: the English negative prompt of the
# WanPipeline docstring example in diffusers.
NEG_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards")


# ------------------------------------------------------------------ hub files --

def teacher_dir(*subfolders: str, files: tuple = ()) -> str:
    """Local snapshot of the teacher repository at the pinned revision, with `subfolders` (and
    single `files`) downloaded into the user's Hugging Face cache if they are not there yet."""
    from huggingface_hub import constants, snapshot_download
    return snapshot_download(C.TEACHER_REPO, revision=C.TEACHER_REVISION,
                             allow_patterns=[f"{s}/*" for s in subfolders] + list(files),
                             local_files_only=constants.HF_HUB_OFFLINE)


def student_file() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C.STUDENT_REPO, C.STUDENT_FILE, revision=C.STUDENT_REVISION)


# --------------------------------------------------------------------- models --

def _load_student():
    """Teacher T2V config skeleton + the converted lightx2v state dict, key set asserted."""
    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
    from safetensors.torch import load_file

    cfg = WanTransformer3DModel.load_config(teacher_dir(files=("transformer/config.json",)),
                                            subfolder="transformer")
    assert cfg["in_channels"] == 16, f"in_channels {cfg['in_channels']} != 16: not the T2V config"
    # parameters on meta (materialised from the checkpoint below); the RoPE buffers are built for real
    with init_empty_weights(include_buffers=False):
        model = WanTransformer3DModel.from_config(cfg)

    sd = load_file(student_file())
    assert len(sd) == C.STUDENT_N_TENSORS, f"{len(sd)} tensors in the student file, want {C.STUDENT_N_TENSORS}"
    dts = {v.dtype for v in sd.values()}
    assert dts == {DTYPE}, f"student dtypes {dts}, want {{{DTYPE}}}"
    sd = convert_wan_transformer_to_diffusers(sd)
    want, got = set(model.state_dict().keys()), set(sd.keys())
    assert got == want and len(got) == C.STUDENT_N_TENSORS, \
        f"key sets differ: student-only {sorted(got - want)[:3]}, model-only {sorted(want - got)[:3]}"
    model.load_state_dict(sd, strict=True, assign=True)
    return model


def load_transformer(arm: str, device):
    """Frozen bf16 WanTransformer3DModel in eval mode on `device`.

    arm  "teacher"       Wan2.1-T2V-14B base
         "teacher_cfgd"  base + the CFG5->CFG1 distillation LoRA (wan.cfgd_teacher), the
                         ReDiverse-FT target teacher
         "student"       lightx2v 4-step StepDistill-CfgDistill
    """
    from diffusers import WanTransformer3DModel

    saved = WanTransformer3DModel._keep_in_fp32_modules
    WanTransformer3DModel._keep_in_fp32_modules = None          # every parameter in bf16
    try:
        if arm in ("teacher", "teacher_cfgd"):
            model = WanTransformer3DModel.from_pretrained(
                teacher_dir("transformer"), subfolder="transformer", torch_dtype=DTYPE)
            if arm == "teacher_cfgd":
                from wan import cfgd_teacher
                cfgd_teacher.attach(model, DTYPE)
        elif arm == "student":
            model = _load_student()
        else:
            raise ValueError(arm)
    finally:
        WanTransformer3DModel._keep_in_fp32_modules = saved

    bad = sorted(n for n, p in model.named_parameters() if p.dtype != DTYPE)
    assert not bad, f"non-{DTYPE} parameters: {bad[:4]}"
    assert len(model.blocks) == C.N_BLOCKS, f"{len(model.blocks)} blocks, want {C.N_BLOCKS}"
    return model.to(device).eval().requires_grad_(False)


def attach_adapter(model, state_dict: dict, adapter_name: str = "default", alpha=None):
    """Add the ReDiverse-FT LoRA (released safetensors keys "<module>.lora_{A,B}.weight") to the
    student in place. Rank and target modules are read off the tensors; alpha is the file's
    metadata value (default: the rank; the released file has alpha = rank = 32). peft creates the
    adapter layers in the base dtype, so the float32 weights are used in bf16."""
    from peft import LoraConfig

    sd = {re.sub(r"\.lora_([AB])\.weight$", rf".lora_\1.{adapter_name}.weight", k): v
          for k, v in state_dict.items()}
    ranks = {min(v.shape) for v in sd.values()}
    assert len(ranks) == 1, f"mixed LoRA ranks: {sorted(ranks)}"
    rank = ranks.pop()
    alpha = rank if alpha is None else int(alpha)
    targets = sorted({re.sub(r"\.lora_[AB]\..*", "", k) for k in sd})
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets), adapter_name=adapter_name)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected adapter keys: {list(unexpected)[:3]}"
    loaded = [k for k in sd if k in dict(model.named_parameters())]
    assert len(loaded) == len(sd), f"only {len(loaded)}/{len(sd)} adapter tensors matched"
    absent = [k for k in missing if f".{adapter_name}." in k]
    assert not absent, f"{len(absent)} adapter tensors missing from the file, e.g. {absent[:3]}"
    nz = sum(1 for k, v in sd.items() if "lora_B" in k and float(v.abs().sum()) > 0)
    assert nz > 0, "every lora_B is zero: this adapter is untrained"
    print(f"[wan] ReDiverse-FT adapter: r={rank}, alpha={alpha}, {len(targets)} modules, {len(sd)} tensors, "
          f"{nz} non-zero lora_B", flush=True)
    return model.eval().requires_grad_(False)


# ------------------------------------------------------------------- text --

def text_encoder(device):
    """(T5Tokenizer, UMT5EncoderModel in bf16) of the teacher repository."""
    from transformers import T5Tokenizer, UMT5EncoderModel

    path = teacher_dir("tokenizer", "text_encoder")
    tok = T5Tokenizer.from_pretrained(path, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=DTYPE)
    return tok, enc.to(device).eval().requires_grad_(False)


def embed(tok, enc, text: str, device):
    """One UMT5 context in the WanPipeline convention: prompt_clean, max_length 512 with special
    tokens, encoder(ids, mask), positions past the prompt zeroed. -> ([1, 512, 4096] bf16, seq_len)."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    inputs = tok([prompt_clean(text)], padding="max_length", max_length=C.TEXT_LEN,
                 truncation=True, add_special_tokens=True,
                 return_attention_mask=True, return_tensors="pt")
    ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    seq_len = int(mask.gt(0).sum())
    with torch.no_grad():
        emb = enc(ids, mask).last_hidden_state.to(DTYPE)       # [1, 512, 4096]
    emb[:, seq_len:] = 0
    return emb, seq_len


def load_ctx(path: str) -> dict:
    """A saved context dict; encoder_hidden_states is [1, 512, 4096] bf16."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    assert d["encoder_hidden_states"].shape == (1, C.TEXT_LEN, C.TEXT_DIM), d["encoder_hidden_states"].shape
    return d


# -------------------------------------------------------------- schedules --

def student_sigmas(steps: int = C.STUDENT_STEPS) -> list[float]:
    """The student's sigma path including the terminal 0: a shift-5 flow schedule over `steps`
    even intervals, [1.0, 0.9375, 0.8333, 0.625, 0.0] at 4 steps (the checkpoint's
    denoising_step_list [1000, 750, 500, 250])."""
    b = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    s = C.STUDENT_SHIFT * b / (1.0 + (C.STUDENT_SHIFT - 1.0) * b)
    assert float(s[0]) == 1.0 and float(s[-1]) == 0.0 and torch.all(s[1:] < s[:-1]), "bad sigma path"
    return [float(v) for v in s]


def teacher_scheduler(device=None, flow_shift: float = None, steps: int = C.TEACHER_STEPS):
    """The teacher's UniPCMultistepScheduler set to `steps` steps from index 0.

    flow_shift None keeps the repository config (3.0, the base teacher's contract);
    the CFG-distilled target teacher samples at C.CFGD_FLOW_SHIFT = 5.0. UniPC is a multistep
    solver with internal history, so each rollout uses a fresh instance stepped in order."""
    from diffusers import UniPCMultistepScheduler

    kw = {} if flow_shift is None else {"flow_shift": flow_shift}
    sch = UniPCMultistepScheduler.from_pretrained(teacher_dir("scheduler"), subfolder="scheduler", **kw)
    sch.set_timesteps(steps, device=device)
    sch.set_begin_index(0)
    assert abs(float(sch.sigmas[0]) - 1.0) < 1e-5, "sigma0 is not 1: the unit-noise latent is not this sampler's start"
    return sch


def teacher_target_sigmas(k: int, flow_shift: float = None) -> dict:
    """Where teacher forward k (one-based) sits: it reads the latent at sigmas[k-1] and its
    update lands at sigmas[k]; matched against the student's sigma after its first step."""
    sch = teacher_scheduler(None, flow_shift)
    ts = [float(t) for t in sch.timesteps]
    sg = [float(s) for s in sch.sigmas]
    assert 1 <= k <= len(ts), f"teacher forward {k} out of range 1..{len(ts)}"
    st = student_sigmas()
    s_out = sg[k]
    return dict(k=k, t=ts[k - 1], sigma_in=sg[k - 1], sigma_out=s_out,
                sigma_out_err_vs_student=min(abs(s_out - v) for v in st),
                nearest_student_sigma=min(st, key=lambda v: abs(s_out - v)),
                student_sigmas=st)


# --------------------------------------------------------------- forwards --

def call(model, lat, emb, t_val: float, device):
    """One denoiser call: the fp32 latent cast to bf16, a float32 timestep, the context
    expanded to the batch. Returns the bf16 velocity."""
    b = lat.shape[0]
    t = torch.full((b,), float(t_val), dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(hidden_states=lat.to(DTYPE), timestep=t,
                     encoder_hidden_states=emb.expand(b, -1, -1), return_dict=False)[0]


# Adapted from huggingface/diffusers WanTransformer3DModel.forward (Apache-2.0,
# Copyright 2025 The Wan Team and The HuggingFace Team).
def _prologue(model, hidden_states, timestep, encoder_hidden_states):
    """WanTransformer3DModel.forward up to the first block (assert_truncation_matches checks it)."""
    rotary_emb = model.rope(hidden_states)
    hidden_states = model.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)
    assert timestep.ndim == 1, f"timestep.ndim={timestep.ndim}: expected one timestep per sample"
    temb, timestep_proj, encoder_hidden_states, ehs_image = model.condition_embedder(
        timestep, encoder_hidden_states, None, timestep_seq_len=None)
    assert ehs_image is None, "image conditioning in a T2V model"
    timestep_proj = timestep_proj.unflatten(1, (6, -1))     # batch, 6, inner_dim
    return hidden_states, encoder_hidden_states, timestep_proj, rotary_emb


def forward_to_block(model, hidden_states, timestep, encoder_hidden_states, n_blocks: int,
                     checkpoint: bool = True):
    """The prologue and blocks 0..n_blocks-1; returns the output of block n_blocks-1.

    With checkpoint=True (and grad enabled) every block runs under torch.utils.checkpoint with
    use_reentrant=False, whose forward runs in grad mode, so a forward hook on the collapsed block
    captures a graph-connected tensor."""
    assert 1 <= n_blocks <= len(model.blocks), n_blocks
    h, ehs, temb, rope = _prologue(model, hidden_states, timestep, encoder_hidden_states)
    use_ckpt = checkpoint and torch.is_grad_enabled()
    for block in model.blocks[:n_blocks]:
        if use_ckpt:
            h = torch.utils.checkpoint.checkpoint(block, h, ehs, temb, rope, use_reentrant=False)
        else:
            h = block(h, ehs, temb, rope)
    return h


def assert_truncation_matches(model, hidden_states, timestep, encoder_hidden_states, n_blocks: int) -> float:
    """Gate: forward_to_block must reproduce, bit for bit, block n_blocks-1's output inside the
    model's own forward. Returns the max absolute difference (0)."""
    captured = {}
    handle = model.blocks[n_blocks - 1].register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("h", o if not isinstance(o, tuple) else o[0]))
    try:
        with torch.no_grad():
            model(hidden_states=hidden_states, timestep=timestep,
                  encoder_hidden_states=encoder_hidden_states, return_dict=False)
    finally:
        handle.remove()
    with torch.no_grad():
        h_trunc = forward_to_block(model, hidden_states, timestep, encoder_hidden_states, n_blocks,
                                   checkpoint=False)
    diff = (captured["h"].float() - h_trunc.float()).abs().max().item()
    assert diff == 0.0, (f"the truncated forward disagrees with model.forward at blocks[{n_blocks - 1}]: "
                         f"max|diff| = {diff:.3e}; re-derive _prologue against the installed diffusers")
    return diff


@torch.no_grad()
def teacher_block_output(model, lat, emb, device, k: int, block: int = C.ANCHOR_BLOCK,
                         flow_shift: float = None, truncate_last: bool = False) -> torch.Tensor:
    """blocks[block] output at forward k of the teacher's own conditional-only UniPC-50 rollout
    from latent `lat` [B, 16, 21, 60, 104] (fp32). Forwards 1..k-1 are full forwards whose
    velocity advances the latent; forward k is hooked. truncate_last=True stops forward k after
    the hooked block (the target bank); the activation is the same either way."""
    sch = teacher_scheduler(device, flow_shift)
    grab = {}
    handle = model.blocks[block].register_forward_hook(lambda _m, _i, o: grab.__setitem__("h", o))
    x = lat.to(device, torch.float32).clone()
    try:
        for i in range(k):
            t = sch.timesteps[i]
            if i == k - 1 and truncate_last:
                forward_to_block(model, x.to(DTYPE),
                                 torch.full((x.shape[0],), float(t), dtype=torch.float32, device=device),
                                 emb.expand(x.shape[0], -1, -1), block + 1, checkpoint=False)
                break
            v = call(model, x, emb, float(t), device)
            if i == k - 1:
                break
            x = sch.step(v, t, x, return_dict=False)[0]
    finally:
        handle.remove()
    return grab["h"].detach()


# --------------------------------------------------------------- samplers --

@torch.no_grad()
def sample_teacher(model, lat, emb, neg, device, steps: int = C.TEACHER_STEPS,
                   guidance: float = C.TEACHER_GUIDANCE):
    """Teacher sampler: repo UniPC schedule (flow shift 3), CFG v = vu + w (v - vu) on the bf16
    velocities, fp32 latent update."""
    sch = teacher_scheduler(device, None, steps)
    for t in sch.timesteps:
        v = call(model, lat, emb, float(t), device)
        vu = call(model, lat, neg, float(t), device)
        v = vu + guidance * (v - vu)
        lat = sch.step(v, t, lat, return_dict=False)[0]
    return lat


@torch.no_grad()
def sample_student(model, lat, emb, device, steps: int = C.STUDENT_STEPS, step_hook=None,
                   adapter_gate: bool = False):
    """Student sampler: flow Euler over student_sigmas, timestep sigma * 1000, guidance 1.

    step_hook(k) runs before step k (0-based); adapter_gate disables the model's adapters from
    the second step on (ReDiverse-FT is active at the first step only; the caller enables them
    before each sample)."""
    sig = student_sigmas(steps)
    for k in range(steps):
        if step_hook is not None:
            step_hook(k)
        if adapter_gate and k == 1:
            model.disable_adapters()
        v = call(model, lat, emb, sig[k] * 1000.0, device)
        lat = lat + (sig[k + 1] - sig[k]) * v.float()
    return lat


_VAE: dict = {}


def vae(device):
    """AutoencoderKLWan in float32, loaded once per device."""
    v = _VAE.get(str(device))
    if v is None:
        from diffusers import AutoencoderKLWan
        v = AutoencoderKLWan.from_pretrained(teacher_dir("vae"), subfolder="vae", torch_dtype=torch.float32)
        _VAE[str(device)] = v = v.to(device).eval().requires_grad_(False)
    return v


def decode(lat, device):
    """Latents -> uint8 [B, T, H, W, 3]: z * latents_std + latents_mean, VAE decode,
    ((x / 2 + 0.5).clamp(0, 1) * 255).round()."""
    m_vae = vae(device)
    cfg = m_vae.config
    z = cfg.z_dim
    assert lat.shape[1] == z == len(cfg.latents_mean) == len(cfg.latents_std)
    m = torch.tensor(cfg.latents_mean, dtype=torch.float32, device=device).view(1, z, 1, 1, 1)
    s = torch.tensor(cfg.latents_std, dtype=torch.float32, device=device).view(1, z, 1, 1, 1)
    with torch.no_grad():
        vid = m_vae.decode((lat.float() * s + m).to(m_vae.dtype), return_dict=False)[0]   # [B, 3, T, H, W]
    vid = ((vid.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
    return vid.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
