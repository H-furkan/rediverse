"""Qwen-Image loading, text contexts, schedules, forwards and LoRA helpers shared by every qwenimage stage.

Text contexts. A single prompt is never padded by either pipeline, so a context is the prompt's text-encoder states
as the pipeline hands them to the transformer: q21 [1, L, 4096] (Qwen3-VL last decoder layer before its final norm,
system tokens dropped), q20 [1, L, 3584] (Qwen2.5-VL). One cached context serves the teacher and every student of a
family (they share the text encoder; the Turbo text encoder is checked against the base one in check.py).

Forwards. Every stage runs the transformer without the q21 prefix KV cache (the Lightning card does the same): text
rows are recomputed in every forward, which equals the cached path for a frozen model (text tokens are modulated at
t = 0 and attend causally, so they do not depend on the latents or the step) and keeps "ReDiverse-FT active in the
first forward only" exact when the adapter also touches text rows. The transformer reads timestep = bf16(1000 sigma)
/ 1000, as both pipelines compute it.

Students. LoRA students are the base transformer with the card's adapter loaded under the name "student" (unmerged,
scale 1); full students (Qwen-Image-2.1-Turbo) are their own transformer. The ReDiverse-FT LoRA is a second adapter,
"rediverse", stacked on top; `set_active` switches the active set per step.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from qwenimage import config as C

arch = C.arch

_CACHE: dict = {}
STUDENT_ADAPTER = "student"
FT_ADAPTER = "rediverse"


# ------------------------------------------------------------------ loading --
def snapshot(repo: str, revision: str) -> str:
    """Local snapshot folder of a pinned revision. Only the files a stage needs are downloaded (allow_patterns), so
    an incomplete snapshot is expected; huggingface_hub >= 1.3 refuses those offline, hence the direct path."""
    from huggingface_hub import constants
    d = os.path.join(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "snapshots", revision)
    assert os.path.isdir(d), f"{repo}@{revision} not in {constants.HF_HUB_CACHE}"
    return d


def base_dir(fam: str) -> str:
    f = C.family(fam)
    return snapshot(f["repo"], f["revision"])


def pipe_cls(fam: str):
    if C.arch(fam) == "q21":
        from diffusers import QwenImage21Pipeline
        return QwenImage21Pipeline
    from diffusers import QwenImagePipeline
    return QwenImagePipeline


def transformer_cls(fam: str):
    if C.arch(fam) == "q21":
        from diffusers import QwenImage21Transformer2DModel
        return QwenImage21Transformer2DModel
    from diffusers import QwenImageTransformer2DModel
    return QwenImageTransformer2DModel


def load_text_pipe(fam: str, device):
    """The family pipeline with its text encoder and tokenizer/processor only."""
    return pipe_cls(fam).from_pretrained(base_dir(fam), transformer=None, vae=None, torch_dtype=C.DTYPE).to(device)


def who_family(who: str) -> str:
    """'teacher:q21' / 'teacher:q20' or a student name -> family."""
    return who.split(":")[1] if who.startswith("teacher") else C.family_of(who)


def _kohya_prefixed_to_peft(sd: dict) -> dict:
    """'transformer.<module>.lora_down/lora_up.weight' + '.alpha' (the lightx2v v2-RL export, which diffusers'
    converter does not recognise with the 'transformer.' prefix) -> peft keys, alpha / rank folded into lora_B."""
    out = {}
    mods = sorted(k[:-len(".lora_down.weight")] for k in sd if k.endswith(".lora_down.weight"))
    for m in mods:
        down, up = sd[f"{m}.lora_down.weight"], sd[f"{m}.lora_up.weight"]
        r = down.shape[0]
        alpha = float(sd[f"{m}.alpha"]) if f"{m}.alpha" in sd else float(r)
        name = m[len("transformer."):] if m.startswith("transformer.") else m
        out[f"{name}.lora_A.weight"] = down
        out[f"{name}.lora_B.weight"] = up * (alpha / r)
    assert len(out) == 2 * len(mods) and len(sd) == len(out) + sum(k.endswith(".alpha") for k in sd), (len(sd), len(out))
    return out


def load_student_lora(fam: str, model, name: str, path: str):
    """Load a student's LoRA file onto `model` under adapter `name` (diffusers' Qwen-Image LoRA conversion)."""
    from safetensors.torch import load_file
    raw = load_file(path)
    if any(k.startswith("transformer.") and k.endswith(".lora_down.weight") for k in raw):
        sd = _kohya_prefixed_to_peft(raw)
        model.load_lora_adapter(sd, prefix=None, adapter_name=name)
        n_keys = len(sd)
    elif all(k.startswith("transformer_blocks.") or k.split(".")[0] in dict(model.named_children()) for k in raw) \
            and all(".lora_A." in k or ".lora_B." in k for k in raw):
        # bare peft keys without the "transformer." prefix and without alphas (ComfyUI-style, scale 1): the diffusers
        # loader would filter every key out by prefix
        model.load_lora_adapter(dict(raw), prefix=None, adapter_name=name)
        n_keys = len(raw)
    else:
        P = pipe_cls(fam)
        sd, md = P.lora_state_dict(path, return_lora_metadata=True)
        n_keys = len(sd)
        P.load_lora_into_transformer(sd, transformer=model, adapter_name=name, metadata=md)
    mods = [m for _, m in model.named_modules() if hasattr(m, "lora_A") and name in getattr(m, "lora_A", {})]
    n_mod = len(mods)
    assert 2 * n_mod == n_keys, f"{path}: {n_keys} adapter tensors but {n_mod} LoRA modules (unmatched keys)"
    # against the raw file: one down/A tensor per loaded module, and every non-zero up/B tensor arrives non-zero
    import re
    down = re.compile(r"(lora_A|lora_down|lora\.down)\.weight$")
    up = re.compile(r"(lora_B|lora_up|lora\.up)\.weight$")
    n_down = sum(1 for k in raw if down.search(k))
    n_up_nz = sum(1 for k, v in raw.items() if up.search(k) and float(v.float().norm()) > 0)
    n_b_nz = sum(1 for m in mods if float(m.lora_B[name].weight.detach().float().norm()) > 0)
    assert n_down == n_mod, f"{path}: {n_down} down/A tensors in the file but {n_mod} LoRA modules loaded"
    assert n_b_nz == n_up_nz, f"{path}: {n_up_nz} non-zero up/B tensors in the file, {n_b_nz} non-zero after loading"
    return dict(path=path, n_keys=n_keys, n_modules=n_mod, n_file_down=n_down, n_nonzero_B=n_b_nz)


@torch.no_grad()
def merge_student_lora(model, weights: str, config_path: str) -> dict:
    """SMD-style adapter merged into the base weights (their fuse_adapter): W <- (W.to(merge_precision) + scale B A
    rounded to delta_precision) cast back, plus the saved replacement tensors; every tensor must match a parameter."""
    import json
    from safetensors.torch import load_file
    cfg = json.load(open(config_path))
    tensors = load_file(weights)
    params = dict(model.named_parameters())
    pairs, repl, used = {}, {}, set()
    for name, a in tensors.items():
        if ".lora_A" not in name:
            continue
        prefix, suffix = name.split(".lora_A", 1)
        b = tensors[prefix + ".lora_B" + suffix]
        assert a.ndim == 2 and b.ndim == 2 and a.shape[0] == b.shape[1], prefix
        pairs[prefix + ".weight"] = (a, b, cfg.get("scaling", {}).get(prefix, cfg["lora_alpha"] / a.shape[0]))
        used.update((name, prefix + ".lora_B" + suffix))
    for name, v in tensors.items():
        if name not in used:
            assert "lora_" not in name, name
            repl[name] = v
    assert set(repl) == set(cfg.get("saved_replacements", [])), (sorted(repl), cfg.get("saved_replacements"))
    for name in list(pairs) + list(repl):
        shape = (pairs[name][1].shape[0], pairs[name][0].shape[1]) if name in pairs else tuple(repl[name].shape)
        assert name in params and tuple(params[name].shape) == tuple(shape), f"adapter/base mismatch {name}"
    mp = getattr(torch, cfg.get("merge_precision", "bfloat16"))
    dp = cfg.get("delta_precision")
    n_nz = 0
    for name, (a, b, scale) in pairs.items():
        w = params[name]
        delta = (b.to(w.device, torch.float32) @ a.to(w.device, torch.float32)) * scale
        if dp:
            delta = delta.to(getattr(torch, dp)).float()
        n_nz += int(float(delta.abs().max()) > 0)
        w.copy_((w.to(mp).float() + delta).to(w.dtype))
    for name, v in repl.items():
        params[name].copy_(v.to(params[name]))
    return dict(path=weights, merged_pairs=len(pairs), nonzero_deltas=n_nz, replacements=sorted(repl),
                merge_precision=str(mp), delta_precision=dp)


def load_transformer(who: str, device):
    """Frozen bf16 transformer of `who` ('teacher:<fam>' or a student name); LoRA students carry their adapter."""
    fam = who_family(who)
    f = C.family(fam)
    cls = transformer_cls(fam)
    info = {}
    if who.startswith("teacher") or C.student(who)["kind"] in ("lora", "merged_lora"):
        m = cls.from_pretrained(base_dir(fam), subfolder="transformer", torch_dtype=C.DTYPE)
    else:
        repo, sub = C.student_weights(who)
        m = cls.from_pretrained(snapshot(repo, C.student(who)["revision"]), subfolder=sub, torch_dtype=C.DTYPE)
    assert len(m.transformer_blocks) == f["n_blocks"], len(m.transformer_blocks)
    m.requires_grad_(False).eval()
    m = m.to(device)          # device only (keeps any _keep_in_fp32_modules in fp32)
    if not who.startswith("teacher") and C.student(who)["kind"] == "lora":
        s = C.student(who)
        info = load_student_lora(fam, m, STUDENT_ADAPTER, f"{snapshot(s['repo'], s['revision'])}/{s['file']}")
        assert info["n_modules"] > 0, info
        set_active(m, [STUDENT_ADAPTER])
    elif not who.startswith("teacher") and C.student(who)["kind"] == "merged_lora":
        s = C.student(who)
        d = snapshot(s["repo"], s["revision"])
        info = merge_student_lora(m, f"{d}/{s['file']}", f"{d}/{s['adapter_config']}")
    m.requires_grad_(False)
    m._qwen_info = dict(who=who, family=fam, student_lora=info)
    return m


def load_vae(fam: str, device):
    from diffusers import AutoencoderKLQwenImage, AutoencoderKLQwenImage21
    cls = AutoencoderKLQwenImage21 if C.arch(fam) == "q21" else AutoencoderKLQwenImage
    return cls.from_pretrained(base_dir(fam), subfolder="vae", torch_dtype=C.DTYPE).to(device).eval()


# ------------------------------------------------------------------ adapters --
def adapter_names(model) -> list[str]:
    return list(getattr(model, "peft_config", {}) or {})


def set_active(model, names: list[str], scales: dict = None):
    """Activate exactly `names` (scale 1 each, or scales[name]); [] disables every adapter. A LoRA scale s multiplies
    the adapter's output, i.e. it equals lora_B x s."""
    have = adapter_names(model)
    if not have:
        assert not names, names
        return
    if not names:
        model.disable_adapters()
        return
    assert all(n in have for n in names), (names, have)
    model.enable_adapters()
    model.set_adapters(list(names), weights=[float((scales or {}).get(n, 1.0)) for n in names])


def base_adapters(model) -> list[str]:
    """Adapters that define the model itself (the student LoRA), i.e. active in every step."""
    return [n for n in adapter_names(model) if n == STUDENT_ADAPTER]


# ------------------------------------------------------------------ text --
@torch.no_grad()
def encode(pipe, fam: str, prompt: str, device) -> torch.Tensor:
    """Context of one prompt: [1, L, d_text] bf16."""
    if C.arch(fam) == "q21":
        pe, mask, ipm = pipe.encode_prompt(prompt=prompt, device=device)
        assert not bool(ipm.any()), "text-only prompt has image slots"
    else:
        pe, mask = pipe.encode_prompt(prompt=prompt, device=device)
    assert pe.shape[0] == 1, pe.shape
    if mask is not None:
        assert bool(mask.bool().all()), "single prompt should not be padded"
    return pe.to(C.DTYPE).contiguous()


def encode_many(fam: str, prompts: list[str], device, log=print) -> list[torch.Tensor]:
    pipe = load_text_pipe(fam, device)
    out = [encode(pipe, fam, p, device).cpu() for p in prompts]
    del pipe
    torch.cuda.empty_cache()
    log(f"[qwen/{fam}] encoded {len(prompts)} prompts (text rows {min(o.shape[1] for o in out)}..{max(o.shape[1] for o in out)})")
    return out


# ------------------------------------------------------------------ schedules --
def _shift_mu(cfg: dict, n_tokens: int) -> float:
    """calculate_shift as both pipelines call it, from the scheduler config."""
    base_seq, max_seq = cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096)
    base_shift, max_shift = cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return n_tokens * m + (base_shift - m * base_seq)


def make_scheduler(who: str):
    from diffusers import FlowMatchEulerDiscreteScheduler
    fam = who_family(who)
    base = FlowMatchEulerDiscreteScheduler.from_pretrained(base_dir(fam), subfolder="scheduler")
    over = C.family(fam)["teacher"]["scheduler"] if who.startswith("teacher") else C.student(who)["scheduler"]
    return FlowMatchEulerDiscreteScheduler.from_config(base.config, **over) if over else base


def schedule(who: str) -> dict:
    """Sampling schedule of `who` exactly as the pipeline call computes it: dict(sigmas incl. terminal 0, timesteps)."""
    key = ("sch", who)
    if key not in _CACHE:
        fam = who_family(who)
        f = C.family(fam)
        sch = make_scheduler(who)
        if who.startswith("teacher"):
            n = f["teacher"]["steps"]
            raw = np.linspace(1.0, 1 / n, n)
        else:
            raw = np.array(C.student(who)["sigmas"], dtype=np.float64)
        mu = _shift_mu(sch.config, f["n_img"])
        sch.set_timesteps(sigmas=raw, mu=mu)
        _CACHE[key] = dict(mu=mu, raw=[float(s) for s in raw], timesteps=[float(t) for t in sch.timesteps],
                           sigmas=[float(s) for s in sch.sigmas], timesteps_t=sch.timesteps.clone(),
                           sigmas_t=sch.sigmas.clone())
    return _CACHE[key]


def teacher_target_sigmas(student_name: str, k: int = None) -> dict:
    """Teacher forward k (1-based) reads sigmas[k-1] and lands at sigmas[k]; the nearest landing to the student's
    sigma after its first step gives tau."""
    fam = C.family_of(student_name)
    tsg = schedule(f"teacher:{fam}")["sigmas"]
    s1 = schedule(student_name)["sigmas"][1]
    nearest = 1 + int(np.argmin([abs(s - s1) for s in tsg[1:]]))
    k = nearest if k is None else k
    return dict(k=k, sigma_in=tsg[k - 1], sigma_out=tsg[k], student_sigma1=s1, gap=tsg[k] - s1, nearest_k=nearest)


def model_timestep(who: str, j: int, b: int, device) -> torch.Tensor:
    """The pipelines' `timestep / 1000` with timestep = scheduler.timesteps[j] cast to the latent dtype (bf16)."""
    t = schedule(who)["timesteps_t"][j].to(device)
    return t.expand(b).to(C.DTYPE) / 1000


def euler_step(x, v, s_from: float, s_to: float):
    """Euler update fully in fp32: x + (sigma_next - sigma) v, cast back (the krea2 convention)."""
    return (x.float() + (s_to - s_from) * v.float()).to(x.dtype)


def euler_step_sched(x, v, who: str, j: int):
    """FlowMatchEulerDiscreteScheduler.step exactly (diffusers 0.43): sample upcast to fp32, dt = sigmas[j+1] -
    sigmas[j] as a 0-dim fp32 tensor, `sample + dt * model_output` (the product stays bf16 under type promotion),
    cast to the model output's dtype."""
    sg = schedule(who)["sigmas_t"]
    dt = (sg[j + 1] - sg[j]).to(x.device)
    return (x.to(torch.float32) + dt * v).to(v.dtype)


EULER = os.environ.get("QWEN_EULER", "scheduler")     # "scheduler" (pipeline-exact) | "fp32"


def euler_mode(who: str) -> str:
    """The Euler arithmetic of `who`: a student's own (config 'euler', e.g. SMD's bf16 update) or model.EULER."""
    if not who.startswith("teacher") and C.student(who).get("euler"):
        return C.student(who)["euler"]
    return EULER


def provenance(who: str, guided: bool = True) -> dict:
    """What a bank / direction / sample was produced with: the Euler arithmetic and the guidance actually applied."""
    fam = who_family(who)
    if who.startswith("teacher"):
        t = C.family(fam)["teacher"]
        cfg = float(t["cfg"]) if guided else 1.0
        neg = t.get("negative") if guided and cfg > 1 else None
    else:
        cfg, neg = float(C.student(who)["cfg"]), None
    return dict(euler=euler_mode(who), cfg=cfg, negative=neg)


def check_provenance(rec: dict, want: dict, what: str):
    """Assert a stored record's euler / cfg / negative equal the current run's."""
    got = {k: rec.get(k, "<missing>") for k in want}
    assert got == want, f"{what}: stored {got} but this run uses {want}"


def step(x, v, who: str, j: int, mode: str = None):
    """One sampling step from forward j of `who`'s schedule, in the configured arithmetic."""
    mode = mode or euler_mode(who)
    if mode == "scheduler":
        return euler_step_sched(x, v, who, j)
    if mode == "bf16":              # sample + v * dt entirely in bf16 (SMD's StudentEulerScheduler)
        sg = schedule(who)["sigmas_t"]
        dt = (sg[j + 1] - sg[j]).to(x.device).to(x.dtype)
        return (x.to(torch.bfloat16) + v.to(torch.bfloat16) * dt).to(x.dtype)
    sg = schedule(who)["sigmas"]
    return euler_step(x, v, sg[j], sg[j + 1])


def velocity(fam: str, model, x, ctx, t, neg=None, cfg: float = 1.0):
    """The sampler's velocity at one step: conditional, or Qwen-Image true CFG with the pipeline's norm rescale
    (comb = v_u + cfg (v_c - v_u); v = comb * |v_c| / |comb| per token) when a negative context and cfg > 1 are given."""
    v = forward(fam, model, x, ctx, t)
    if neg is None or cfg <= 1:
        return v
    vu = forward(fam, model, x, neg, t)
    comb = vu + cfg * (v - vu)
    if arch(fam) != "q20":           # QwenImage21Pipeline: plain true CFG, no norm rescale
        return comb
    return comb * (torch.norm(v, dim=-1, keepdim=True) / torch.norm(comb, dim=-1, keepdim=True))


@torch.no_grad()
def rollout(who: str, model, x, ctx, n: int, neg=None, cfg: float = 1.0, mode: str = None):
    """Latents after the first n sampling steps of `who`'s schedule from x (teacher rollouts of the bank and the
    TF directions; the generator has its own loop for the FT/TF arms)."""
    fam = who_family(who)
    for j in range(n):
        t = model_timestep(who, j, x.shape[0], x.device)
        x = step(x, velocity(fam, model, x, ctx, t, neg, cfg), who, j, mode)
    return x



# ------------------------------------------------------------------ geometry --
def q21_img_mask(text_len: int, device) -> torch.Tensor:
    """The pipeline's image_pad_mask after append_target_slots: L text slots, then one slot per 2x2 target tokens."""
    f = C.family("q21")
    return torch.cat([torch.zeros(1, text_len, dtype=torch.bool, device=device),
                      torch.ones(1, f["n_img"] // 4, dtype=torch.bool, device=device)], 1)


def img_shapes(fam: str, b: int):
    g = C.family(fam)["grid"]
    return [[(1, g, g)]] * b


def _expand_ctx(ctx, b, x):
    ctx = ctx.to(x.device, x.dtype)
    return ctx.expand(b, -1, -1) if ctx.shape[0] == 1 and b > 1 else ctx


def forward(fam: str, model, x: torch.Tensor, ctx: torch.Tensor, t: torch.Tensor):
    """Velocity of the target latents [B, 4096, 64] for context ctx [1 or B, L, d_text] at model timestep t [B]."""
    b = x.shape[0]
    ctx = _expand_ctx(ctx, b, x)
    if C.arch(fam) == "q21":
        out = model(hidden_states=x, timestep=t, encoder_hidden_states=ctx, encoder_hidden_states_mask=None,
                    img_shapes=img_shapes(fam, b), img_mask=q21_img_mask(ctx.shape[1], x.device),
                    return_dict=False)[0]
        return out[:, -x.shape[1]:]
    return model(hidden_states=x, encoder_hidden_states=ctx, encoder_hidden_states_mask=None, timestep=t,
                 img_shapes=img_shapes(fam, b), return_dict=False)[0]


def extract(fam: str, name: str, out) -> torch.Tensor:
    """Image rows of a block output: q21 tensor [B, L + 4096, D] -> last 4096 rows; q20 (text, image) -> image."""
    f = C.family(fam)
    if C.arch(fam) == "q20":
        assert isinstance(out, tuple) and len(out) == 2, (name, type(out))
        out = out[1]
    assert torch.is_tensor(out) and out.shape[-1] == f["d_model"] and out.shape[1] >= f["n_img"], (name, out.shape)
    return out[:, -f["n_img"]:]


def add_to_image_rows(fam: str, out, delta: torch.Tensor):
    """Block output with delta [B, 4096, D] added to its image rows (forward-hook return value)."""
    n = C.family(fam)["n_img"]
    if C.arch(fam) == "q20":
        txt, img = out
        return txt, img + delta.to(img.dtype)
    o = out.clone()
    o[:, -n:] += delta.to(out.dtype)
    return o


def forward_to_blocks(fam: str, model, x, ctx, t, blocks, checkpoint: bool = False) -> dict:
    """{b: image rows [B, 4096, D]} of the requested blocks, running only up to max(blocks). Mirrors the diffusers
    0.43 forward (no KV cache, no text mask); verified against hooks on the full forward by check_truncation()."""
    from torch.utils.checkpoint import checkpoint as ckpt
    blocks = sorted(set(int(b) for b in blocks))
    b = x.shape[0]
    ctx = _expand_ctx(ctx, b, x)
    n = C.family(fam)["n_img"]
    out = {}
    use_ckpt = checkpoint and torch.is_grad_enabled()
    if C.arch(fam) == "q21":
        from diffusers.models.transformers.transformer_qwenimage21 import _qwenimage21_prefix_segments
        hs = model.img_in(x)
        ehs = model.txt_in(ctx)
        img_mask = q21_img_mask(ctx.shape[1], x.device)
        repeats = torch.where(img_mask, 4, 1)[0]
        ipm = torch.repeat_interleave(img_mask[0], repeats)
        joint = torch.cat([ehs, hs], dim=1)           # text slots then the target image (no condition images)
        shapes = img_shapes(fam, b)
        rot = model.pos_embed(shapes[0], ipm, device=x.device)
        image_ids, ttm = model.build_token_metadata(ipm, shapes[0])
        tt = t.to(hs.dtype)
        tt = torch.cat([tt, tt.new_zeros(1)], dim=0)
        modulation = model.modulation(model.time_text_embed(tt, hs))
        segments = _qwenimage21_prefix_segments(image_ids, int((~ttm).sum()))
        for i, blk in enumerate(model.transformer_blocks[:blocks[-1] + 1]):
            if use_ckpt:
                joint = ckpt(blk, joint, modulation, rot, None, ttm, None, None, None, segments, None, use_reentrant=False)
            else:
                joint = blk(hidden_states=joint, modulation=modulation, rotary_emb=rot, attention_mask=None,
                            target_token_mask=ttm, segments=segments)
            if i in blocks:
                out[i] = joint[:, -n:]
        return out
    from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask
    hs = model.img_in(x)
    tt = t.to(hs.dtype)
    assert not getattr(model, "zero_cond_t", False)
    ehs = model.txt_in(model.txt_norm(ctx))
    text_len, _, mask = compute_text_seq_len_from_mask(ehs, None)
    temb = model.time_text_embed(tt, hs, None)
    rot = model.pos_embed(img_shapes(fam, b), max_txt_seq_len=text_len, device=x.device)
    for i, blk in enumerate(model.transformer_blocks[:blocks[-1] + 1]):
        if use_ckpt:
            ehs, hs = ckpt(blk, hs, ehs, mask, temb, rot, None, None, use_reentrant=False)
        else:
            ehs, hs = blk(hidden_states=hs, encoder_hidden_states=ehs, encoder_hidden_states_mask=mask, temb=temb,
                          image_rotary_emb=rot)
        if i in blocks:
            out[i] = hs
    return out


@torch.no_grad()
def check_truncation(fam: str, model, ctx, device, blocks=None) -> float:
    """Max relative difference between forward_to_blocks and hooks on the full forward (expect 0)."""
    f = C.family(fam)
    blocks = blocks or (0, f["n_blocks"] // 2, f["n_blocks"] - 1)
    x = torch.randn(2, *f["latent_shape"], generator=torch.Generator("cpu").manual_seed(0)).to(device, C.DTYPE)
    t = torch.full((2,), 0.9, device=device, dtype=C.DTYPE)
    got = {}
    hs = [model.transformer_blocks[bk].register_forward_hook(
        lambda _m, _i, o, _b=bk: got.__setitem__(_b, extract(fam, f"b{_b}", o).clone())) for bk in blocks]
    try:
        forward(fam, model, x, ctx, t)
    finally:
        for h in hs:
            h.remove()
    mine = forward_to_blocks(fam, model, x, ctx, t, blocks)
    return max(float((mine[bk].float() - got[bk].float()).norm() / got[bk].float().norm()) for bk in blocks)


# ------------------------------------------------------------------ decoding --
class Decoder:
    def __init__(self, fam: str, device):
        self.fam, self.device = fam, device
        self.vae = load_vae(fam, device)
        zc = self.vae.config.z_dim
        self.mean = torch.tensor(self.vae.config.latents_mean).view(1, zc, 1, 1, 1).to(device, C.DTYPE)
        self.std = torch.tensor(self.vae.config.latents_std).view(1, zc, 1, 1, 1).to(device, C.DTYPE)
        f = C.family(fam)
        self.scale = f["vae_scale"]

    @torch.no_grad()
    def __call__(self, x):
        from PIL import Image
        P = pipe_cls(self.fam)
        lat = P._unpack_latents(x.to(C.DTYPE), C.RES, C.RES, self.scale if C.arch(self.fam) == "q21" else 8)
        lat = lat.to(self.vae.dtype) * self.std + self.mean
        out = []
        for k in range(lat.shape[0]):
            img = self.vae.decode(lat[k:k + 1], return_dict=False)[0][:, :, 0]
            arr = (img[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            out.append(Image.fromarray((arr * 255).round().astype("uint8")))
        return out


# ------------------------------------------------------------------ ReDiverse-FT LoRA --
LORA_SUFFIXES = {
    "q21": ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"),
    # q20 dual-stream: image and text attention projections and both MLPs (no modulation layers)
    "q20": ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj",
            "attn.add_v_proj", "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj",
            "txt_mlp.net.2"),
}


def lora_target_modules(fam: str, model, last_block: int) -> list[str]:
    import re
    pat = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
    suf = LORA_SUFFIXES[C.arch(fam)]
    names = []
    for name, mod in model.named_modules():
        m = pat.match(name)
        if m and int(m.group(1)) <= last_block and m.group(2) in suf:
            base = getattr(mod, "base_layer", mod)
            if isinstance(base, torch.nn.Linear):
                names.append(name)
    assert len(names) == (last_block + 1) * len(suf), (len(names), last_block)
    return sorted(names)


def _freeze_except_ft(model):
    for n, p in model.named_parameters():
        is_ft = "lora_" in n and f".{FT_ADAPTER}." in n
        if is_ft and p.dtype != torch.float32:
            p.data = p.data.float()
        p.requires_grad_(is_ft)


def attach_lora(fam: str, model, last_block: int, rank: int = 32, alpha: int = 32) -> dict:
    """Trainable ReDiverse-FT LoRA (B = 0) on blocks 0..last_block, fp32, stacked on any student adapter."""
    from peft import LoraConfig
    targets = lora_target_modules(fam, model, last_block)
    model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets,
                                 init_lora_weights=True), adapter_name=FT_ADAPTER)
    set_active(model, base_adapters(model) + [FT_ADAPTER])
    _freeze_except_ft(model)
    params = [p for p in model.parameters() if p.requires_grad]
    return dict(n_modules=len(targets), n_tensors=len(params), n_params=sum(p.numel() for p in params),
                blocks=f"0..{last_block}", rank=rank, alpha=alpha, suffixes=list(LORA_SUFFIXES[C.arch(fam)]))


def lora_state_dict(model) -> dict:
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()
            if "lora_" in n and f".{FT_ADAPTER}." in n}


def load_lora(fam: str, model, path: str) -> dict:
    """Attach the ReDiverse-FT adapter and load trained weights written by qwenimage.train (metadata: lora_last, rank)."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, "pt") as fh:
        md = fh.metadata() or {}
    rank = int(md.get("rank", 32))
    last = int(md["lora_last"])
    attach_lora(fam, model, last, rank, rank)
    sd = load_file(path)
    res = model.load_state_dict(sd, strict=False)
    own = {n for n, _ in model.named_parameters() if "lora_" in n and f".{FT_ADAPTER}." in n}
    assert not res.unexpected_keys and own == set(sd), (res.unexpected_keys[:3], len(own), len(sd))
    assert sum(float(v.abs().sum()) for k, v in sd.items() if "lora_B" in k) > 0, "all-zero lora_B"
    model.requires_grad_(False)
    set_active(model, base_adapters(model))
    return dict(path=path, n_tensors=len(sd), lora_last=last, rank=rank)
