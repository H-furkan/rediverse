"""The SDXL teacher's own 50-step EulerDiscrete trajectory, hooked at the collapsed block.

One implementation serves both uses of a teacher activation at step tau:

  * the ReDiverse-FT target (training, fastgen_overlay): guidance=8.0, the teacher rolled with
    classifier-free guidance for forwards 1..tau-1 and the conditional activation of forward tau;
  * the ReDiverse-TF directions (sdxl/estimate_directions.py): guidance=1.0, a conditional-only
    rollout.

Conventions:

  * x0 is unit-variance noise. The rollout multiplies it by the scheduler's init_noise_sigma, as
    diffusers' prepare_latents does; SDXL's Euler config (timestep_spacing="leading") makes
    scale_model_input divide it back out, so the UNet's first input is x0 itself, the same tensor
    the student's step-1 forward receives.
  * step k is the k-th UNet forward, 1-indexed: step 1 runs at timesteps[0] (t=981 under the
    50-step leading schedule), step 7 at t=861. Steps 1..k-1 are run in full, because each one
    advances the latent; the velocity of step k is never used.
  * guided steps: eps_u + w (eps_c - eps_u), conditional forward first, unconditional branch =
    zero prompt and pooled embeddings with the same time ids (SDXL's force_zeros_for_empty_prompt
    convention, as used by the teacher arm). The hooked activation is the conditional one.
  * the hook takes o[0] for a module that returns a 1-tuple (Transformer2DModel inside the UNet).
"""
from __future__ import annotations

import inspect
import threading

import torch

from sdxl.config import SDXL_ID, SDXL_REVISION, TEACHER_STEPS

_CFG_CACHE: dict = {}
_CFG_LOCK = threading.Lock()


def scheduler_config(model_id: str = SDXL_ID):
    """The repository's EulerDiscrete config, loaded once per model id (from_config on the same
    config gives the same sigmas)."""
    with _CFG_LOCK:
        cfg = _CFG_CACHE.get(model_id)
        if cfg is None:
            from diffusers import EulerDiscreteScheduler
            revision = SDXL_REVISION if model_id == SDXL_ID else None
            cfg = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", revision=revision).config
            _CFG_CACHE[model_id] = cfg
    return cfg


def euler_scheduler(device=None, steps: int = TEACHER_STEPS, model_id: str = SDXL_ID):
    """A fresh scheduler (set_timesteps and step_index are mutable state)."""
    from diffusers import EulerDiscreteScheduler
    sch = EulerDiscreteScheduler.from_config(scheduler_config(model_id))
    sch.set_timesteps(steps, device=device)
    return sch


def _grab(o):
    return (o[0] if isinstance(o, tuple) else o).detach()


def _model_out(o):
    """The eps prediction: UNet2DConditionOutput.sample, or the bare tensor returned by
    FastGen's classify_forward."""
    if hasattr(o, "sample"):
        return o.sample
    return o[0] if isinstance(o, (tuple, list)) else o


def _unet_extra_kwargs(unet) -> dict:
    """FastGen's classify_forward needs feature_indices to be a set; plain diffusers UNets do not
    take the argument."""
    try:
        params = inspect.signature(unet.forward).parameters
    except (TypeError, ValueError):
        return {}
    return {"feature_indices": set()} if "feature_indices" in params else {}


def _rows(t: torch.Tensor, b: int, device, dtype) -> torch.Tensor:
    t = t.to(device, dtype)
    if t.shape[0] == b:
        return t
    assert t.shape[0] == 1, f"conditioning has batch {t.shape[0]}, expected 1 or {b}"
    return t.expand(b, *t.shape[1:])


@torch.no_grad()
def teacher_states(unet, modules: dict, x0: torch.Tensor, cond: dict, kmax: int,
                   *, keep=None, steps: int = TEACHER_STEPS, model_id: str = SDXL_ID,
                   dtype=None, device=None, guidance: float = 1.0) -> dict:
    """{k: {name: activation of `modules[name]` at the teacher's k-th forward}} from one rollout.

    Args:
      unet:     the frozen teacher UNet (diffusers' or FastGen's wrapped one).
      modules:  {name: nn.Module} to hook, e.g. {"up0.attn0": unet.up_blocks[0].attentions[0]}.
      x0:       unit-variance initial noise [B, 4, H, W].
      cond:     {"pe": [*, 77, 2048], "pool": [*, 1280], "tid": [*, 6]}, conditional embeddings;
                a batch of 1 is broadcast to B.
      kmax:     number of forwards; the last one is step kmax.
      keep:     steps to return (default all of 1..kmax).
      guidance: 1.0 = conditional-only rollout; w != 1 = forwards 1..kmax-1 guided with CFG w,
                forward kmax conditional on that guided latent.
    """
    assert kmax >= 1, f"kmax must be >= 1, got {kmax}"
    assert modules, "no modules to hook"
    dev = device if device is not None else x0.device
    dt = dtype if dtype is not None else next(unet.parameters()).dtype
    sch = euler_scheduler(dev, steps=steps, model_id=model_id)
    assert kmax <= len(sch.timesteps), \
        f"kmax={kmax} exceeds the teacher's {len(sch.timesteps)}-step schedule"
    want = set(range(1, kmax + 1)) if keep is None else {int(k) for k in keep}

    grab: dict = {}
    handles = [m.register_forward_hook(
        lambda _m, _i, o, _n=name: grab.__setitem__(_n, _grab(o)))
        for name, m in modules.items()]

    lat = x0.to(dev, dt) * sch.init_noise_sigma
    b = lat.shape[0]
    pe = _rows(cond["pe"], b, dev, dt)
    added = {"text_embeds": _rows(cond["pool"], b, dev, dt),
             "time_ids": _rows(cond["tid"], b, dev, dt)}
    extra = _unet_extra_kwargs(unet)

    guided = float(guidance) != 1.0
    if guided:
        pe_u = torch.zeros_like(pe)
        added_u = {"text_embeds": torch.zeros_like(added["text_embeds"]), "time_ids": added["time_ids"]}
    out: dict = {}
    try:
        for i, t in enumerate(sch.timesteps):
            if i >= kmax:
                break
            grab.clear()
            inp = sch.scale_model_input(lat, t)
            v = _model_out(unet(inp, t, encoder_hidden_states=pe,
                                added_cond_kwargs=added, **extra))
            missing = [n for n in modules if n not in grab]
            assert not missing, f"teacher hooks did not fire at step {i + 1}: {missing}"
            if (i + 1) in want:
                out[i + 1] = dict(grab)          # the conditional forward's activation
            if i + 1 == kmax:
                break                            # the last forward's velocity is never used
            if guided:
                vu = _model_out(unet(inp, t, encoder_hidden_states=pe_u,
                                     added_cond_kwargs=added_u, **extra))
                v = vu + float(guidance) * (v - vu)
            lat = sch.step(v, t, lat, return_dict=False)[0]
    finally:
        for h in handles:
            h.remove()
    missing = sorted(want - set(out))
    assert not missing, f"requested steps not produced: {missing}"
    return out
