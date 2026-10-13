#!/usr/bin/env python
"""Convert a ReDiverse-FT adapter written by training (adapters/rediverse_ft_<iteration>.pth) into
the released adapter: <out-dir>/rediverse_ft.safetensors + <out-dir>/config.json.

The .pth holds {"lora": {name: tensor}, "src": "net", "iter": ...} with float32 tensors named
'unet.<module>.lora_{A|B}.default.weight' (FastGen wraps the UNet as `unet`, peft names the
adapter `default`); sdxl/extract_adapter.py writes it from a FastGen checkpoint. The released
file keeps every tensor unchanged and names it '<module>.lora_{A|B}.weight'.

  python -m sdxl.convert_weights --src outputs/sdxl/train/adapters/rediverse_ft_0000200.pth \\
      --out-dir outputs/sdxl/release
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from rediverse import data as D
from rediverse import hub
from sdxl import config as C

_KEY = re.compile(r"^unet\.(.+)\.(lora_[AB])\.([^.]+)\.weight$")


def to_release_keys(sd: dict) -> dict:
    """Training keys 'unet.<module>.lora_X.<adapter>.weight' -> '<module>.lora_X.weight'."""
    out = {}
    for k, v in sd.items():
        m = _KEY.match(k)
        assert m, f"unexpected adapter key {k!r}"
        out[f"{m.group(1)}.{m.group(2)}.weight"] = v
    assert len(out) == len(sd)
    return out


def release_metadata(sd: dict, steps: int = D.STEPS) -> dict:
    """The safetensors metadata of the released adapter (stored as strings)."""
    ranks = {v.shape[0] for k, v in sd.items() if ".lora_A." in k}
    assert ranks == {D.LORA_RANK}, ranks
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in sd})
    return dict(
        format="peft-lora", rank=D.LORA_RANK, alpha=D.LORA_ALPHA, denoiser="unet",
        backbone=C.BACKBONE,
        student=f"{C.DMD2_ID}@{C.DMD2_REVISION}:{C.DMD2_UNET_FP16}",
        teacher=f"{C.SDXL_ID}@{C.SDXL_REVISION}",
        anchor_block=C.ANCHOR_MODULE,
        teacher_target_forward=C.TEACHER_TARGET_STEP,
        teacher_guidance=C.TEACHER_GUIDANCE,
        active_student_steps="1",
        inference_dtype="float16",
        target_regex=C.LORA_TARGET_REGEX,
        n_modules=len(modules),
        training_steps=steps,
        resolution=f"{C.WIDTH}x{C.HEIGHT}",
    )


def release_config(sd: dict, steps: int = D.STEPS) -> dict:
    """The config.json written next to the released adapter; `sd` uses the released key names."""
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in sd})
    return dict(
        backbone=C.BACKBONE, denoiser="unet", rank=D.LORA_RANK, alpha=D.LORA_ALPHA,
        base_models=[dict(role="teacher", repo=C.SDXL_ID, revision=C.SDXL_REVISION),
                     dict(role="student", repo=C.DMD2_ID, revision=C.DMD2_REVISION, file=C.DMD2_UNET_FP16),
                     dict(role="training initialisation", repo=C.DMD2_ID, revision=C.DMD2_REVISION,
                          file=C.DMD2_UNET_FP32)],
        collapsed_block=C.ANCHOR_MODULE,
        active_student_steps="1",
        n_modules=len(modules),
        n_parameters=sum(v.numel() for v in sd.values()),
        training_steps=steps,
        usage='python -m sdxl.generate --arm rediverse_ft --prompt "<prompt>"',
        recipe=dict(
            student_pipeline="the teacher's pipeline with the DMD2 UNet and an LCMScheduler built from its "
                             "scheduler config, guidance scale 0",
            student_timesteps=C.STUDENT_TIMESTEPS,
            teacher_target_forward=C.TEACHER_TARGET_STEP, teacher_guidance=C.TEACHER_GUIDANCE,
            lora_target_regex=C.LORA_TARGET_REGEX,
            adapter_dtype="float32", inference_dtype="float16", resolution=f"{C.WIDTH}x{C.HEIGHT}",
            training=dict(captions="prompts/train_captions.txt (100 DiffusionDB captions)",
                          noises_per_caption=D.NOISES_PER_CAPTION, captions_per_step=D.CAPTIONS_PER_STEP,
                          samples_per_step=D.SAMPLES_PER_STEP, loss=f"{D.REG_LOSS_WEIGHT} * MSE",
                          lr=D.LR, weight_decay=D.WEIGHT_DECAY, grad_clip=D.GRAD_CLIP, precision="float32")),
        modules=modules,
    )


def write_release(sd_train: dict, out_dir: str, steps: int = D.STEPS) -> str:
    """Write <out_dir>/rediverse_ft.safetensors (+ config.json) from training-named tensors."""
    assert all(v.dtype == torch.float32 for v in sd_train.values())
    sd = to_release_keys(sd_train)
    meta = release_metadata(sd, steps)
    assert meta["n_modules"] == C.N_LORA_MODULES and len(sd) == 2 * C.N_LORA_MODULES, (meta["n_modules"], len(sd))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, hub.ADAPTER_FILE)
    hub.save_lora_safetensors(path, sd, meta)
    back, _ = hub.load_lora_safetensors(path)
    assert back.keys() == sd.keys() and all(torch.equal(back[k], sd[k]) for k in sd)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(release_config(sd, steps), f, indent=2)
    print(f"[sdxl/convert] {len(sd)} tensors, {meta['n_modules']} modules -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="rediverse_ft_<iteration>.pth ({'lora': {...}, 'src': 'net', "
                                                 "'iter': ...})")
    ap.add_argument("--out-dir", required=True, help="writes <out-dir>/rediverse_ft.safetensors + config.json")
    a = ap.parse_args()
    blob = torch.load(a.src, map_location="cpu", weights_only=True)
    sd = blob["lora"] if "lora" in blob else blob
    steps = int(blob["iter"]) if "iter" in blob else D.STEPS
    write_release(sd, a.out_dir, steps)


if __name__ == "__main__":
    main()
