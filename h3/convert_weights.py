#!/usr/bin/env python
"""Convert a ReDiverse-FT training checkpoint (h3.train, rediverse_ft_<iter>.pth) into the adapter
file read by h3.generate --adapter: <out-dir>/rediverse_ft.safetensors + <out-dir>/config.json.

The checkpoint is a flat dict of float32 tensors named
'transformer_blocks.{b}.{module}.lora_{A|B}.<adapter name>.weight'; the adapter file keeps every
tensor unchanged and drops the adapter-name segment ('transformer_blocks.{b}.{module}.lora_{A|B}.weight').

  python -m h3.convert_weights --src outputs/minimax-h3/train/adapters/rediverse_ft_0000200.pth \\
      --out-dir outputs/minimax-h3/train/rediverse_ft
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from h3 import config as C
from rediverse import data, hub

_KEY = re.compile(r"^(transformer_blocks\.\d+\..+)\.(lora_[AB])\.([^.]+)\.weight$")


def to_release_keys(sd: dict) -> dict:
    """Training-checkpoint keys -> adapter-file keys (the adapter-name segment removed)."""
    out = {}
    for k, v in sd.items():
        m = _KEY.match(k)
        assert m, f"unexpected checkpoint key {k!r}"
        out[f"{m.group(1)}.{m.group(2)}.weight"] = v
    assert len(out) == len(sd)
    return out


def release_metadata(sd: dict) -> dict:
    """The safetensors metadata of the adapter file (all values strings)."""
    ranks = {min(v.shape) for v in sd.values()}
    assert ranks == {C.LORA_RANK}, ranks
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in sd})
    blocks = sorted({int(m.split(".")[1]) for m in modules})
    return dict(
        format="peft-lora", rank=C.LORA_RANK, alpha=C.LORA_ALPHA, denoiser="transformer",
        backbone=C.BACKBONE,
        student=f"{C.STUDENT_REPO}@{C.STUDENT_REVISION}:{C.STUDENT_FILE}",
        teacher=f"{C.TEACHER_REPO}@{C.TEACHER_REVISION}",
        base_transformer=f"{C.TEACHER_REPO}@{C.TEACHER_REVISION}/transformer",
        anchor_block=C.ANCHOR_BLOCK, anchor_stream="video rows",
        teacher_target_forward=C.TAU,
        active_student_steps="1",
        stacking="applied on top of the unfused Turbo LoRA adapter; active only in the student's first forward",
        inference_dtype="bfloat16",
        target_suffixes=",".join(C.LORA_SUFFIXES),
        target_blocks=f"{blocks[0]}-{blocks[-1]}",
        n_modules=len(modules),
        frames=C.NUM_FRAMES, resolution=f"{C.WIDTH}x{C.HEIGHT}",
    )


def release_config(sd: dict, steps: int = C.MAX_ITER - 1) -> dict:
    """The config.json written next to the adapter file; `sd` uses the adapter-file key names."""
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in sd})
    blocks = sorted({int(m.split(".")[1]) for m in modules})
    return dict(
        backbone=C.BACKBONE, denoiser="transformer", rank=C.LORA_RANK, alpha=C.LORA_ALPHA,
        base_models=[dict(role="teacher", repo=C.TEACHER_REPO, revision=C.TEACHER_REVISION),
                     dict(role="student", repo=C.STUDENT_REPO, revision=C.STUDENT_REVISION, file=C.STUDENT_FILE)],
        collapsed_block=f"transformer_blocks.{C.ANCHOR_BLOCK}",
        active_student_steps="1",
        n_modules=len(modules),
        n_parameters=sum(v.numel() for v in sd.values()),
        training_steps=steps,
        usage='python -m h3.generate --arm rediverse_ft --adapter <directory holding rediverse_ft.safetensors> '
              '--prompt "<prompt>"',
        recipe=dict(
            student_pipeline=f"the teacher's transformer with the unfused Turbo LoRA, {C.STEPS['student']} "
                             f"scheduler steps ({C.STEPS['student'] - 1} transformer forwards), no guidance "
                             f"(guidance-distilled); the adapter is stacked on the Turbo LoRA",
            collapsed_stream="video rows",
            teacher_target_forward=C.TAU,
            lora_blocks=f"{blocks[0]}-{blocks[-1]}", lora_suffixes=list(C.LORA_SUFFIXES),
            adapter_dtype="float32", inference_dtype="bfloat16",
            frames=C.NUM_FRAMES, fps=C.FPS, resolution=f"{C.WIDTH}x{C.HEIGHT}",
            training=dict(prompts="prompts/h3_train_geneval_ids.txt over prompts/geneval_553.txt",
                          latents_per_prompt=C.FT_LATENTS_PER_PROMPT, samples_per_prompt=C.SAMPLES_PER_PROMPT,
                          prompts_per_step=C.CAPTIONS_PER_STEP,
                          loss=f"{C.REG_LOSS_WEIGHT} * MSE / Var(target) of each latent",
                          lr=data.LR, weight_decay=data.WEIGHT_DECAY, grad_clip=data.GRAD_CLIP)),
        modules=modules,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="training checkpoint rediverse_ft_<iter>.pth written by h3.train")
    ap.add_argument("--out-dir", required=True, help="writes rediverse_ft.safetensors and config.json here")
    a = ap.parse_args()
    src = torch.load(a.src, map_location="cpu")
    assert all(v.dtype == torch.float32 for v in src.values())
    sd = to_release_keys(src)
    meta = release_metadata(sd)
    d = a.out_dir
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, hub.ADAPTER_FILE)
    hub.save_lora_safetensors(path, sd, meta)
    back, back_meta = hub.load_lora_safetensors(path)
    assert back.keys() == sd.keys() and all(torch.equal(back[k], sd[k]) for k in sd)
    m = re.search(r"_(\d+)\.pth$", a.src)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(release_config(sd, int(m.group(1)) if m else C.MAX_ITER - 1), f, indent=2)
    print(f"[h3/convert] {len(sd)} tensors, {meta['n_modules']} modules -> {path}", flush=True)


if __name__ == "__main__":
    main()
