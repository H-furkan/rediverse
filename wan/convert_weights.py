#!/usr/bin/env python
"""Convert a wan.train checkpoint (.pth) into the released ReDiverse-FT adapter.

The checkpoint holds {"lora": {peft name -> fp32 tensor}, "step", "meta"}, with peft names
"blocks.N.<module>.lora_{A,B}.default.weight". The released file keeps the 580 float32 tensors
unchanged under "blocks.N.<module>.lora_{A,B}.weight" with string metadata, plus a config.json
next to it. wan.generate loads it through rediverse.hub.

  python -m wan.convert_weights --src outputs/wan2.1/train/adapters/rediverse_ft_0000200.pth \\
      --out-dir outputs/wan2.1/release
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from rediverse import data, hub
from wan import config as C


def release_name(k: str) -> str:
    return re.sub(r"\.lora_([AB])\.[^.]+\.weight$", r".lora_\1.weight", k)


def release_config(sd: dict, steps: int = data.STEPS) -> dict:
    """The config.json written next to the released adapter; `sd` uses the released key names."""
    modules = sorted({re.sub(r"\.lora_[AB]\.weight$", "", k) for k in sd})
    return dict(
        backbone=C.BACKBONE, denoiser="transformer", rank=C.LORA_RANK, alpha=C.LORA_ALPHA,
        base_models=[dict(role="teacher", repo=C.TEACHER_REPO, revision=C.TEACHER_REVISION),
                     dict(role="student", repo=C.STUDENT_REPO, revision=C.STUDENT_REVISION, file=C.STUDENT_FILE),
                     dict(role="target teacher LoRA", repo=C.CFGD_REPO, revision=C.CFGD_REVISION,
                          file=C.CFGD_FILE)],
        collapsed_block=f"blocks.{C.ANCHOR_BLOCK}",
        active_student_steps="1",
        n_modules=len(modules),
        n_parameters=sum(v.numel() for v in sd.values()),
        training_steps=steps,
        usage='python -m wan.generate --arm rediverse_ft --prompt "<prompt>"',
        recipe=dict(
            student_pipeline=f"WanTransformer3DModel with the teacher's T2V transformer config and the student "
                             f"weights, the teacher's text encoder and VAE; {C.STUDENT_STEPS} steps, flow shift "
                             f"{C.STUDENT_SHIFT}, guidance 1",
            teacher_target=f"forward {C.FT_TARGET_FORWARD} of the CFG-distilled teacher (the teacher with the "
                           f"target teacher LoRA; UniPC 50 steps, flow shift {C.CFGD_FLOW_SHIFT}, guidance 1)",
            teacher_target_forward=C.FT_TARGET_FORWARD,
            train_timestep=C.TRAIN_TIMESTEP,
            lora_blocks=f"0-{C.LORA_LAST_BLOCK}",
            lora_suffixes=sorted({".".join(m.split(".")[2:]) for m in modules}),
            adapter_dtype="float32", inference_dtype="bfloat16",
            frames=C.FRAMES, fps=C.FPS, resolution=f"{C.WIDTH}x{C.HEIGHT}",
            training=dict(captions="prompts/train_captions.txt (100 DiffusionDB captions)",
                          noises_per_caption=data.NOISES_PER_CAPTION, captions_per_step=data.CAPTIONS_PER_STEP,
                          samples_per_step=data.SAMPLES_PER_STEP, loss=f"{data.REG_LOSS_WEIGHT} * MSE",
                          lr=data.LR, weight_decay=data.WEIGHT_DECAY, grad_clip=data.GRAD_CLIP)),
        modules=modules,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="training checkpoint rediverse_ft_{step:07d}.pth")
    ap.add_argument("--out-dir", required=True, help="writes rediverse_ft.safetensors and config.json here")
    a = ap.parse_args()

    blob = torch.load(a.src, map_location="cpu", weights_only=True)
    src = blob["lora"]
    sd = {release_name(k): v for k, v in src.items()}
    assert len(sd) == len(src) == 2 * C.N_LORA_MODULES, (len(sd), len(src))
    assert all(v.dtype == torch.float32 for v in sd.values())
    modules = sorted({re.sub(r"\.lora_[AB]\.weight$", "", k) for k in sd})
    ranks = {min(v.shape) for v in sd.values()}
    assert ranks == {C.LORA_RANK}, ranks
    n_params = sum(v.numel() for v in sd.values())

    student = f"{C.STUDENT_REPO}@{C.STUDENT_REVISION} ({C.STUDENT_FILE})"
    teacher = (f"{C.TEACHER_REPO}@{C.TEACHER_REVISION} + {C.CFGD_REPO}@{C.CFGD_REVISION} "
               f"(UniPC 50 steps, flow shift {C.CFGD_FLOW_SHIFT}, guidance 1, forward {C.FT_TARGET_FORWARD})")
    meta = dict(format="peft-lora", rank=C.LORA_RANK, alpha=C.LORA_ALPHA, dropout=data.LORA_DROPOUT,
                denoiser="transformer", backbone=C.BACKBONE, student=student, teacher=teacher,
                anchor_block=f"blocks.{C.ANCHOR_BLOCK}", active_student_steps="1",
                lora_blocks=f"0..{C.LORA_LAST_BLOCK}", n_modules=len(modules),
                target_modules=",".join(sorted({".".join(m.split(".")[2:]) for m in modules})),
                train_timestep=C.TRAIN_TIMESTEP, step=blob.get("step"))
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, hub.ADAPTER_FILE)
    hub.save_lora_safetensors(path, sd, meta)

    back, m2 = hub.load_lora_safetensors(path)
    assert set(back) == set(sd) and all(torch.equal(back[k], sd[k]) for k in sd), "round trip differs"
    assert m2 == {k: str(v) for k, v in meta.items()}

    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(release_config(sd, int(blob.get("step", data.STEPS))), f, indent=2)
    print(f"wrote {path}: {len(sd)} tensors ({n_params:,} params), {len(modules)} modules, step {blob.get('step')}")


if __name__ == "__main__":
    main()
