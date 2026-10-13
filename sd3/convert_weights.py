"""Convert a trainer checkpoint (.pth) into the released rediverse_ft.safetensors + config.json.

    python -m sd3.convert_weights --src outputs/sd3/train/adapters/rediverse_ft_0000200.pth \\
        --out-dir outputs/sd3/release

The tensors are copied unchanged (fp32); keys lose peft's adapter-name segment:
'transformer_blocks.{b}.{module}.lora_A.<adapter>.weight' -> 'transformer_blocks.{b}.{module}.lora_A.weight'.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from rediverse import data as D
from rediverse.data import LORA_ALPHA, LORA_DROPOUT, LORA_RANK
from rediverse.hub import ADAPTER_FILE, load_lora_safetensors, save_lora_safetensors
from sd3.config import (ANCHOR_BLOCK, BACKBONE, FT_ADAPTER, GUIDANCE, LORA_ID, LORA_LAST_BLOCK,
                        LORA_REVISION, LORA_SUFFIXES, LORA_WEIGHT_NAME, MODEL_ID, MODEL_REVISION,
                        N_LORA_MODULES, STEPS, TDM_ADAPTER, TDM_SCALE, TEACHER_GUIDANCE,
                        TEACHER_TARGET_STEP, TEXT_ONLY_AT_ANCHOR)

SEGMENT = f".{FT_ADAPTER}."


def released_key(k: str) -> str:
    assert SEGMENT in k and k.endswith(".weight"), k
    return k.replace(SEGMENT, ".")


def metadata(n_modules: int) -> dict:
    """String metadata stored in the safetensors header."""
    return dict(
        format="peft LoRA, keys <module path in SD3Transformer2DModel>.lora_{A,B}.weight",
        backbone="sd3", denoiser="transformer",
        teacher=MODEL_ID, teacher_revision=MODEL_REVISION,
        student=LORA_ID, student_revision=LORA_REVISION, student_weight_name=LORA_WEIGHT_NAME,
        student_adapter_name=TDM_ADAPTER, student_adapter_scale=str(TDM_SCALE),
        adapter_name=FT_ADAPTER, rank=str(LORA_RANK), alpha=str(LORA_ALPHA), dropout=str(LORA_DROPOUT),
        dtype="float32", anchor_block=str(ANCHOR_BLOCK), anchor_stream="image",
        lora_blocks=f"0-{LORA_LAST_BLOCK}", lora_suffixes=",".join(LORA_SUFFIXES),
        excluded_at_anchor_block=",".join(TEXT_ONLY_AT_ANCHOR), n_modules=str(n_modules),
        active_student_steps="1", stream_restriction="block-12 text stream restored to the student's at step 1",
        teacher_target_step=str(TEACHER_TARGET_STEP), teacher_guidance=str(TEACHER_GUIDANCE),
    )


def release_config(sd: dict, steps: int = D.STEPS) -> dict:
    """The config.json written next to the released adapter; `sd` uses the released key names."""
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in sd})
    return dict(
        backbone=BACKBONE, denoiser="transformer", rank=LORA_RANK, alpha=LORA_ALPHA,
        base_models=[dict(role="teacher", repo=MODEL_ID, revision=MODEL_REVISION),
                     dict(role="student", repo=LORA_ID, revision=LORA_REVISION, file=LORA_WEIGHT_NAME)],
        collapsed_block=f"transformer_blocks.{ANCHOR_BLOCK}",
        active_student_steps="1",
        n_modules=len(modules),
        n_parameters=sum(v.numel() for v in sd.values()),
        training_steps=steps,
        usage='python -m sd3.generate --arm rediverse_ft --prompt "<prompt>"',
        recipe=dict(
            student_pipeline=f"the teacher's pipeline with the TDM LoRA at scale {TDM_SCALE}, "
                             f"{STEPS['student']} steps, guidance {GUIDANCE['student']}",
            student_lora_scale=TDM_SCALE,
            collapsed_stream="image",
            stream_restriction=f"in step 1 the block-{ANCHOR_BLOCK} text-stream output is the student's own, "
                               "so the adapter acts on the image stream",
            lora_blocks=f"0-{LORA_LAST_BLOCK}", lora_suffixes=list(LORA_SUFFIXES),
            excluded_at_collapsed_block=list(TEXT_ONLY_AT_ANCHOR),
            teacher_target_forward=TEACHER_TARGET_STEP, teacher_guidance=TEACHER_GUIDANCE,
            adapter_dtype="float32", inference_dtype="bfloat16", resolution="1024x1024",
            training=dict(captions="prompts/train_captions.txt (100 DiffusionDB captions)",
                          noises_per_caption=D.NOISES_PER_CAPTION, captions_per_step=D.CAPTIONS_PER_STEP,
                          samples_per_step=D.SAMPLES_PER_STEP, loss=f"{D.REG_LOSS_WEIGHT} * MSE",
                          lr=D.LR, weight_decay=D.WEIGHT_DECAY, grad_clip=D.GRAD_CLIP)),
        modules=modules,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="trainer checkpoint rediverse_ft_*.pth")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    ck = torch.load(a.src, map_location="cpu")
    src = ck["state_dict"]
    sd = {released_key(k): v for k, v in src.items()}
    assert len(sd) == len(src) == 2 * N_LORA_MODULES, len(sd)
    assert all(v.dtype == torch.float32 for v in sd.values())
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in sd})
    assert len(modules) == N_LORA_MODULES, len(modules)
    meta = metadata(len(modules))

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, ADAPTER_FILE)
    save_lora_safetensors(out, sd, meta)

    back, back_meta = load_lora_safetensors(out)
    assert set(back) == set(sd) and all(torch.equal(back[k], v) for k, v in sd.items())
    assert back_meta == {k: str(v) for k, v in meta.items()}
    sum_b = sum(float(v.abs().sum()) for k, v in sd.items() if "lora_B" in k)
    n_params = sum(v.numel() for v in sd.values())
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(release_config(sd, int(ck.get("iteration", D.STEPS))), f, indent=2)
    print(f"[convert] {a.src} -> {out}: {len(sd)} tensors ({len(modules)} modules, {n_params} params), "
          f"sum|lora_B| = {sum_b:.3f}", flush=True)


if __name__ == "__main__":
    main()
