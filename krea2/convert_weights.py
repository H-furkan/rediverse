"""Convert a krea2.train adapter into the released Krea 2 ReDiverse files.

    python -m krea2.convert_weights --src <run>/adapters/rediverse_ft_0000200.safetensors --out-dir outputs/krea2/release

Writes
  krea-2-rediverse.safetensors        diffusers LoRA format (transformer.transformer_blocks.{b}.<module>.lora_{A,B}.weight),
                                      loadable with Krea2Pipeline.load_lora_weights(..., adapter_name="rediverse")
  krea-2-rediverse-comfy.safetensors  original Krea 2 / ComfyUI module names (diffusion_model.blocks.{b}.attn.wq ...),
                                      checked to convert back to the diffusers file exactly with diffusers' own converter
  config.json                         the recipe and the recommended inference settings
Tensors stay fp32 (as trained).
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch

from krea2 import config as C

ATTN = {"to_q": "wq", "to_k": "wk", "to_v": "wv", "to_out.0": "wo", "to_gate": "gate"}
FF = {"ff.gate": "mlp.gate", "ff.up": "mlp.up", "ff.down": "mlp.down"}
PAT = re.compile(r"^transformer_blocks\.(\d+)\.(.+)\.(lora_[AB])\.rediverse\.weight$")


def convert(sd: dict) -> tuple[dict, dict]:
    diff, comfy = {}, {}
    for k, v in sd.items():
        m = PAT.match(k)
        assert m, f"unexpected key {k}"
        b, mod, ab = m.groups()
        diff[f"transformer.transformer_blocks.{b}.{mod}.{ab}.weight"] = v.float().contiguous()
        if mod.startswith("attn."):
            orig = f"blocks.{b}.attn.{ATTN[mod[len('attn.'):]]}"
        else:
            orig = f"blocks.{b}.{FF[mod]}"
        comfy[f"diffusion_model.{orig}.{ab}.weight"] = v.float().contiguous()
    return diff, comfy


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--band", type=float, default=4.0, help="recommended TF alpha range (paper: 2)")
    a = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    with safe_open(a.src, "pt") as fh:
        md = fh.metadata() or {}
    sd = load_file(a.src)
    diff, comfy = convert(sd)
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_krea2_lora_to_diffusers
    back = _convert_non_diffusers_krea2_lora_to_diffusers(dict(comfy))
    assert set(back) == set(diff) and all(torch.equal(back[k], diff[k]) for k in diff), "comfy round-trip mismatch"
    rank = int(md.get("rank", 32))
    last = int(md.get("lora_last", C.ANCHOR_BLOCK))
    meta = {"format": "pt", "rank": str(rank), "alpha": str(rank), "lora_last": str(last), "block": str(md.get("block", last)),
            "step": str(md.get("step", "")), "base_model": C.STUDENT_ID, "teacher": C.TEACHER_ID, "method": "ReDiverse-FT"}
    os.makedirs(a.out_dir, exist_ok=True)
    save_file(diff, os.path.join(a.out_dir, "krea-2-rediverse.safetensors"), metadata=meta)
    save_file(comfy, os.path.join(a.out_dir, "krea-2-rediverse-comfy.safetensors"), metadata=meta)
    cfg = dict(
        name="Krea 2 ReDiverse", method="ReDiverse (ReDiverse-FT adapter + ReDiverse-TF steering)", paper="arXiv:2611.18550",
        base_model=C.STUDENT_ID, base_revision=C.STUDENT_REVISION, teacher=C.TEACHER_ID, teacher_revision=C.TEACHER_REVISION,
        resolution=[C.RES, C.RES], student_steps=C.STEPS["student"], student_guidance=0.0,
        collapsed_block=last, lora=dict(rank=rank, alpha=rank, blocks=f"0..{last}",
                                        modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate", "attn.to_out.0",
                                                 "ff.gate", "ff.up", "ff.down"], dtype="float32", active_steps=[1]),
        training=dict(target=f"Krea-2-Raw block {last} image rows at teacher forward {C.TEACHER_TARGET_STEP} of 52 "
                             f"(4 guided steps, Krea guidance {C.GUIDANCE['teacher']} = CFG {1 + C.GUIDANCE['teacher']})",
                      captions="100 DiffusionDB captions x 16 noises", steps=200, batch=32, lr=1e-4, loss="MSE",
                      student_input="Turbo step 1 (sigma 1) from the same noise"),
        tf=dict(block=last, teacher_forward=C.DIRECTION_STEP, rollouts=25, rollout_guidance=0.0, directions=8,
                alpha_range=a.band, paper_alpha_range=2.0, active_steps=[1]),
    )
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    n = sum(v.numel() for v in diff.values())
    print(f"{len(diff)} tensors, {n / 1e6:.1f}M params -> {a.out_dir} (comfy round-trip OK)")


if __name__ == "__main__":
    main()
