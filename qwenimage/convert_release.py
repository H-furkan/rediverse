"""Convert a qwenimage.train adapter into a released ReDiverse LoRA file (diffusers format).

    python -m qwenimage.convert_release --student lightning20 \
        --src <run>/adapters/rediverse_ft_0000200.safetensors --out <dir>/qwen-image-lightning-rediverse.safetensors

Keys become transformer.transformer_blocks.{b}.<module>.lora_{A,B}.weight (diffusers LoRA format, no alpha tensors:
alpha = rank, scale 1, as trained), loadable with QwenImagePipeline.load_lora_weights(..., adapter_name="rediverse").
Tensors stay fp32 (as trained). The file's metadata records the recipe.
"""
from __future__ import annotations

import argparse
import os
import re

import torch

from qwenimage import config as C

PAT = re.compile(r"^transformer_blocks\.(\d+)\.(.+)\.(lora_[AB])\.rediverse\.weight$")


def convert(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        m = PAT.match(k)
        assert m, f"unexpected key {k}"
        b, mod, ab = m.groups()
        out[f"transformer.transformer_blocks.{b}.{mod}.{ab}.weight"] = v.float().contiguous()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    with safe_open(a.src, "pt") as fh:
        md = fh.metadata() or {}
    assert md.get("student") == a.student, (md, a.student)
    sd = load_file(a.src)
    out = convert(sd)
    s = C.student(a.student)
    f = C.family(s["family"])
    meta = {"format": "pt", "method": "ReDiverse-FT", "paper": "arXiv:2611.18550", "rank": md.get("rank", "32"),
            "alpha": md.get("rank", "32"), "blocks": f"0..{md['lora_last']}", "collapsed_block": md["block"],
            "teacher_forward": md["tau"], "step": md.get("step", ""), "base_model": f["repo"],
            "base_revision": f["revision"], "student_lora": f"{s['repo']}/{s.get('file', '')}",
            "student_lora_revision": s["revision"], "active_steps": "1"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(out, a.out + ".tmp", metadata=meta)
    os.replace(a.out + ".tmp", a.out)
    back = load_file(a.out)
    assert set(back) == set(out) and all(torch.equal(back[k], out[k]) for k in out)
    n_mod = len(out) // 2
    print(f"[convert] {len(out)} tensors ({n_mod} modules, {sum(v.numel() for v in out.values()) / 1e6:.1f}M params) "
          f"-> {a.out} ({os.path.getsize(a.out) / 2 ** 20:.0f} MiB); metadata {meta}")


if __name__ == "__main__":
    main()
