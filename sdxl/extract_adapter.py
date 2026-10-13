#!/usr/bin/env python
"""Extract the ReDiverse-FT adapter from a FastGen training checkpoint.

The checkpoint (<run>/checkpoints/<iteration:07d>.pth) holds the full fp32 student, its LoRA and
the optimizer state; only the generator LoRA tensors are kept, unchanged, in
<out-dir>/rediverse_ft_<iteration:07d>.pth = {"lora": {name: tensor}, "src": "net", "iter": iteration}.
sdxl.train runs this on its last checkpoint; sdxl/convert_weights.py turns the result into the
released safetensors.

  python -m sdxl.extract_adapter --src outputs/sdxl/train/checkpoints/0000200.pth
  # -> outputs/sdxl/train/adapters/rediverse_ft_0000200.pth
"""
from __future__ import annotations

import argparse
import os

import torch


def lora_from_checkpoint(path: str):
    """({name: tensor} of the generator LoRA, iteration) from a FastGen checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = blob["model"] if "model" in blob else blob
    lora = {k: v.clone() for k, v in model["net"].items() if "lora_" in k}
    it = blob.get("iteration")
    if it is None:
        it = int(os.path.splitext(os.path.basename(path))[0])
    return lora, int(it)


def extract(checkpoint: str, out_dir: str = None) -> str:
    """Write the generator LoRA of `checkpoint` to <out_dir>/rediverse_ft_<iteration:07d>.pth
    (default out_dir: adapters/ beside the checkpoints/ directory)."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint))), "adapters")
    lora, it = lora_from_checkpoint(checkpoint)
    assert lora, f"no LoRA tensors in {checkpoint}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rediverse_ft_{it:07d}.pth")
    torch.save({"lora": lora, "src": "net", "iter": it}, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[sdxl/extract] {len(lora)} tensors, {sum(v.numel() for v in lora.values()) / 1e6:.1f}M params "
          f"-> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="FastGen checkpoint <iteration:07d>.pth")
    ap.add_argument("--out-dir", default=None, help="default: adapters/ beside the checkpoints/ directory")
    a = ap.parse_args()
    extract(a.src, a.out_dir)


if __name__ == "__main__":
    main()
