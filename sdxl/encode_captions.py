#!/usr/bin/env python
"""Encode the 100 ReDiverse-FT training captions with SDXL's text encoders.

Writes <REDIVERSE_DATA>/sdxl/train_ctx/p{ci:03d}.pt = {pe [77, 2048] float16, pool [1280] float16,
prompt, ci, ctx_index}: the pipeline's own encode_prompt, conditional only, with the float16 text
encoders. Captions longer than 77 CLIP tokens are truncated, as in every SDXL pipeline call. A file
that already holds its caption is kept; one encoded from another caption is rewritten.

  python -m sdxl.encode_captions                     # ~2 min on one GPU
"""
from __future__ import annotations

import argparse
import os

import torch

from rediverse import data as D
from sdxl import config as C


def encoded(path: str, caption: str) -> bool:
    """True when the context file `path` exists and was encoded from `caption`."""
    return os.path.exists(path) and torch.load(path, map_location="cpu", mmap=True).get("prompt") == caption


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", default=None, help="caption file (default prompts/train_captions.txt)")
    ap.add_argument("--out-dir", default=C.ctx_dir())
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    caps = D.captions(a.captions)
    os.makedirs(a.out_dir, exist_ok=True)
    pending = [ci for ci in range(len(caps)) if not encoded(os.path.join(a.out_dir, f"p{ci:03d}.pt"), caps[ci])]
    print(f"[sdxl/encode] {len(pending)}/{len(caps)} captions to encode -> {a.out_dir}", flush=True)
    if not pending:
        return
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(C.SDXL_ID, revision=C.SDXL_REVISION, unet=None, vae=None,
                                                     torch_dtype=torch.float16, variant="fp16",
                                                     add_watermarker=False)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(a.device)
    with torch.no_grad():
        for ci in pending:
            pe, _, pool, _ = pipe.encode_prompt(prompt=caps[ci], device=a.device, num_images_per_prompt=1,
                                                do_classifier_free_guidance=False)
            assert pe.shape == (1, 77, 2048) and pool.shape == (1, 1280), (pe.shape, pool.shape)
            f = os.path.join(a.out_dir, f"p{ci:03d}.pt")
            torch.save(dict(pe=pe[0].cpu().to(torch.float16), pool=pool[0].cpu().to(torch.float16),
                            prompt=caps[ci], ci=ci, ctx_index=D.ctx_index(ci)), f + ".tmp")
            os.replace(f + ".tmp", f)
    print(f"[sdxl/encode] done: {len(caps)} files in {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
