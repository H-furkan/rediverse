#!/usr/bin/env python
"""Build the FastGen initialisation checkpoint for ReDiverse-FT: the fp32 DMD2 SDXL 4-step UNet
(tianweiy/DMD2, dmd2_sdxl_4step_unet.bin) as the student `net`.

Layout read by FastGen's Checkpointer: {"model": {"net": <state_dict>}}. FastGen's SDXL wrapper
holds the UNet at attribute `unet`, so the diffusers UNet keys get the `unet.` prefix; the wrapper's
extra logvar_linear head is not in the file (FastGen loads with strict=False).

  python -m sdxl.build_init_ckpt                     # -> data/sdxl/dmd2_sdxl_init.pth (10.3 GB)
"""
from __future__ import annotations

import argparse
import os

import torch

from sdxl import config as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="local dmd2_sdxl_4step_unet.bin (default: download from the Hub)")
    ap.add_argument("--out", default=C.init_ckpt_path())
    a = ap.parse_args()

    src = a.src
    if src is None:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(C.DMD2_ID, C.DMD2_UNET_FP32, revision=C.DMD2_REVISION)
    sd = torch.load(src, map_location="cpu", weights_only=True)
    unet_sd = sd if any(k.startswith("unet.") for k in sd) else {f"unet.{k}": v for k, v in sd.items()}
    n_params = sum(v.numel() for v in unet_sd.values())
    dtypes = {str(v.dtype) for v in unet_sd.values()}
    print(f"params: {n_params/1e6:.1f}M  dtypes: {dtypes}  keys: {len(unet_sd)}")
    assert n_params > 2.5e9 and dtypes == {"torch.float32"}, "expected the full fp32 SDXL UNet (~2.57B params)"

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"model": {"net": unet_sd}}, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    print(f"saved {a.out} ({os.path.getsize(a.out)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
