"""TPIPS-Embed checkpoint and factors (paper App. F), shared by the image and video metrics.
TPIPS-Embed is instructed with 'Represent the similarity of the image.' for 'overall' and
'Represent the similarity of the image based on <factor>.' for the four aspects.
Standard library only at import.
"""
from __future__ import annotations

import re

# package tpips 0.1.2; its Qwen3-VL-Embedding-8B backbone is resolved by the package
TPIPS_REPO = "sywang/TPIPS-Embed-Qwen3VL-8B"
TPIPS_REV = "86a1493c60121e656a1c075bc7aed88f25fcd4d4"

FACTORS = {                         # slug -> factor text handed to TPIPS
    "overall": "overall",
    "subject": "the main subject",
    "background": "the background scene",
    "camera": "the camera viewpoint and framing",
    "lighting": "lighting and color",
}


def load_model(device: str = "cuda"):
    """TPIPS-Embed at the pinned checkpoint revision."""
    import tpips
    from huggingface_hub import snapshot_download
    return tpips.load_model("embedding", model_path=snapshot_download(TPIPS_REPO, revision=TPIPS_REV), device=device)


def slug(factor: str) -> str:
    """Directory name of a factor: its FACTORS key, or a sanitised free-text factor."""
    for k, v in FACTORS.items():
        if factor == v or factor == k:
            return k
    return re.sub(r"[^a-z0-9]+", "_", factor.lower()).strip("_")
