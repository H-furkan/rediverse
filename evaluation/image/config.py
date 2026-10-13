"""Models behind the image metrics (paper App. F). The Hugging Face models are pinned to the
revisions that produced the paper's numbers (TPIPS-Embed in evaluation/common/tpips_factors.py).
Weights download on first use into the user's caches: Hugging Face models into HF_HOME, DreamSim
into DREAMSIM_CACHE (default <torch hub dir>/dreamsim), the LPIPS VGG16 trunk into the torch hub
dir, ImageReward-v1.0 into ~/.cache/ImageReward (fetched by the image-reward package).
"""
from __future__ import annotations

import os

# DINO (patch-token diversity) and Vendi (CLS-token kernel)
DINO_ID = "facebook/dinov2-base"
DINO_REV = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
# CLIPScore
CLIP_ID = "openai/clip-vit-base-patch32"
CLIP_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
# DreamSim ensemble (DINO + CLIP + OpenCLIP ViT-B/16 with LoRA), package dreamsim 0.2.1
DREAMSIM_TYPE = "ensemble"
# HPSv2.1: open_clip ViT-H-14 with the HPS_v2.1 state dict
HPS_REPO = "xswu/HPSv2"
HPS_REV = "697403c78157020a1ae59d23f111aa58ced35b0a"
HPS_FILES = {"v2.1": "HPS_v2.1_compressed.pt", "v2": "HPS_v2_compressed.pt"}
# ImageReward, package image-reward 1.5
IMAGEREWARD_NAME = "ImageReward-v1.0"
# PickScore (optional; in no paper table)
PICKSCORE_ID = "yuvalkirstain/PickScore_v1"
PICKSCORE_REV = "a4e4367c6dfa7288a00c550414478f865b875800"
PICKSCORE_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_PROCESSOR_REV = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"


def dreamsim_cache() -> str:
    """DreamSim weight directory: $DREAMSIM_CACHE, else <torch hub dir>/dreamsim."""
    if os.environ.get("DREAMSIM_CACHE"):
        return os.environ["DREAMSIM_CACHE"]
    import torch
    return os.path.join(torch.hub.get_dir(), "dreamsim")
