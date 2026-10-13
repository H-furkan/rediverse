"""Frame-wise DINOv2 and clip-level V-JEPA 2 embeddings of a video set (main comparison table: DINO, VJ).

DINO  16 frames per clip at linspace(0, T-1, 16).round() (first and last frame included), decoded at native
      resolution and passed through the DINOv2-B image processor (shortest edge 256, bicubic, centre crop 224).
      The feature is the CLS token after the final layer norm (last_hidden_state[:, 0]), fp16 model, batch 64,
      L2-normalised  ->  <out>/vdino/dinov2/p<idx>.npz {cls [10, 16, 768] fp16, prompt}.
VJ    64 frames at linspace(0, T-1, 64).round(), scaled by ffmpeg to 256x256 (aspect ratio not kept), then the
      V-JEPA 2 video processor (shortest edge 292, centre crop 256, ImageNet normalisation). The embedding is
      the mean over all encoder tokens, L2-normalised  ->  <out>/vjepa2/p<idx>.npy [10, 1024] fp16.

video_metrics.py turns both caches into per-prompt pairwise diversity. Prompts whose caches exist are skipped.

  python -m evaluation.video.video_dino --videos-dir <set>/videos --out <set>/feats [--shard i --num-shards 4]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from evaluation.common.atomic import save
from evaluation.video.sets import groups, prompt_ids, shard
from rediverse.video_io import read_frames

DINO_ID, DINO_REVISION = "facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
VJEPA_ID, VJEPA_REVISION = "facebook/vjepa2-vitl-fpc64-256", "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
DINO_FRAMES = 16
VJEPA_FRAMES, VJEPA_SIZE = 64, (256, 256)


def load_dino(dev):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(DINO_ID, revision=DINO_REVISION)
    model = AutoModel.from_pretrained(DINO_ID, revision=DINO_REVISION, dtype=torch.float16).eval().to(dev)
    return proc, model


@torch.no_grad()
def dino_cls(proc, model, frames_uint8, dev, batch=64):
    """L2-normalised CLS embeddings [N, 768] (float32) of N uint8 frames."""
    out = []
    for i in range(0, len(frames_uint8), batch):
        x = proc(images=list(frames_uint8[i:i + batch]), return_tensors="pt").to(dev)
        h = model(pixel_values=x["pixel_values"].half()).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out)


def load_vjepa(dev):
    from transformers import AutoModel, AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained(VJEPA_ID, revision=VJEPA_REVISION)
    model = AutoModel.from_pretrained(VJEPA_ID, revision=VJEPA_REVISION, dtype=torch.float16).eval().to(dev)
    return proc, model


@torch.no_grad()
def vjepa_embed(proc, model, clip_uint8, dev):
    """L2-normalised token-mean embedding [1024] (float32) of one clip [T, H, W, 3] uint8."""
    x = proc(videos=[list(clip_uint8)], return_tensors="pt").to(dev)                     # [1, T, C, H, W]
    h = model(pixel_values_videos=x["pixel_values_videos"].half()).last_hidden_state     # [1, N, D]
    return torch.nn.functional.normalize(h.float().mean(1), dim=-1)[0].cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--out", required=True, help="feature directory of the set (<set>/feats)")
    ap.add_argument("--prompt-ids", default="all", help="'all' (default), a subset name, or a comma list")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker takes the prompts whose list index modulo --num-shards equals --shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts of this shard only")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in 0..{a.num_shards - 1}")
    gs = shard(groups(a.videos_dir, prompt_ids(a.prompt_ids)), a.shard, a.num_shards)
    tag = f"{a.shard}/{a.num_shards}"
    if a.limit:
        gs = gs[:a.limit]
    od = os.path.join(a.out, "vdino", "dinov2"); os.makedirs(od, exist_ok=True)
    ov = os.path.join(a.out, "vjepa2"); os.makedirs(ov, exist_ok=True)
    pending = [g for g in gs if not os.path.exists(os.path.join(od, f"p{g[0]:03d}.npz"))
            or not os.path.exists(os.path.join(ov, f"p{g[0]:03d}.npy"))]
    print(f"[vdino {tag}] {len(pending)}/{len(gs)} prompts", flush=True)
    if not pending:
        return
    dev = a.device
    proc, model = load_dino(dev)
    vp = load_vjepa(dev)
    t0 = time.time()
    for n, (k, prompt, paths) in enumerate(pending, 1):
        fd = os.path.join(od, f"p{k:03d}.npz"); fv = os.path.join(ov, f"p{k:03d}.npy")
        if not os.path.exists(fd):
            clips = [read_frames(p, n_frames=DINO_FRAMES) for p in paths]                   # 10 x [F, H, W, 3]
            cls = dino_cls(proc, model, np.concatenate(clips), dev).reshape(len(clips), DINO_FRAMES, -1)
            save(fd, lambda fh: np.savez(fh, cls=cls.to(torch.float16).numpy(), prompt=prompt))
        if not os.path.exists(fv):
            embs = [vjepa_embed(*vp, read_frames(p, n_frames=VJEPA_FRAMES, size=VJEPA_SIZE), dev) for p in paths]
            save(fv, lambda fh: np.save(fh, torch.stack(embs).to(torch.float16).numpy()))
        if n % 10 == 0:
            el = time.time() - t0
            print(f"[vdino {tag}] {n}/{len(pending)}  {el / n:.1f}s/prompt  eta {(len(pending) - n) * el / n / 60:.0f} min", flush=True)
    print(f"[vdino {tag}] done {len(pending)} prompts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
