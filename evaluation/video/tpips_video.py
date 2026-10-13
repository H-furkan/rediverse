"""Frame-wise TPIPS embeddings of a video set (main comparison table: TP; per-aspect TPIPS table).

TPIPS-Embed is an image model, so clips are compared through sampled frames: 8 frames per clip at
linspace(0, T-1, 8).round(), passed at native resolution as PIL images to TPIPS-Embed (Qwen3-VL-8B, at the
checkpoint revision pinned in evaluation/common/tpips_factors.py) with the aspect instruction ("Represent
the similarity of the image." for overall, "Represent the similarity of the image based on <aspect text>."
otherwise), batch 16, L2-normalised  ->  <out>/tpips_video/<aspect>/p<idx>.npy [10, 8, 4096] fp16.
video_metrics.py takes, per frame index, 1 - mean pairwise cosine over the 10 clips and averages over the
8 frame indices.

Prompts: 'overall' runs on all 300 prompts, the four aspects on prompts/vbench_subset100.txt (the defaults).
Workers claim prompts through <out>/tpips_video/<aspect>/.claims, so any number of processes can share a set.

  python -m evaluation.video.tpips_video --videos-dir <set>/videos --out <set>/feats --factor overall
  python -m evaluation.video.tpips_video --videos-dir <set>/videos --out <set>/feats --factor subject
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time

import numpy as np
import torch
from PIL import Image

from evaluation.common.atomic import save
from evaluation.common.tpips_factors import FACTORS, load_model
from evaluation.video.sets import groups, prompt_ids
from rediverse.claims import claim_tag, release_tag
from rediverse.video_io import read_frames

DEFAULT_PROMPTS = {"overall": "all", "subject": "subset100", "background": "subset100", "camera": "subset100",
                   "lighting": "subset100"}
FRAMES = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--out", required=True, help="feature directory of the set (<set>/feats)")
    ap.add_argument("--factor", default="overall", choices=list(FACTORS))
    ap.add_argument("--prompt-ids", default=None, help="default: all 300 for overall, subset100 for the aspects")
    ap.add_argument("--batch", type=int, default=16, help="frames per TPIPS call (16 for the reference caches)")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    a = ap.parse_args()
    factor = FACTORS[a.factor]
    gs = groups(a.videos_dir, prompt_ids(a.prompt_ids or DEFAULT_PROMPTS[a.factor]))
    if a.limit:
        gs = gs[:a.limit]
    od = os.path.join(a.out, "tpips_video", a.factor); os.makedirs(od, exist_ok=True)
    claims = os.path.join(od, ".claims")
    pending = [g for g in gs if not os.path.exists(os.path.join(od, f"p{g[0]:03d}.npy"))]
    print(f"[tpips-video {a.factor}] {len(pending)}/{len(gs)} prompts pending, {FRAMES} frames/clip", flush=True)
    if not pending:
        return
    model = load_model("cuda")

    def producer(qout):
        for (k, prompt, paths) in pending:
            tag = f"p{k:03d}"
            if os.path.exists(os.path.join(od, tag + ".npy")) or not claim_tag(claims, tag):
                continue
            frames = [Image.fromarray(fr) for p in paths for fr in read_frames(p, n_frames=FRAMES)]   # 10*F images
            qout.put((k, frames))
        qout.put(None)
    qb = queue.Queue(maxsize=3)
    threading.Thread(target=producer, args=(qb,), daemon=True).start()

    t0, n = time.time(), 0
    with torch.inference_mode():
        while True:
            item = qb.get()
            if item is None:
                break
            k, frames = item
            embs = []
            for i in range(0, len(frames), a.batch):
                e = model.embed(frames[i:i + a.batch], factor=factor)
                embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
            E = torch.cat(embs).reshape(len(frames) // FRAMES, FRAMES, -1).to(torch.float16).numpy()
            save(os.path.join(od, f"p{k:03d}.npy"), lambda fh: np.save(fh, E)); n += 1
            release_tag(claims, f"p{k:03d}")
            if n % 5 == 0:
                el = time.time() - t0
                print(f"[tpips-video {a.factor}] {n} prompts  {n * len(frames) / el:.1f} frames/s  "
                      f"eta {(len(pending) - n) * el / n / 60:.0f} min", flush=True)
    print(f"[tpips-video {a.factor}] done {n} prompts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
