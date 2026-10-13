"""ViCLIP video and prompt embeddings of a video set (main comparison table: VC, VCS; per-aspect TPIPS
table: Vendi).
Runs in the vbench environment with VBench's bundled ViCLIP (ViClip-InternVid-10M-FLT).

Per clip: VBench's read_frames_decord_by_fps(num_frames=8, sample='middle') (the middle frame of each of 8
equal segments) and clip_transform(224), then encode_vision(test=True), L2-normalised. Per prompt: encode_text
of the prompt text, L2-normalised. Saved to <out>/viclip/p<idx>.npz {video [10, 768] fp16, text [768] fp16,
prompt}. video_metrics.py derives
  viclip_div    1 - mean pairwise cosine over the 10 clips              (VC)
  viclip_vendi  Vendi score of the 10 clips in ViCLIP space
  viclip_text   mean cosine between clip and prompt embeddings          (VCS; the computation of VBench's
                overall_consistency, applied to all 300 prompts)

The ViCLIP weights and BPE vocabulary are downloaded by VBench into $VBENCH_CACHE_DIR/ViCLIP
(default ~/.cache/vbench) on first use.

  python -m evaluation.video.viclip_scores --videos-dir <set>/videos --out <set>/feats [--shard i --num-shards 2]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from evaluation.common.atomic import save
from evaluation.video.sets import groups, prompt_ids, shard

FRAMES = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--out", required=True, help="feature directory of the set (<set>/feats)")
    ap.add_argument("--prompt-ids", default="all", help="'all' (default), a subset name, or a comma list")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker takes the prompts whose list index modulo --num-shards equals --shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts of this shard only")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in 0..{a.num_shards - 1}")
    gs = shard(groups(a.videos_dir, prompt_ids(a.prompt_ids)), a.shard, a.num_shards)
    tag = f"{a.shard}/{a.num_shards}"
    if a.limit:
        gs = gs[:a.limit]
    od = os.path.join(a.out, "viclip"); os.makedirs(od, exist_ok=True)
    pending = [g for g in gs if not os.path.exists(os.path.join(od, f"p{g[0]:03d}.npz"))]
    print(f"[viclip {tag}] {len(pending)}/{len(gs)} prompts", flush=True)
    if not pending:
        return
    from vbench.utils import CACHE_DIR, clip_transform, init_submodules, read_frames_decord_by_fps
    from vbench.third_party.ViCLIP.viclip import ViCLIP
    from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
    dev = torch.device(a.device)
    sub = init_submodules(["overall_consistency"])["overall_consistency"]          # downloads the weights on first use
    tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
    model = ViCLIP(tokenizer=tokenizer, **sub).to(dev).eval()
    tf = clip_transform(224)
    t0 = time.time()
    with torch.no_grad():
        for n, (k, prompt, paths) in enumerate(pending, 1):
            vids = []
            for p in paths:
                frames = tf(read_frames_decord_by_fps(p, num_frames=FRAMES, sample="middle")).to(dev)   # [8, 3, 224, 224]
                v = model.encode_vision(frames.unsqueeze(0), test=True).float()
                vids.append((v / v.norm(dim=-1, keepdim=True))[0].cpu())
            t = model.encode_text(prompt).float(); t = (t / t.norm(dim=-1, keepdim=True))[0].cpu()
            save(os.path.join(od, f"p{k:03d}.npz"),
                 lambda fh: np.savez(fh, video=torch.stack(vids).to(torch.float16).numpy(),
                                     text=t.to(torch.float16).numpy(), prompt=prompt))
            if n % 10 == 0:
                el = time.time() - t0
                print(f"[viclip {tag}] {n}/{len(pending)}  {el / n:.1f}s/prompt  eta {(len(pending) - n) * el / n / 60:.0f} min", flush=True)
    print(f"[viclip {tag}] done {len(pending)} prompts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
