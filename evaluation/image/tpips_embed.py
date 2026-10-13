"""Stage T of the image evaluation: TPIPS-Embed embeddings (Wang et al., "The Many Senses of
Visual Similarity"), one file per prompt and factor. TPIPS-Embed encodes an image with a text
factor into a 4096-d vector; similarity = cosine. The cached L2-normalised embeddings make the
per-prompt diversity (1 - mean pairwise cosine, evaluation.image.feature_metrics) a numpy pass.

Output <out>/tpips/<factor>/p<idx>.npy  (R, 4096) fp16. Runs in the `tpips` environment
(transformers 5.x); select the GPU with CUDA_VISIBLE_DEVICES.

  CUDA_VISIBLE_DEVICES=0 python -m evaluation.image.tpips_embed --images-dir <set>/images --out <set>/feats --factor overall
  CUDA_VISIBLE_DEVICES=0 python -m evaluation.image.tpips_embed --images-dir <set>/images --out <set>/feats \\
      --factor subject --prompt-ids prompts/geneval_subset100.txt

Prompts are claimed one at a time (<out>/tpips/<factor>/.claims/p<idx>.claim), so any number of
workers on one host share the remaining prompts; a claim whose process has exited is taken
over. A prompt whose embedding file is newer than all of its images is skipped.
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
from evaluation.common.image_sets import SEEDS, load_groups, shard, up_to_date
from evaluation.common.tpips_factors import FACTORS, load_model, slug
from rediverse.claims import claim_tag, release_tag
from rediverse.data import GENEVAL_553, parse_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--prompts", default=GENEVAL_553)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", default="overall", help="slug from FACTORS or free text")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker takes the prompts whose list index modulo --num-shards equals --shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--prompt-ids", default=None, help="comma list / a-b ranges / file (default: every prompt)")
    ap.add_argument("--seeds", type=int, default=SEEDS, help="samples per prompt the set must hold")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in 0..{a.num_shards - 1}")
    factor = FACTORS.get(a.factor, a.factor); sl = slug(a.factor)
    ids = parse_ids(a.prompt_ids) if a.prompt_ids else None
    groups = shard(load_groups(a.images_dir, a.prompts, ids, a.seeds), a.shard, a.num_shards)
    od = os.path.join(a.out, "tpips", sl); os.makedirs(od, exist_ok=True)
    claims = os.path.join(od, ".claims")

    model = load_model("cuda")
    pending = [g for g in groups if not up_to_date(os.path.join(od, f"p{g.prompt_idx:03d}.npy"), g.paths)]

    # decode the next batches on a background thread so the GPU never waits on PNG I/O
    def producer(qout):
        for g in pending:
            tag = f"p{g.prompt_idx:03d}"
            if up_to_date(os.path.join(od, tag + ".npy"), g.paths) or not claim_tag(claims, tag):
                continue
            for k in range(0, len(g.paths), a.batch):
                qout.put((g, k, [Image.open(p).convert("RGB") for p in g.paths[k:k + a.batch]]))
        qout.put(None)
    qb = queue.Queue(maxsize=6); threading.Thread(target=producer, args=(qb,), daemon=True).start()

    t0 = time.time(); n_img = 0; gi = -1; cur = None; embs = []
    with torch.inference_mode():
        while True:
            item = qb.get()
            if item is None:
                break
            g, k, imgs = item
            if g is not cur:
                cur = g; gi += 1; embs = []
            tag = f"p{g.prompt_idx:03d}"; f = os.path.join(od, tag + ".npy")
            e = model.embed(imgs, factor=factor)
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
            if k + a.batch < len(g.paths):
                continue
            E = torch.cat(embs).to(torch.float16).numpy()
            save(f, lambda fh: np.save(fh, E)); n_img += len(g.paths)
            release_tag(claims, tag)
            if gi % 5 == 0:
                el = time.time() - t0
                print(f"[tpips {sl} {a.shard}/{a.num_shards}] {gi+1}/{len(pending)} {tag}  {n_img/el:.2f} img/s  "
                      f"eta {(len(pending)-gi-1)*el/(gi+1)/60:.0f} min", flush=True)
    print(f"[tpips {sl} {a.shard}/{a.num_shards}] done {len(pending)} prompts in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
