"""Stage H of the image evaluation: GPU metrics of one sample set (paper App. F).

  lpips        diversity: mean pairwise LPIPS-VGG per prompt, mean over prompts
  hpsv2        quality: HPSv2.1 per image, mean over images
  imagereward  quality: ImageReward per image, mean over images
  pickscore    optional quality metric (not in the paper tables)

  CUDA_VISIBLE_DEVICES=0 python -m evaluation.image.run_metrics --images-dir <set>/images \\
      --out <set>/metrics_gpu.json [--metrics lpips hpsv2 imagereward] [--shard i --num-shards n]

Output: per metric {mean, per_prompt (lpips) or per_image (quality; prompt-major, seed-minor)}.
The file is written fresh on every run (after each metric). With --shard i --num-shards n the run
keeps the prompts with list index k % n == i; merge the n outputs with evaluation.image.merge_shards.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

from evaluation.common.image_sets import SEEDS, flatten, load_groups, shard
from rediverse.data import GENEVAL_553, parse_ids

DIVERSITY = ["lpips"]
QUALITY = ["hpsv2", "imagereward", "pickscore"]
DEFAULT = ["lpips", "hpsv2", "imagereward"]


def build(name: str, device: str, a):
    if name == "lpips":
        from evaluation.image.diversity_lpips import LpipsDiversity
        return LpipsDiversity(device=device, pair_batch=a.lpips_pair_batch)
    if name == "hpsv2":
        from evaluation.image.quality_hpsv2 import HpsV2Score
        return HpsV2Score(device=device, batch_size=a.batch)
    if name == "imagereward":
        from evaluation.image.quality_imagereward import ImageRewardScore
        return ImageRewardScore(device=device)
    if name == "pickscore":
        from evaluation.image.quality_pickscore import PickScore
        return PickScore(device=device, batch_size=a.batch)
    raise ValueError(f"unknown metric {name}")


def write(path, results):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(results, f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--prompts", default=GENEVAL_553)
    ap.add_argument("--prompt-ids", default=None, help="comma list / a-b ranges / file (default: every prompt)")
    ap.add_argument("--seeds", type=int, default=SEEDS, help="samples per prompt the set must hold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", nargs="+", default=DEFAULT, choices=DIVERSITY + QUALITY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker takes the prompts whose list index modulo --num-shards equals --shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--lpips-pair-batch", type=int, default=64, help="image pairs per LPIPS forward (value-neutral)")
    ap.add_argument("--batch", type=int, default=32, help="images per forward of HPSv2 / PickScore")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in 0..{a.num_shards - 1}")

    ids = parse_ids(a.prompt_ids) if a.prompt_ids else None
    groups = shard(load_groups(a.images_dir, a.prompts, ids, a.seeds), a.shard, a.num_shards)
    pairs = flatten(groups)
    print(f"{len(groups)} prompts, {len(pairs)} images (R={a.seeds}) in {a.images_dir}")

    results = {"images_dir": os.path.abspath(a.images_dir), "prompts": os.path.basename(a.prompts),
               "prompt_ids": [g.prompt_idx for g in groups], "num_prompts": len(groups),
               "num_images": len(pairs), "shard": f"{a.shard}/{a.num_shards}", "metrics": {}}
    for name in a.metrics:
        t0 = time.time()
        metric = build(name, a.device, a)
        res = metric.compute(pairs) if name in QUALITY else metric.compute(groups)
        res["seconds"] = round(time.time() - t0, 1)
        results["metrics"][name] = res
        print(f"[{name}] mean = {res['mean']:.4f}  ({res['seconds']}s)", flush=True)
        del metric
        gc.collect()
        torch.cuda.empty_cache()
        write(a.out, results)

    print("\n=== summary ===")
    for name, res in results["metrics"].items():
        print(f"{name:12s} {res['mean']:.4f}")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
