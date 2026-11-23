#!/usr/bin/env python
"""HPSv2.1 + ImageReward for many image sets with one model load (image env).

  python -m evaluation.image.score_many --sets DIR1 DIR2 ... --prompt-ids 0,11,17 --seeds 8 --out-name metrics_quality.json \
      --eval-root <root>      # writes <eval-root>/<basename of the set's parent>/metrics_quality.json

Each DIR holds <idx:03d>_s<seed>.png (an `images/` folder). Same scorers as run_metrics.py (evaluation/image/quality_*.py).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from evaluation.common.image_sets import flatten, load_groups
from rediverse.data import GENEVAL_553, parse_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="+", required=True, help="images directories")
    ap.add_argument("--names", nargs="+", default=None, help="output names (default: parent folder of each images dir)")
    ap.add_argument("--prompt-ids", required=True)
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out-name", default="metrics_quality.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from evaluation.image.quality_hpsv2 import HpsV2Score
    from evaluation.image.quality_imagereward import ImageRewardScore
    hps, ir = HpsV2Score(device=a.device), ImageRewardScore(device=a.device)
    ids = parse_ids(a.prompt_ids)
    names = a.names or [os.path.basename(os.path.dirname(os.path.abspath(d))) for d in a.sets]
    for d, name in zip(a.sets, names):
        out = os.path.join(a.eval_root, name, a.out_name)
        if os.path.exists(out):
            print(f"[skip] {name}")
            continue
        t0 = time.time()
        pairs = flatten(load_groups(d, GENEVAL_553, ids, a.seeds))
        r = {"images_dir": os.path.abspath(d), "prompt_ids": ids, "seeds": a.seeds, "num_images": len(pairs),
             "metrics": {"hpsv2": hps.compute(pairs), "imagereward": ir.compute(pairs)}}
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w") as f:
            json.dump(r, f)
        os.replace(out + ".tmp", out)
        print(f"[{name}] hpsv2 {r['metrics']['hpsv2']['mean']:.4f}  imagereward {r['metrics']['imagereward']['mean']:.4f}"
              f"  ({time.time() - t0:.0f}s, {len(pairs)} imgs)", flush=True)


if __name__ == "__main__":
    main()
