"""Assemble <arm_dir>/metrics_ablate.json from the stage outputs of evaluation/image/score_ablation.sh:

  <arm_dir>/eval/metrics_cpu.json   evaluation.image.feature_metrics: dino, vendi, dreamsim, clipscore,
                                    tpips_{overall,subject,background,camera,lighting} (per_prompt {"p027": v})
  <arm_dir>/eval/metrics_gpu.json   evaluation.image.run_metrics: hpsv2, imagereward (per_image, prompt-major,
                                    seed-minor over the sorted prompt indices, R images each)

Output {metric: {mean, std_over_prompts, n_prompts, per_prompt {"p027": v}}}; `mean` is the mean of the
per-prompt means (equal to the per-image mean at equal R). Exits non-zero unless every --require metric
is present with the expected prompt count and the images form the full grid.

  python -m evaluation.image.collect_ablation --arm-dir <arm_dir> --n-prompts 50 --seeds 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from evaluation.common.image_sets import FNAME_RE

TABLE_METRICS = ["dino", "dreamsim", "tpips_overall", "vendi", "clipscore", "hpsv2", "imagereward",
                 "tpips_subject", "tpips_background", "tpips_lighting", "tpips_camera"]


def pstd(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--eval-dir", default=None, help="default <arm-dir>/eval")
    ap.add_argument("--out", default=None, help="default <arm-dir>/metrics_ablate.json")
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--require", default=",".join(TABLE_METRICS), help="metrics that must be present ('' = none)")
    a = ap.parse_args()
    ev = a.eval_dir or os.path.join(a.arm_dir, "eval")
    out_path = a.out or os.path.join(a.arm_dir, "metrics_ablate.json")
    images = os.path.join(a.arm_dir, "images")
    idxs = sorted({int(m.group(1)) for f in os.listdir(images) for m in [FNAME_RE.match(f)] if m})
    tags = [f"p{i:03d}" for i in idxs]
    res = {}

    cpu = os.path.join(ev, "metrics_cpu.json")
    if os.path.exists(cpu):
        for k, v in json.load(open(cpu))["metrics"].items():
            if isinstance(v, dict) and "per_prompt" in v:
                pp = {t: float(x) for t, x in v["per_prompt"].items()}
                vals = list(pp.values())
                res[k] = {"mean": sum(vals) / len(vals), "std_over_prompts": pstd(vals), "n_prompts": len(vals),
                          "per_prompt": pp, "source": "eval/metrics_cpu.json"}
    gpu = os.path.join(ev, "metrics_gpu.json")
    if os.path.exists(gpu):
        j = json.load(open(gpu)); R = j["num_images"] // j["num_prompts"]
        assert j["num_prompts"] == len(tags), f"{gpu}: {j['num_prompts']} prompts scored, {len(tags)} in images/"
        for k, v in j["metrics"].items():
            if "per_image" in v:                       # hpsv2 / imagereward / pickscore
                pi = v["per_image"]; assert len(pi) == R * len(tags), (k, len(pi), R, len(tags))
                pp = {t: sum(pi[q * R:(q + 1) * R]) / R for q, t in enumerate(tags)}
            elif isinstance(v.get("per_prompt"), list):  # lpips
                pp = {t: float(x) for t, x in zip(tags, v["per_prompt"])}
            else:
                continue
            vals = list(pp.values())
            res[k] = {"mean": sum(vals) / len(vals), "std_over_prompts": pstd(vals), "n_prompts": len(vals),
                      "per_prompt": pp, "source": "eval/metrics_gpu.json"}

    missing = [m for m in (a.require.split(",") if a.require else []) if m not in res]
    bad_n = {m: r["n_prompts"] for m, r in res.items() if r["n_prompts"] != a.n_prompts}
    n_img = sum(1 for f in os.listdir(images) if FNAME_RE.match(f))
    out = {"arm": os.path.basename(os.path.normpath(a.arm_dir)), "n_prompts": len(tags), "prompt_ids": idxs,
           "seeds": a.seeds, "n_images": n_img,
           "protocol": f"{len(tags)} prompts x {a.seeds} seeds, all C({a.seeds},2) pairs per prompt, mean over prompts",
           "table_metrics": TABLE_METRICS, "complete": not missing and not bad_n and n_img == len(tags) * a.seeds,
           "missing": missing, "metrics": res}
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    for m in TABLE_METRICS:
        print(f"  {m:16s} {res[m]['mean']:.4f} ± {res[m]['std_over_prompts']:.4f}  (n={res[m]['n_prompts']})" if m in res
              else f"  {m:16s} MISSING")
    print(f"[collect] wrote {out_path}  complete={out['complete']}")
    if missing or bad_n or n_img != len(tags) * a.seeds:
        print(f"[collect] INCOMPLETE: missing={missing} wrong_n_prompts={bad_n} n_images={n_img} "
              f"(expected {len(tags) * a.seeds})", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
