#!/usr/bin/env python
"""Simple colour / exposure statistics of image sets (quality diagnostics; numpy + PIL only).

  python -m qwenimage.image_stats --root <REDIVERSE_OUT> --sets qwen21/lightning/student_preview ... [--md out.md] [--json out.json]

Per set (all images under <root>/<set>/images, pooled over images):
  sat_mean   mean HSV saturation (0-1)
  sat_p99    99th percentile of HSV saturation over all pixels
  lum_mean   mean luminance Y = 0.299 R + 0.587 G + 0.114 B (0-1)
  contrast   mean over images of the per-image std of Y
  clip_lo    fraction of pixels with all channels <= 2/255 (crushed blacks)
  clip_hi    fraction of pixels with any channel >= 253/255 (blown highlights / clipped colours)
  n          images
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image


def stats_of(paths: list[str], max_side: int = 512) -> dict:
    sats, lums, cons, lo, hi, p99 = [], [], [], [], [], []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if max(im.size) > max_side:
            im = im.resize((max_side, max_side), Image.BILINEAR)
        rgb = np.asarray(im, dtype=np.float32) / 255.0
        mx, mn = rgb.max(-1), rgb.min(-1)
        s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        sats.append(s.mean())
        p99.append(np.percentile(s, 99))
        lums.append(y.mean())
        cons.append(y.std())
        lo.append(float((rgb.max(-1) <= 2 / 255).mean()))
        hi.append(float((rgb.max(-1) >= 253 / 255).mean()))
    return dict(n=len(paths), sat_mean=float(np.mean(sats)), sat_p99=float(np.mean(p99)), lum_mean=float(np.mean(lums)),
                contrast=float(np.mean(cons)), clip_lo=float(np.mean(lo)), clip_hi=float(np.mean(hi)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = {}
    for s in a.sets:
        paths = sorted(glob.glob(os.path.join(a.root, s, "images", "*.png")))
        rows[s] = stats_of(paths) if paths else dict(n=0)
    lines = ["| set | n | sat mean | sat p99 | lum mean | contrast | clip lo % | clip hi % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, r in rows.items():
        if not r["n"]:
            lines.append(f"| {s} | 0 | (missing) ||||||")
            continue
        lines.append(f"| {s} | {r['n']} | {r['sat_mean']:.3f} | {r['sat_p99']:.3f} | {r['lum_mean']:.3f} | "
                     f"{r['contrast']:.3f} | {100 * r['clip_lo']:.2f} | {100 * r['clip_hi']:.2f} |")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
