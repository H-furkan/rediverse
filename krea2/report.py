#!/usr/bin/env python
"""Seed-matched grids and a metric table for the Krea 2 arms (CPU only).

  python -m krea2.report --grid --prompt-ids 0,27,131,381 --sets student,rediverse_tf_b26,teacher --seeds 8
  python -m krea2.report --table --sets student,rediverse_tf_b26,rediverse_ft_b26_mse,rediverse_b26_mse,teacher

Grids: one JPEG per prompt, a row per set, column s = sample s (same initial noise in every set).
Table: reads <out>/krea2/eval/<set>/metrics_{cpu,gpu}.json written by evaluation/image/evaluate_set.sh.
"""
from __future__ import annotations

import argparse
import json
import os

from krea2 import config as C
from rediverse import data as D


def grid(sets, ids, seeds, out_dir, cell=256):
    from PIL import Image, ImageDraw
    texts = D.read_prompts(D.GENEVAL_553)
    os.makedirs(out_dir, exist_ok=True)
    for i in ids:
        g = Image.new("RGB", (cell * seeds + 220, cell * len(sets) + 30), "white")
        d = ImageDraw.Draw(g)
        d.text((5, 8), f"p{i:03d}: {texts[i]}", fill="black")
        for r, s in enumerate(sets):
            d.text((5, 30 + r * cell + cell // 2), s, fill="black")
            for k in range(seeds):
                f = os.path.join(C.output_dir(s), "images", f"{i:03d}_s{k}.png")
                if os.path.exists(f):
                    g.paste(Image.open(f).convert("RGB").resize((cell, cell)), (220 + k * cell, 30 + r * cell))
        p = os.path.join(out_dir, f"p{i:03d}.jpg")
        g.save(p, quality=88)
        print(p)


def table(sets):
    keys = None
    rows = []
    for s in sets:
        m = {}
        for part in ("metrics_cpu.json", "metrics_gpu.json"):
            f = os.path.join(C.output_dir("eval", s), part)
            if os.path.exists(f):
                with open(f) as fh:
                    d = json.load(fh)
                m.update(d.get("metrics", d) if isinstance(d, dict) else {})
        flat = {}
        for k, v in m.items():
            if isinstance(v, dict) and "mean" in v:
                flat[k] = v["mean"]
            elif isinstance(v, (int, float)):
                flat[k] = v
        rows.append((s, flat))
        keys = sorted(set(keys or []) | set(flat))
    if not keys:
        print("no metrics yet")
        return
    print("set".ljust(28) + "".join(k[:12].rjust(13) for k in keys))
    for s, flat in rows:
        print(s.ljust(28) + "".join((f"{flat[k]:.4g}" if k in flat else "-").rjust(13) for k in keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", required=True)
    ap.add_argument("--prompt-ids", default="0,27,131,381")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out", default=None, help="default <out>/krea2/grids")
    a = ap.parse_args()
    sets = a.sets.split(",")
    if a.grid:
        grid(sets, D.parse_ids(a.prompt_ids), a.seeds, a.out or C.output_dir("grids"))
    if a.table:
        table(sets)


if __name__ == "__main__":
    main()
