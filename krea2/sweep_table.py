#!/usr/bin/env python
"""Collect sweep metrics (CPU diversity + HPSv2/ImageReward) into one table. Standard library only.

  python3 krea2/sweep_table.py [--root <REDIVERSE_OUT>/krea2/eval] [--md out.md]
"""
import argparse
import glob
import json
import os

REF = ["preview_student", "preview_tf_b26", "preview_ft_b26_mse_s50", "preview_rd_b26_mse_s50", "preview_teacher"]


def load(root, name):
    r = {}
    for part in ("metrics_cpu.json", "metrics_quality.json"):
        f = os.path.join(root, name, part)
        if os.path.exists(f):
            m = json.load(open(f))["metrics"]
            for k, v in m.items():
                r[k] = v["mean"] if isinstance(v, dict) else v
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("REDIVERSE_OUT", "outputs"), "krea2", "eval"))
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    names = REF + sorted("sweep/" + os.path.basename(d.rstrip("/")) for d in glob.glob(os.path.join(a.root, "sweep", "*/")))
    rows = [(n, load(a.root, n)) for n in names]
    S = dict(rows).get("preview_student", {})
    T = dict(rows).get("preview_teacher", {})
    gap = (T.get("dino", 0) - S.get("dino", 0)) or 1.0
    hdr = "| config | DINO | gap closed | Vendi | DreamSim | CLIPScore | HPSv2 | ΔHPS | ImageReward | ΔIR |"
    lines = [hdr, "|" + "---|" * 10]
    for n, m in rows:
        if "dino" not in m:
            lines.append(f"| {n} | (pending) |||||||||")
            continue
        g = 100 * (m["dino"] - S.get("dino", 0)) / gap
        hps, ir = m.get("hpsv2"), m.get("imagereward")
        dh = f"{100 * (hps - S['hpsv2']):+.2f}" if hps is not None and "hpsv2" in S else "-"
        di = f"{ir - S['imagereward']:+.3f}" if ir is not None and "imagereward" in S else "-"
        lines.append(f"| {n.replace('sweep/', '')} | {100 * m['dino']:.1f} | {g:.0f}% | {m['vendi']:.2f} | "
                     f"{100 * m['dreamsim']:.1f} | {100 * m['clipscore']:.1f} | "
                     f"{'-' if hps is None else f'{100 * hps:.2f}'} | {dh} | {'-' if ir is None else f'{ir:.3f}'} | {di} |")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")


if __name__ == "__main__":
    main()
