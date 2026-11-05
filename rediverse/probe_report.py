#!/usr/bin/env python
"""Aggregate probe records (rediverse.probe) into the paper's per-block view (Fig. 2, Tab. 1, App. A).

For every block: TV, PR and Conc of the teacher and of the student (means over prompts), and the
student/teacher ratio of each, as the mean of the per-prompt ratios with a 95% confidence interval
(1.96 standard errors). Only prompts probed in both arms are used. The collapsed block is marked;
its three ratios are the backbone's row of Tab. 1.

  python -m rediverse.probe_report --backbone sdxl
  python -m rediverse.probe_report --teacher DIR --student DIR --collapsed up0.attn0 --tsv out.tsv --plot out.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from rediverse.data import output_root

COLLAPSED = {"sdxl": "up0.attn0", "sd3": "block12", "wan2.1": "block28", "minimax-h3": "block45"}
QUANTITIES = (("pr", "PR"), ("tv", "TV"), ("conc", "Conc"))


def load_arm(d: str) -> dict:
    """{prompt_idx: {block: stats}} of one arm's directory of p*.json records."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "p*.json"))):
        with open(f) as fh:
            rec = json.load(fh)
        out[rec["prompt_idx"]] = {b["name"]: b for b in rec["blocks"]}
    if not out:
        raise SystemExit(f"no probe records in {d}")
    return out


def mean_ci(v) -> tuple[float, float]:
    v = np.asarray(v, dtype=np.float64)
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), float(half)


def summarize(T: dict, S: dict) -> tuple[list[dict], list[int]]:
    common = sorted(set(T) & set(S))
    if not common:
        raise SystemExit("the teacher and student directories share no prompt")
    blocks = [b for b in T[common[0]] if all(b in T[p] and b in S[p] for p in common)]
    rows = []
    for b in blocks:
        r = {"block": b}
        for key, _ in QUANTITIES:
            t = [T[p][b][key] for p in common]
            s = [S[p][b][key] for p in common]
            r[f"{key}_t"], r[f"{key}_s"] = float(np.mean(t)), float(np.mean(s))
            r[f"{key}_ratio"], r[f"{key}_ci"] = mean_ci(np.asarray(s) / np.asarray(t))
        rows.append(r)
    return rows, common


def write_tsv(rows, path: str):
    cols = ["block"] + [f"{k}_{a}" for k, _ in QUANTITIES for a in ("t", "s", "ratio", "ci")]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join([r["block"]] + [f"{r[c]:.6g}" for c in cols[1:]]) + "\n")


def plot(rows, collapsed: str, path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for a, (key, label) in zip(ax, QUANTITIES):
        m = np.array([r[f"{key}_ratio"] for r in rows])
        ci = np.array([r[f"{key}_ci"] for r in rows])
        a.axhline(1.0, color="#888", lw=1, label="teacher")
        a.plot(x, m, color="#c0392b", lw=1.8, marker="o", ms=3, label="student / teacher")
        a.fill_between(x, m - ci, m + ci, color="#c0392b", alpha=0.18, lw=0)
        names = [r["block"] for r in rows]
        if collapsed in names:
            a.axvline(names.index(collapsed), color="#2c3e50", ls="--", lw=1, label=f"collapsed block {collapsed}")
        a.set_title(f"{label} ratio")
        a.set_xticks(x[:: max(1, len(x) // 12)])
        a.set_xticklabels([names[i] for i in x[:: max(1, len(x) // 12)]], rotation=60, fontsize=7)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=sorted(COLLAPSED), default=None,
                    help="read <REDIVERSE_OUT>/<backbone>/probe/{teacher,student}")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--student", default=None)
    ap.add_argument("--collapsed", default=None, help="block to mark (default: the backbone's collapsed block)")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    if a.backbone:
        base = os.path.join(output_root(), a.backbone, "probe")
        a.teacher = a.teacher or os.path.join(base, "teacher")
        a.student = a.student or os.path.join(base, "student")
        a.collapsed = a.collapsed or COLLAPSED[a.backbone]
    if not (a.teacher and a.student):
        ap.error("give --backbone, or --teacher and --student")

    rows, common = summarize(load_arm(a.teacher), load_arm(a.student))
    print(f"{len(common)} prompts probed in both arms; ratios = student / teacher, mean of per-prompt ratios "
          f"(+- 95% CI)\n")
    def cell(r, key):
        ci = r[f"{key}_ci"]
        return f"{r[f'{key}_ratio']:>9.2f}" + (f" +-{ci:<4.2f}" if np.isfinite(ci) else " " * 7)

    print(f"{'block':<18}{'PR t':>9}{'PR s':>9}{'PR s/t':>16}{'TV s/t':>16}{'Conc s/t':>16}")
    for r in rows:
        mark = "  <- collapsed block" if r["block"] == a.collapsed else ""
        print(f"{r['block']:<18}{r['pr_t']:>9.1f}{r['pr_s']:>9.1f}"
              f"{cell(r, 'pr')}{cell(r, 'tv')}{cell(r, 'conc')}{mark}")
    hit = [r for r in rows if r["block"] == a.collapsed]
    if hit:
        r = hit[0]
        print(f"\nTab. 1 row at {a.collapsed}: PR {r['pr_ratio']:.2f}  TV {r['tv_ratio']:.2f}  "
              f"Conc {r['conc_ratio']:.2f}")
    if a.tsv:
        write_tsv(rows, a.tsv)
        print(f"tsv -> {a.tsv}")
    if a.plot:
        plot(rows, a.collapsed, a.plot, f"{a.backbone or ''} ({len(common)} prompts)".strip())


if __name__ == "__main__":
    main()
