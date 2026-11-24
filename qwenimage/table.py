#!/usr/bin/env python
"""Markdown tables of Qwen-Image eval sets (standard library only).

  python3 qwenimage/table.py --root <REDIVERSE_OUT>/eval --teacher qwen21/teacher/teacher_preview \
      --group lightning=qwen21/lightning/student_preview,qwen21/lightning/rediverse_preview ... [--md out.md]

Each --group is <label>=<student baseline set>,<other sets...>: rows are the baseline, the other sets and the
teacher; "gap" columns are the % of the baseline -> teacher gap closed (DINO, Vendi, DreamSim); quality columns are
CLIPScore, HPSv2 (x100) and ImageReward with their change from the baseline. Metrics are read from
metrics_cpu.json and metrics_quality.json / metrics_gpu.json of each set.
"""
import argparse
import json
import os


def load(root, name):
    r = {}
    for part in ("metrics_cpu.json", "metrics_quality.json", "metrics_gpu.json"):
        f = os.path.join(root, name, part)
        if os.path.exists(f):
            for k, v in json.load(open(f))["metrics"].items():
                if isinstance(v, dict) and "mean" in v:
                    r[k] = v["mean"]
    return r


def fmt_gap(m, s, t, k):
    if k not in m or k not in s or k not in t or abs(t[k] - s[k]) < 1e-12:
        return "-"
    return f"{100 * (m[k] - s[k]) / (t[k] - s[k]):.0f}%"


def rows_for(root, label, sets, teacher):
    S, T = load(root, sets[0]), load(root, teacher)
    out = []
    for name in sets + [teacher]:
        m = load(root, name)
        if "dino" not in m:
            out.append(f"| {label} | {name} | (pending) |||||||||||")
            continue
        d = lambda k, sc=1.0, p=2: "-" if k not in m else f"{sc * m[k]:.{p}f}"
        dd = lambda k, sc=1.0, p=2: "-" if k not in m or k not in S else f"{sc * (m[k] - S[k]):+.{p}f}"
        out.append(f"| {label} | {name} | {d('dino', 100, 1)} | {fmt_gap(m, S, T, 'dino')} | {d('vendi')} | "
                   f"{fmt_gap(m, S, T, 'vendi')} | {d('dreamsim', 100, 1)} | {fmt_gap(m, S, T, 'dreamsim')} | "
                   f"{d('clipscore', 100, 1)} | {d('hpsv2', 100)} | {dd('hpsv2', 100)} | {d('imagereward', 1, 3)} | "
                   f"{dd('imagereward', 1, 3)} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("REDIVERSE_OUT", "outputs"), "eval"))
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    lines = ["| student | set | DINO | gap | Vendi | gap | DreamSim | gap | CLIPScore | HPSv2 | dHPS | ImageReward | dIR |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in a.group:
        label, sets = g.split("=", 1)
        lines += rows_for(a.root, label, sets.split(","), a.teacher)
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")


if __name__ == "__main__":
    main()
