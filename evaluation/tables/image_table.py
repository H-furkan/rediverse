"""Image rows of the main comparison table (images) and the SD3 ablation table (plus the columns of
the design-choice tables and the image half of the per-aspect TPIPS table, supplementary) from
per-set metric files. Standard library only.

Main protocol (553 prompts x 100 seeds), one directory per arm as written by
evaluation/image/evaluate_set.sh:
  <dir>/metrics_cpu.json   dino dreamsim vendi clipscore tpips_{overall,subject,background,lighting,camera}
  <dir>/metrics_gpu.json   lpips hpsv2 imagereward
Ablation protocol (SD3, 50 prompts x 100 seeds), one directory per arm as written by
evaluation/image/score_ablation.sh:
  <dir>/metrics_ablate.json

Printed cells follow the paper: DINO, DS (DreamSim), LP (LPIPS), CS (CLIPScore), HPS (HPSv2.1)
and the five TPIPS values x100 with one decimal; Vendi with two decimals; IR (ImageReward) with
three decimals. TP = the mean of the five TPIPS cells (overall, subject, background, lighting,
camera) after rounding each to one decimal, printed with one decimal. Diversity metrics are means
over prompts, quality metrics means over images; TPIPS overall uses all prompts and the four
aspects the 100 prompts of prompts/geneval_subset100.txt (all 50 prompts in the ablation).

  python -m evaluation.tables.image_table \\
      --block SDXL teacher=outputs/sdxl/teacher student=... rediverse_tf=... rediverse_ft=... rediverse=... \\
      --block SD3  teacher=outputs/sd3/teacher  ... [--csv image_table.csv] [--latex image_table.tex]
  python -m evaluation.tables.image_table --protocol ablation --block SD3 teacher=<arm dir> student=<arm dir> ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ASPECTS = ("subject", "background", "lighting", "camera")
TP_KEYS = ["tpips_overall"] + [f"tpips_{s}" for s in ASPECTS]
FEATURE_KEYS = ["dino", "dreamsim", "vendi", "clipscore"] + TP_KEYS
GPU_KEYS = ["lpips", "hpsv2", "imagereward"]
GPU_FILES = ("metrics_gpu.json",)
PROTOCOLS = {"main": {"prompts": 553, "aspect_prompts": 100}, "ablation": {"prompts": 50, "aspect_prompts": 50}}

# (column, metric key or "TP", scale, decimals)
COLUMNS = {
    "main": [("DINO", "dino", 100, 1), ("DS", "dreamsim", 100, 1), ("LP", "lpips", 100, 1), ("TP", "TP", 100, 1),
             ("Vendi", "vendi", 1, 2), ("CS", "clipscore", 100, 1), ("HPS", "hpsv2", 100, 1), ("IR", "imagereward", 1, 3)],
    "ablation": [("DINO", "dino", 100, 1), ("DS", "dreamsim", 100, 1), ("TPIPS", "tpips_overall", 100, 1),
                 ("Vendi", "vendi", 1, 2), ("CS", "clipscore", 100, 1), ("HPS", "hpsv2", 100, 1),
                 ("IR", "imagereward", 1, 3), ("TP", "TP", 100, 1)],
}
TPIPS_COLUMNS = [("Overall", "tpips_overall"), ("Subject", "tpips_subject"), ("Backgr.", "tpips_background"),
                 ("Lighting", "tpips_lighting"), ("Camera", "tpips_camera")]
LABELS = {"teacher": "Teacher", "student": "Student", "rediverse_tf": "ReDiverse-TF",
          "rediverse_ft": "ReDiverse-FT", "rediverse": "ReDiverse"}


def fmt(v, scale, nd):
    return f"{v * scale:.{nd}f}"


def tp_cell(means):
    """Mean of the five one-decimal TPIPS cells. Five one-decimal values sum to a multiple of 0.1,
    so their mean has an even second decimal and the one-decimal rounding never meets a tie."""
    if any(k not in means for k in TP_KEYS):
        return None
    return f"{sum(float(fmt(means[k], 100, 1)) for k in TP_KEYS) / len(TP_KEYS):.1f}"


def _load(path):
    with open(path) as f:
        return json.load(f)["metrics"]


def load_main(d, seeds):
    """{metric: {mean, n}} of one main-protocol set directory."""
    out = {}
    for fn in GPU_FILES:
        f = os.path.join(d, fn)
        if os.path.exists(f):
            for k, v in _load(f).items():
                if k in GPU_KEYS and isinstance(v, dict) and "mean" in v:
                    if "per_image" in v:
                        n_img = len(v["per_image"])
                        assert n_img % seeds == 0, f"{f}: {k} has {n_img} images, not a multiple of {seeds} seeds"
                        out[k] = {"mean": v["mean"], "n": n_img // seeds}
                    else:
                        out[k] = {"mean": v["mean"], "n": len(v["per_prompt"])}
    f = os.path.join(d, "metrics_cpu.json")
    if os.path.exists(f):
        for k, v in _load(f).items():
            if k in FEATURE_KEYS:
                out[k] = {"mean": v["mean"], "n": v["n_prompts"]}
    return out


def load_ablation(d, seeds):
    """{metric: {mean, n}} of one ablation arm directory."""
    f = os.path.join(d, "metrics_ablate.json")
    j = json.load(open(f))
    assert j.get("seeds", seeds) == seeds, f"{f}: scored with {j.get('seeds')} seeds, expected {seeds}"
    return {k: {"mean": v["mean"], "n": v["n_prompts"]} for k, v in j["metrics"].items() if k in FEATURE_KEYS + GPU_KEYS}


def check(label, m, protocol):
    p = PROTOCOLS[protocol]
    for k, v in m.items():
        want = p["aspect_prompts"] if k in TP_KEYS[1:] else p["prompts"]
        if v["n"] != want:
            raise SystemExit(f"[table] {label}: {k} covers {v['n']} prompts, the {protocol} protocol has {want}")
    missing = [k for k in FEATURE_KEYS + GPU_KEYS if k not in m and not (protocol == "ablation" and k == "lpips")]
    if missing:
        print(f"[table] {label}: no value for {', '.join(missing)}", file=sys.stderr)


def cells(m, cols):
    means = {k: v["mean"] for k, v in m.items()}
    out = []
    for _, key, scale, nd in cols:
        if key == "TP":
            out.append(tp_cell(means))
        else:
            out.append(fmt(means[key], scale, nd) if key in means else None)
    return out


def marks(rows, ncol, reference):
    """LaTeX bold / underline on the displayed values of the non-reference rows: the highest value
    bold (all rows if tied), the second highest underlined when the highest is unique."""
    out = [["--" if c is None else "$-$" + c[1:] if c.startswith("-") else c for c in r[2]] for r in rows]
    for j in range(ncol):
        vals = {i: float(r[2][j]) for i, r in enumerate(rows) if r[0] not in reference and r[2][j] is not None}
        distinct = sorted(set(vals.values()), reverse=True)
        if not distinct:
            continue
        best = distinct[0]; second = distinct[1] if len(distinct) > 1 else None
        n_best = sum(1 for v in vals.values() if v == best)
        for i, v in vals.items():
            if v == best:
                out[i][j] = r"\textbf{" + out[i][j] + "}"
            elif n_best == 1 and v == second:
                out[i][j] = r"\underline{" + out[i][j] + "}"
    for i, r in enumerate(rows):
        if r[0] in reference:
            out[i] = [r"\tg{" + c + "}" for c in out[i]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", nargs="+", action="append", required=True, metavar="NAME ARM=DIR",
                    help="a backbone block: its name, then arm=directory pairs in row order")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="main")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--reference", nargs="+", default=["teacher"], help="rows shown grey and excluded from the marks")
    ap.add_argument("--csv", default=None, help="raw means and printed cells, one row per arm")
    ap.add_argument("--latex", default=None, help="LaTeX rows with bold / underline marks; reference rows in \\tg{} "
                                                  "(the paper's grey-text macro)")
    a = ap.parse_args()
    cols = COLUMNS[a.protocol]
    load = load_main if a.protocol == "main" else load_ablation
    blocks = []
    for spec in a.block:
        name, sets = spec[0], spec[1:]
        rows = []
        for s in sets:
            arm, d = s.split("=", 1)
            m = load(d, a.seeds); check(f"{name}/{arm}", m, a.protocol)
            rows.append((arm, m, cells(m, cols), [fmt(m[k]["mean"], 100, 1) if k in m else None for _, k in TPIPS_COLUMNS]))
        blocks.append((name, rows))

    head = ["Method"] + [c[0] for c in cols] + ["|"] + [c[0] for c in TPIPS_COLUMNS]
    w = max(14, max(len(LABELS.get(r[0], r[0])) for _, rows in blocks for r in rows) + 2)
    print(f"{head[0]:{w}s}" + "".join(f"{h:>9s}" for h in head[1:]))
    for name, rows in blocks:
        print(f"-- {name}")
        for arm, m, c, tp in rows:
            print(f"{LABELS.get(arm, arm):{w}s}" + "".join(f"{x or '--':>9s}" for x in c + ["|"] + tp))
    if a.csv:
        keys = FEATURE_KEYS + GPU_KEYS
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["block", "arm"] + keys + [c[0] for c in cols])
            for name, rows in blocks:
                for arm, m, c, _ in rows:
                    wr.writerow([name, arm] + [repr(m[k]["mean"]) if k in m else "" for k in keys] + [x or "" for x in c])
        print("[table] wrote", a.csv)
    if a.latex:
        lines = []
        for name, rows in blocks:
            lines.append(f"% {name}")
            for (arm, *_), c in zip(rows, marks(rows, len(cols), set(a.reference))):
                lines.append(f"{LABELS.get(arm, arm)} & " + " & ".join(c) + r" \\")
        with open(a.latex, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("[table] wrote", a.latex)


if __name__ == "__main__":
    main()
