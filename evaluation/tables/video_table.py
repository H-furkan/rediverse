"""Video rows of the paper's tables from the per-set metric files.

  --table main     main comparison table (videos): DINO, VJ, TP, Cam, VC (diversity) and SC, BC, VCS (quality)
  --table aspects  per-aspect TPIPS table (supplementary, video rows): TPIPS overall / subject / background /
                   lighting / camera, CamEnd, ViCLIP Vendi

Every cell is the set mean x100 printed with one decimal (Vendi: unscaled, two decimals). TP is the mean of
the five per-aspect TPIPS cells as printed (each rounded to one decimal), rounded to one decimal; the overall
aspect averages all 300 prompts and the other four the 100-prompt subset. Best and second-best values among
the rows other than the teacher are marked (bold / underline): all rows tied at the best are bold, and the
second-best value is marked only when the best is unique.

Sources per arm: <set>/metrics_video.json (evaluation/video/video_metrics.py) and
<set>/vbench/{subject,background}_consistency.json (evaluation/video/vbench_run.py). The set of an arm is
$REDIVERSE_OUT/<backbone>/<arm> (default outputs/wan2.1/<arm>, outputs/minimax-h3/<arm>), or as given by --set.

  python -m evaluation.tables.video_table --backbone wan
  python -m evaluation.tables.video_table --backbone h3 --table aspects --format tex
  python -m evaluation.tables.video_table --backbone wan --set teacher=path/to/teacher --extra Other=path/to/set
"""
from __future__ import annotations

import argparse
import json
import os

from rediverse.data import output_root
from rediverse.hub import hub_backbone

ARMS = [("teacher", "Teacher"), ("student", "Student"), ("rediverse_tf", "ReDiverse-TF"),
        ("rediverse_ft", "ReDiverse-FT"), ("rediverse", "ReDiverse")]
TP_ASPECTS = ("overall", "subject", "background", "lighting", "camera")

# metric key -> number of prompts it must cover
N_EXPECTED = {"dino_frame_dinov2": 300, "vjepa2": 300, "viclip_div": 300, "viclip_vendi": 300, "viclip_text": 300,
              "camtraj_traj_norm": 30, "camtraj_endpoint_norm": 30,
              **{f"tpips_video_{a}": (300 if a == "overall" else 100) for a in TP_ASPECTS}}
VBENCH_EXPECTED = {"subject_consistency": (23, 230), "background_consistency": (27, 270)}


def r1(x):
    """x rounded to one decimal as printed."""
    return float(f"{x:.1f}")


def pct(key):
    return lambda m: f"{m[key] * 100:.1f}"


def tp(m):
    return f"{r1(sum(r1(m[f'tpips_video_{a}'] * 100) for a in TP_ASPECTS) / len(TP_ASPECTS)):.1f}"


TABLES = {
    "main": [("DINO", pct("dino_frame_dinov2")), ("VJ", pct("vjepa2")), ("TP", tp), ("Cam", pct("camtraj_traj_norm")),
             ("VC", pct("viclip_div")), ("SC", pct("vb:subject_consistency")), ("BC", pct("vb:background_consistency")),
             ("VCS", pct("viclip_text"))],
    "aspects": [("Overall", pct("tpips_video_overall")), ("Subject", pct("tpips_video_subject")),
                ("Backgr.", pct("tpips_video_background")), ("Lighting", pct("tpips_video_lighting")),
                ("Camera", pct("tpips_video_camera")), ("CamEnd", pct("camtraj_endpoint_norm")),
                ("ViCLIP Vendi", lambda m: f"{m['viclip_vendi']:.2f}")],
}


def load_set(set_dir):
    """{metric key: set mean} from metrics_video.json and the two VBench files, with coverage checks."""
    with open(os.path.join(set_dir, "metrics_video.json")) as f:
        metrics = json.load(f)["metrics"]
    m = {}
    for key, n in N_EXPECTED.items():
        if key not in metrics:
            raise KeyError(f"{set_dir}/metrics_video.json has no {key}")
        assert metrics[key]["n_prompts"] == n, f"{set_dir}: {key} covers {metrics[key]['n_prompts']} prompts, expected {n}"
        m[key] = metrics[key]["mean"]
    for dim, (n_prompts, n_videos) in VBENCH_EXPECTED.items():
        with open(os.path.join(set_dir, "vbench", f"{dim}.json")) as f:
            d = json.load(f)
        assert (d["n_prompts"], d["n_videos"]) == (n_prompts, n_videos), \
            f"{set_dir}: {dim} covers {d['n_prompts']} prompts / {d['n_videos']} clips, expected {n_prompts} / {n_videos}"
        m["vb:" + dim] = d["mean"]
    return m


def rank_marks(rows):
    """rows [(label, cells, ranked)] -> marks[label][j] in {'best', 'second', ''} on the printed values."""
    marks = {label: [""] * len(cells) for label, cells, _ in rows}
    for j in range(len(rows[0][1])):
        vals = {label: float(cells[j]) for label, cells, ranked in rows if ranked}
        distinct = sorted(set(vals.values()), reverse=True)
        best = distinct[0]
        second = distinct[1] if len(distinct) > 1 else None
        n_best = sum(1 for v in vals.values() if v == best)
        for label, v in vals.items():
            if v == best:
                marks[label][j] = "best"
            elif n_best == 1 and v == second:
                marks[label][j] = "second"
    return marks


def build(sets, table):
    """sets [(label, set_dir, ranked)] -> (header, [(label, printed cells, marks)])."""
    cols = TABLES[table]
    rows = []
    for label, d, ranked in sets:
        m = load_set(d)
        rows.append((label, [fn(m) for _, fn in cols], ranked))
    marks = rank_marks(rows)
    return [h for h, _ in cols], [(label, cells, marks[label]) for label, cells, _ in rows]


def render(header, rows, fmt):
    wrap = {"md": {"best": "**{}**", "second": "<ins>{}</ins>", "": "{}"},
            "tex": {"best": r"\textbf{{{}}}", "second": r"\underline{{{}}}", "": "{}"}}[fmt]
    if fmt == "md":
        lines = ["| Method | " + " | ".join(header) + " |", "|---|" + "|".join("---:" for _ in header) + "|"]
        lines += ["| " + label + " | " + " | ".join(wrap[k].format(c) for c, k in zip(cells, mk)) + " |"
                  for label, cells, mk in rows]
    else:
        lines = ["Method & " + " & ".join(header) + r" \\"]
        lines += [label + " & " + " & ".join(wrap[k].format(c) for c, k in zip(cells, mk)) + r" \\"
                  for label, cells, mk in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, help="wan or h3 (or wan2.1 / minimax-h3)")
    ap.add_argument("--root", default=None, help="directory holding one set per arm (default $REDIVERSE_OUT/<backbone>)")
    ap.add_argument("--set", action="append", default=[], metavar="ARM=DIR", help="set directory of one arm")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=DIR",
                    help="additional ranked row, placed after Student")
    ap.add_argument("--table", default="main", choices=list(TABLES))
    ap.add_argument("--format", default="md", choices=["md", "tex"])
    ap.add_argument("--out", default=None, help="also write the table to this file")
    a = ap.parse_args()
    root = a.root or os.path.join(output_root(), hub_backbone(a.backbone))
    dirs = {arm: os.path.join(root, arm) for arm, _ in ARMS}
    for spec in a.set:
        arm, d = spec.split("=", 1)
        assert arm in dirs, f"--set {spec}: arm must be one of {list(dirs)}"
        dirs[arm] = d
    sets = [(label, dirs[arm], arm != "teacher") for arm, label in ARMS]
    extra = [(spec.split("=", 1)[0], spec.split("=", 1)[1], True) for spec in a.extra]
    sets = sets[:2] + extra + sets[2:]
    text = render(*build(sets, a.table), a.format)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
