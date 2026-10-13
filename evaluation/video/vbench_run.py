"""VBench subject and background consistency of a video set (main comparison table: SC, BC).
Runs in the vbench environment (vbench==0.1.5).

Each dimension is scored with VBench in custom_input mode on the clips of the prompts that VBench itself
assigns to that dimension: the prompts of prompts/vbench_300.txt whose dimension list in the package's
VBench_full_info.json contains it (subject_consistency: 23 prompts / 230 clips, background_consistency:
27 prompts / 270 clips). The reported value is VBench's mean over those clips.

  subject_consistency     DINO ViT-B/16 (torch.hub facebookresearch/dino), frame-to-first and frame-to-previous
  background_consistency  CLIP ViT-B/32 (OpenAI clip package)

Output: <set>/vbench/<dimension>.json {dimension, set, mean, n_videos, n_prompts, prompt_list, per_video, seconds}.
A dimension whose json exists is skipped. VBench reads the prompt from the clip name and caches its weights
in $VBENCH_CACHE_DIR (default ~/.cache/vbench); torch.hub and clip use their own caches under ~/.cache.

  python -m evaluation.video.vbench_run --set <set>
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

from evaluation.common.atomic import save
from evaluation.video.sets import groups, prompts

DIMS = {"subject_consistency": 23, "background_consistency": 27}      # dimension -> prompts of the 300 it covers


def vbench_dim_prompt_ids(dim):
    """Indices of the prompts of prompts/vbench_300.txt whose VBench_full_info.json dimension list contains `dim`."""
    import vbench
    with open(os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json")) as f:
        info = json.load(f)
    by = {d["prompt_en"]: set(d["dimension"]) for d in info}
    ids = [k for k, p in enumerate(prompts()) if dim in by.get(p, set())]
    assert len(ids) == DIMS[dim], f"{dim}: VBench_full_info.json assigns {len(ids)} of our prompts, expected {DIMS[dim]}"
    return ids


def run_dim(set_dir, dim, out, limit=None, tmp_dir=None):
    import torch
    from vbench import VBench
    gs = groups(os.path.join(set_dir, "videos"), vbench_dim_prompt_ids(dim))
    if limit:
        gs = gs[:limit]
    paths = [os.path.abspath(p) for _, _, ps in gs for p in ps]
    print(f"[vbench {dim}] {len(gs)} prompts / {len(paths)} clips from {set_dir}", flush=True)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as td:
        vdir = os.path.join(td, "videos"); os.makedirs(vdir)
        for p in paths:
            os.symlink(p, os.path.join(vdir, os.path.basename(p)))
        vb = VBench(torch.device("cuda"), os.path.join(td, "full_info.json"), td)
        t0 = time.time()
        vb.evaluate(videos_path=vdir, name=f"run_{dim}", dimension_list=[dim], mode="custom_input")
        with open(os.path.join(td, f"run_{dim}_eval_results.json")) as f:
            r = json.load(f)[dim]
        mean, per = float(r[0]), r[1]
        per_video = [{"video": os.path.basename(x["video_path"]), "score": x["video_results"]} for x in per]
    res = dict(dimension=dim, set=set_dir, mean=mean, n_videos=len(per_video), n_prompts=len(gs),
               prompt_list="VBench_full_info", per_video=per_video, seconds=round(time.time() - t0, 1))
    save(out, lambda f: json.dump(res, f, indent=1), mode="w")
    print(f"[vbench {dim}] mean {mean:.4f} over {len(per_video)} clips in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="set directory containing videos/")
    ap.add_argument("--dim", nargs="+", default=list(DIMS), choices=list(DIMS))
    ap.add_argument("--out-dir", default=None, help="default <set>/vbench")
    ap.add_argument("--tmp-dir", default=None, help="parent of the temporary symlink directory (default: $TMPDIR)")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts of the dimension's list only")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(a.set, "vbench"); os.makedirs(out_dir, exist_ok=True)
    for dim in a.dim:
        out = os.path.join(out_dir, f"{dim}.json")
        if os.path.exists(out):
            print("[vbench] exists", out); continue
        run_dim(a.set, dim, out, a.limit, a.tmp_dir)


if __name__ == "__main__":
    main()
