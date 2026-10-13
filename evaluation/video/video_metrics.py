"""Per-set video metrics from the feature caches (CPU, numpy) -> <set>/metrics_video.json.

  dino_frame_dinov2      mean over the 16 frame indices t of pair_div(CLS_t of the 10 clips)      DINO
  vjepa2                 pair_div of the 10 V-JEPA 2 clip embeddings                              VJ
  tpips_video_<aspect>   mean over the 8 frame indices t of pair_div(TPIPS_t of the 10 clips)     TP (5 aspects)
  viclip_div             pair_div of the 10 ViCLIP clip embeddings                                VC
  viclip_vendi           Vendi score of the 10 ViCLIP clip embeddings
  viclip_text            mean cosine between the 10 ViCLIP clip embeddings and the prompt's      VCS
  camtraj_traj_norm      mean over clip pairs of the mean pointwise distance between
                         length-normalised camera trajectories                                    Cam
  camtraj_endpoint_norm  mean over clip pairs of the endpoint distance of those trajectories      CamEnd
with pair_div(X) = 1 - mean off-diagonal cosine over the rows of X. Every metric is a per-prompt value averaged
over prompts: {mean, std_over_prompts, n_prompts, per_prompt}. A cache directory that does not exist is
reported as not computed; a cache directory that exists must hold exactly the expected prompts (all 300;
TPIPS aspects prompts/vbench_subset100.txt; camera trajectories prompts/vbench_subset30.txt) with 10 clips each.

  python -m evaluation.video.video_metrics --feats <set>/feats --out <set>/metrics_video.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

import numpy as np

from evaluation.video.sets import N_PER_PROMPT, prompt_ids

ASPECTS = ("background", "camera", "lighting", "overall", "subject")
EXPECTED = {"vdino/dinov2": "all", "vjepa2": "all", "viclip": "all", "camtraj": "subset30",
            **{f"tpips_video/{a}": ("all" if a == "overall" else "subset100") for a in ASPECTS}}


def pair_div(X):
    """1 - mean pairwise cosine over rows of X [N, D] (rows L2-normalised)."""
    X = X.astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    S = X @ X.T; n = X.shape[0]
    return float(1.0 - (S.sum() - np.trace(S)) / (n * (n - 1)))


def vendi(X):
    """exp(entropy) of the eigenvalues of K = X X^T / N on L2-normalised rows."""
    X = X.astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    K = X @ X.T / X.shape[0]
    w = np.clip(np.linalg.eigvalsh(K), 1e-12, None); w = w / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def summarize(per):
    keys = sorted(per); vals = [per[k] for k in keys]
    return dict(mean=float(np.mean(vals)), std_over_prompts=float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                n_prompts=len(vals), per_prompt={str(k): float(v) for k, v in zip(keys, vals)})


def prompt_idx(path):
    """p<idx>.npz / p<idx>.npy -> idx."""
    return int(os.path.basename(path)[1:4])


def cache_files(feats, sub, pattern):
    """Sorted per-prompt cache files of feats/<sub>, checked against the expected prompt set; None when the
    cache directory does not exist."""
    d = os.path.join(feats, sub)
    if not os.path.isdir(d):
        print(f"[video-metrics] {sub}: not computed")
        return None
    fs = sorted(glob.glob(os.path.join(d, pattern)))
    have, want = [prompt_idx(f) for f in fs], prompt_ids(EXPECTED[sub])
    if have != want:
        miss, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
        raise ValueError(f"{d}: {len(have)} prompt caches, expected {len(want)} "
                         f"(missing {miss[:10]}{'...' if len(miss) > 10 else ''}, unexpected {extra[:10]})")
    return fs


def clips(A, f):
    assert A.shape[0] == N_PER_PROMPT, f"{f}: {A.shape[0]} clips, expected {N_PER_PROMPT}"
    return A


def dino_metrics(M, feats):
    fs = cache_files(feats, "vdino/dinov2", "p*.npz")
    if fs:
        pf = {}
        for f in fs:
            cls = clips(np.load(f)["cls"].astype(np.float32), f)                          # [10, 16, 768]
            pf[prompt_idx(f)] = float(np.mean([pair_div(cls[:, t]) for t in range(cls.shape[1])]))
        M["dino_frame_dinov2"] = summarize(pf)


def vjepa2_metrics(M, feats):
    fs = cache_files(feats, "vjepa2", "p*.npy")
    if fs:
        M["vjepa2"] = summarize({prompt_idx(f): pair_div(clips(np.load(f), f)) for f in fs})


def tpips_metrics(M, feats):
    for sl in ASPECTS:
        fs = cache_files(feats, f"tpips_video/{sl}", "p*.npy")
        if fs:
            pt = {}
            for f in fs:
                E = clips(np.load(f).astype(np.float32), f)                                 # [10, 8, 4096]
                pt[prompt_idx(f)] = float(np.mean([pair_div(E[:, t]) for t in range(E.shape[1])]))
            M[f"tpips_video_{sl}"] = summarize(pt)


def viclip_metrics(M, feats):
    fs = cache_files(feats, "viclip", "p*.npz")
    if fs:
        pd, pvn, ptx = {}, {}, {}
        for f in fs:
            k = prompt_idx(f); z = np.load(f); V = clips(z["video"].astype(np.float32), f)      # [10, 768]; text [768]
            pd[k] = pair_div(V); pvn[k] = vendi(V)
            t = z["text"].astype(np.float32); t /= np.linalg.norm(t) + 1e-8
            Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8); ptx[k] = float((Vn @ t).mean())
        M["viclip_div"] = summarize(pd); M["viclip_vendi"] = summarize(pvn); M["viclip_text"] = summarize(ptx)


def camtraj_metrics(M, feats):
    fs = cache_files(feats, "camtraj", "p*.npz")
    if fs:
        pen, ptn = {}, {}
        for f in fs:
            k = prompt_idx(f); z = np.load(f)
            clips(z["path_length"], f)
            pen[k] = float(z["endpoint_norm"]); ptn[k] = float(z["trajectory_norm"])
        M["camtraj_endpoint_norm"] = summarize(pen); M["camtraj_traj_norm"] = summarize(ptn)


def compute_metrics(feats):
    """Every metric available in the caches under `feats`."""
    M = {}
    dino_metrics(M, feats)
    vjepa2_metrics(M, feats)
    tpips_metrics(M, feats)
    viclip_metrics(M, feats)
    camtraj_metrics(M, feats)
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help="feature directory of the set (<set>/feats)")
    ap.add_argument("--out", required=True, help="<set>/metrics_video.json")
    a = ap.parse_args()
    M = compute_metrics(a.feats)
    with open(a.out, "w") as f:
        json.dump(dict(feats=a.feats, metrics=M), f, indent=1)
    for k, v in M.items():
        print(f"{k:28s} {v['mean']:.4f} ± {v['std_over_prompts']:.4f}  (n={v['n_prompts']})")
    print("[video-metrics] wrote", a.out)


if __name__ == "__main__":
    main()
