"""Stage B of the image evaluation: metrics from the feature caches (paper App. F).

  dino           1 - mean over pairs i<j of the mean cosine of the 256 aligned DINOv2 patch tokens
  vendi          exp(entropy) of the eigenvalues of the DINOv2 CLS kernel K = C C^T
                 (symmetrised, + 1e-6 I, normalised to sum 1); the DPP logdet(I + K) is stored as `dpp`
  dreamsim       1 - mean pairwise cosine of the DreamSim embeddings (= mean DreamSim distance)
  clipscore      mean cos(CLIP image, CLIP prompt) (raw cosine; Hessel's 2.5 * max(cos, 0) as `hessel`)
  tpips_<factor> 1 - mean pairwise cosine of the TPIPS-Embed embeddings of one factor

Every value is per prompt over all R seeds; the set value is the mean over prompts. The paper
protocol has 553 prompts for every metric except the four TPIPS aspects, which use the 100 prompts
of prompts/geneval_subset100.txt. Each metric must find a cache for every one of its prompts.

  python -m evaluation.image.feature_metrics --feats <set>/feats --out <set>/metrics_cpu.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from evaluation.common.image_sets import SEEDS
from evaluation.common.pairwise import cos_diversity, l2n
from evaluation.common.tpips_factors import FACTORS
from rediverse.data import GENEVAL_553, GENEVAL_SUBSET100, parse_ids, read_prompts

REGISTRY = {}
ASPECT_METRICS = {f"tpips_{s}" for s in FACTORS if s != "overall"}
DEFAULT_METRICS = ["dino", "vendi", "dreamsim", "clipscore"] + [f"tpips_{s}" for s in FACTORS]
R_EXPECTED = SEEDS


def metric(name):
    def deco(fn):
        REGISTRY[name] = fn; return fn
    return deco


def _rows(x, f):
    assert x.shape[0] == R_EXPECTED, f"{f}: {x.shape[0]} samples, expected {R_EXPECTED}"
    return x


# ------------------------------------------------------------------ loaders --

def cache_path(feats, name, tag):
    if name.startswith("tpips_"):
        return os.path.join(feats, "tpips", name[len("tpips_"):], tag + ".npy")
    sub = {"dino": "dino", "vendi": "dino", "dreamsim": "dreamsim", "clipscore": "clip"}[name]
    return os.path.join(feats, sub, tag + (".npy" if sub == "dreamsim" else ".npz"))


def load_dino(feats, tag):
    f = os.path.join(feats, "dino", tag + ".npz")
    return _rows(np.load(f)["tokens"].astype(np.float32), f)                 # [R, 257, 768]


def load_clip(feats, tag):
    f = os.path.join(feats, "clip", tag + ".npz")
    d = np.load(f); return _rows(d["img"], f), d["txt"]


def load_dreamsim(feats, tag):
    f = os.path.join(feats, "dreamsim", tag + ".npy")
    return _rows(np.load(f), f)


# ------------------------------------------------------------------ metrics --

@metric("dino")
def m_dino(feats, tag):
    P = l2n(load_dino(feats, tag)[:, 1:])                     # [R, 256, 768] unit patches
    R = P.shape[0]
    # mean over aligned patches of cos, then mean over pairs  ==  mean over pairs of mean_p cos
    G = np.einsum("ipd,jpd->ij", P, P) / P.shape[1]           # [R, R] mean patch cosine
    return {"mean": 1.0 - float((G.sum() - np.trace(G)) / (R * (R - 1)))}


@metric("vendi")
def m_vendi(feats, tag):
    C = l2n(load_dino(feats, tag)[:, 0]); R = C.shape[0]
    k = C @ C.T; k = (k + k.T) / 2 + 1e-6 * np.eye(R)
    lam = np.clip(np.linalg.eigvalsh(k), 1e-12, None); p = lam / lam.sum()
    vendi = float(np.exp(-(p * np.log(p)).sum()))
    dpp = float(np.linalg.slogdet(np.eye(R) + k)[1])
    return {"mean": vendi, "dpp": dpp}


@metric("dreamsim")
def m_dreamsim(feats, tag):
    return {"mean": cos_diversity(load_dreamsim(feats, tag))}


@metric("clipscore")
def m_clip(feats, tag):
    img, txt = load_clip(feats, tag)
    s = l2n(img) @ l2n(txt)
    return {"mean": float(s.mean()), "hessel": float(2.5 * np.clip(s, 0, None).mean()), "per_image": s.tolist()}


def _tpips_metric(slug):
    def fn(feats, tag):
        f = os.path.join(feats, "tpips", slug, tag + ".npy")
        return {"mean": cos_diversity(_rows(np.load(f).astype(np.float32), f))}
    return fn


for _slug in FACTORS:
    metric(f"tpips_{_slug}")(_tpips_metric(_slug))


# --------------------------------------------------------------------- main --

def main():
    global R_EXPECTED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--prompts", default=GENEVAL_553)
    ap.add_argument("--prompt-ids", default=None, help="prompts of every metric except the TPIPS aspects (default: all)")
    ap.add_argument("--aspect-ids", default=GENEVAL_SUBSET100, help="prompts of the four TPIPS aspects")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--images-dir", default=None, help="recorded in the output only")
    a = ap.parse_args()
    R_EXPECTED = a.seeds
    names = a.metrics.split(",")
    ids = parse_ids(a.prompt_ids) if a.prompt_ids else list(range(len(read_prompts(a.prompts))))
    aspect_ids = parse_ids(a.aspect_ids)
    if ASPECT_METRICS & set(names):
        assert set(aspect_ids) <= set(ids), "--aspect-ids must be a subset of --prompt-ids"
    tags = [f"p{i:03d}" for i in ids]
    out = {"feats_dir": a.feats, "images_dir": a.images_dir, "num_prompts": len(tags), "seeds": a.seeds, "metrics": {}}
    t0 = time.time()
    for name in names:
        fn = REGISTRY[name]
        use = [f"p{i:03d}" for i in aspect_ids] if name in ASPECT_METRICS else tags
        missing = [t for t in use if not os.path.exists(cache_path(a.feats, name, t))]
        if missing:
            raise SystemExit(f"[metrics] {name}: no cache for {len(missing)}/{len(use)} prompts "
                             f"(first: {', '.join(missing[:5])}) under {a.feats}")
        per = {}; extra = {}
        for tag in use:
            r = fn(a.feats, tag)
            per[tag] = r["mean"]
            for k, v in r.items():
                if k not in ("mean", "per_image"):
                    extra.setdefault(k, {})[tag] = v
            if "per_image" in r:
                extra.setdefault("per_image", {})[tag] = r["per_image"]
        vals = np.array(list(per.values()))
        entry = {"mean": float(vals.mean()), "std_over_prompts": float(vals.std()), "n_prompts": len(vals), "per_prompt": per}
        for k, d in extra.items():
            if k == "per_image":
                entry["per_image"] = d
            else:
                entry[k + "_mean"] = float(np.mean(list(d.values()))); entry[k + "_per_prompt"] = d
        out["metrics"][name] = entry
        print(f"  {name:16s} mean {entry['mean']:.4f} ± {entry['std_over_prompts']:.4f} over {len(vals)} prompts"
              f"  [{time.time()-t0:.0f}s]", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print("[metrics] wrote", a.out)


if __name__ == "__main__":
    main()
