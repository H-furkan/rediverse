#!/usr/bin/env python
"""ReDiverse-TF directions on MiniMax-H3 (paper Sec. 4.2): per prompt, the top-8 teacher principal
directions of the block-45 video rows at teacher forward tau = 16.

For prompt i the teacher is rolled from N = 25 latents randn(seed 1_000_000 + (10000 + i) * 100003
+ j), j = 100..124 (disjoint from the evaluation clips 0..9), and the block-45 video rows of its
16th forward are flattened to D = 37296 * 5376 = 200,503,296. rediverse.directions.
directions_from_samples gives U (bfloat16 [8, D]), s ([8]) and the eigenvalue spectrum. About
1 GPU-hour and 30 GB of host memory per prompt; a direction file is 3.2 GB.

Benchmark mode writes <out-dir>/v{i:03d}.pt for the VBench prompts; several workers can share the
output directory (prompts are claimed atomically), and an existing file counts as done only when it
holds that prompt:

  python -m h3.estimate_directions --device cuda:0        # run one per GPU

Single-prompt mode encodes the prompt, estimates its directions with prompt index --prompt-index
in the seed rule and writes <outputs>/prompt/<slug>/directions.pt, where h3.generate --prompt
looks for them:

  python -m h3.estimate_directions --prompt "a corgi running on the beach" --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from h3 import config as C
from h3 import model as H
from rediverse import data, seeds
from rediverse.claims import claim
from rediverse.directions import N_DIR, directions_from_samples, direction_file, matches, save_directions


def estimate(teacher, ctx, prompt_idx: int, device, k: int = C.TAU, n_lat: int = C.N_DIRECTION_SAMPLES):
    """(U, s, eigvals, sample indices) for one prompt context."""
    idx = seeds.direction_sample_indices(n_lat)
    X = torch.empty(n_lat, H.NUM_VIDEO_ROWS * C.HIDDEN, dtype=torch.bfloat16)   # CPU
    for j, s in enumerate(idx):
        x0 = H.direction_latent(prompt_idx, s).unsqueeze(0)
        acts, _ = H.teacher_states(teacher, x0, ctx, device, [k])
        X[j] = acts[k][0].reshape(-1).cpu()
        del acts
    U, S, evals = directions_from_samples(X, N_DIR)
    return U, S, evals, idx


def direction_meta(prompt_idx: int, prompt: str, k: int, idx) -> dict:
    return dict(prompt_idx=prompt_idx, prompt=prompt, k=k, block=H.BLOCK, n=len(idx), seeds=list(idx),
                shape=(H.NUM_VIDEO_ROWS, C.HIDDEN))


def prompt_direction_file(prompt: str) -> str:
    """<outputs>/prompt/<slug>/directions.pt: the direction file of a --prompt run."""
    return os.path.join(data.prompt_dir(C.BACKBONE, prompt), "directions.pt")


def estimate_prompt(ctx, prompt: str, prompt_idx: int, device, path: str,
                    k: int = C.TAU, n_lat: int = C.N_DIRECTION_SAMPLES) -> dict:
    """Load the teacher, estimate the directions of one prompt context, write them to `path` and
    release the teacher before returning the saved record."""
    teacher = H.load_arm("teacher", device)
    U, S, evals, idx = estimate(teacher, ctx, prompt_idx, device, k, n_lat)
    del teacher
    H.free()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return save_directions(path, U, S, evals, **direction_meta(prompt_idx, prompt, k, idx))


def run_prompt(a):
    from h3.encode_prompts import prompt_context
    path = a.out or prompt_direction_file(a.prompt)
    ctx = prompt_context(a.prompt, a.prompt_index, a.device)
    t0 = time.time()
    rec = estimate_prompt(ctx, a.prompt, a.prompt_index, a.device, path, a.k, a.n_lat)
    print(f"[h3/directions] {path}: s1={float(rec['s'][0]):.2f} pc1_share={rec['pc1_share']:.3f} "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def run_benchmark(a):
    texts = data.read_prompts(data.VBENCH_300)
    out = a.out_dir or C.direction_dir(a.k)
    ctx_dir = a.ctx_dir or C.vbench_ctx_dir()
    os.makedirs(out, exist_ok=True)
    dev = a.device
    si, sn = a.shard, a.num_shards
    sel = data.parse_ids(a.prompt_ids) if a.prompt_ids else list(range(len(texts)))
    idxs = sel[si::sn] if a.prompt_ids else [i for i in sel if i % sn == si]
    pending = [i for i in idxs if not os.path.exists(direction_file(out, i, "v"))]
    other = [i for i in sorted(set(idxs) - set(pending)) if not matches(direction_file(out, i, "v"), texts[i], i)]
    if other:
        sys.exit(f"{len(other)} files in {out} hold the directions of other prompts, e.g. v{other[0]:03d}.pt; "
                 "pass --out-dir to write a separate bank for a different prompt file")
    print(f"[h3/directions] k={a.k} D={H.NUM_VIDEO_ROWS*C.HIDDEN/1e6:.0f}M "
          f"shard {si}/{sn}: {len(pending)}/{len(idxs)} prompts to do", flush=True)
    if not pending:
        return
    teacher = H.load_arm("teacher", dev)
    t0, done = time.time(), 0
    for i in pending:
        f_out = direction_file(out, i, "v")
        if os.path.exists(f_out) or not claim(f_out):
            continue
        ctx = H.load_ctx(os.path.join(ctx_dir, f"v{i:03d}.pt"))
        U, S, evals, idx = estimate(teacher, ctx, i, dev, a.k, a.n_lat)
        rec = save_directions(f_out, U, S, evals, **direction_meta(i, texts[i], a.k, idx))
        done += 1; el = time.time() - t0
        print(f"  [{done}/{len(pending)}] v{i:03d} s1={float(S[0]):.2f} s8={float(S[-1]):.2f} "
              f"pc1_share={rec['pc1_share']:.3f} {el/done/60:.1f} min/prompt "
              f"eta {(len(pending)-done)*el/done/3600:.1f}h {texts[i][:40]!r}", flush=True)
        del U, S, rec
    print(f"[h3/directions] shard {si}/{sn} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=C.TAU, help="teacher forward of the activations")
    ap.add_argument("--n-lat", type=int, default=C.N_DIRECTION_SAMPLES)
    ap.add_argument("--device", default="cuda:0")
    # benchmark mode
    ap.add_argument("--prompt-ids", default="", help="subset of VBench indices, e.g. '0,5' or '0-29' (default: all)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    ap.add_argument("--num-shards", type=int, default=1, help="number of shards")
    ap.add_argument("--ctx-dir", default="", help="VBench context directory (default <data>/ctx/vbench)")
    ap.add_argument("--out-dir", default="", help="direction bank (default <data>/directions/b45_k<k>)")
    # single-prompt mode
    ap.add_argument("--prompt", default="", help="estimate the directions of this prompt (single-prompt mode)")
    ap.add_argument("--prompt-index", type=int, default=0,
                    help="--prompt: prompt index of the seed rules (a VBench index reproduces that prompt's bank file)")
    ap.add_argument("--out", default="", help="--prompt: output file (default <outputs>/prompt/<slug>/directions.pt)")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error("--shard must be in [0, --num-shards)")
    if a.prompt:
        run_prompt(a)
    else:
        run_benchmark(a)


if __name__ == "__main__":
    main()
