#!/usr/bin/env python
"""ReDiverse-FT target bank: teacher transformer_blocks[45] video rows at teacher forward tau = 16.

For each training prompt p (prompts/h3_train_geneval_ids.txt) and latent i in 0..19, the teacher is
rolled 15 steps from latent(p, i) = randn(seed 1_000_000 + p * 100003 + i) and the block-45
video rows of its 16th forward are stored. One entry <bank-dir>/p{idx:03d}_s{i}.pt =
{prompt_idx, seed, k, latent float32 [24,37,48,84], target bfloat16 [37296, 5376]}, about 415 MB;
the full bank is 400 entries. Several workers (one per GPU) can fill the same directory: entries
are claimed atomically.

  python -m h3.build_targets --device cuda:0        # run one per GPU
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from h3 import config as C
from h3 import model as H
from rediverse import data
from rediverse.claims import claim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=C.TAU, help="teacher forward (1-indexed) of the target")
    ap.add_argument("--n-latents", type=int, default=C.FT_LATENTS_PER_PROMPT)
    ap.add_argument("--prompt-ids", default="", help="subset of the training prompt indices (default: all 20)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    ap.add_argument("--num-shards", type=int, default=1, help="number of shards; shard i takes every N-th entry")
    ap.add_argument("--ctx-dir", default="", help="training context directory (default <data>/ctx/train)")
    ap.add_argument("--bank-dir", default="", help="bank directory (default <data>/targets/b45_k<k>)")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error("--shard must be in [0, --num-shards)")
    si, sn = a.shard, a.num_shards
    out = a.bank_dir or C.target_dir(a.k)
    ctx_dir = a.ctx_dir or C.train_ctx_dir()
    os.makedirs(out, exist_ok=True)
    dev = a.device

    prompts = data.parse_ids(a.prompt_ids) if a.prompt_ids else C.ft_prompt_ids()
    missing = [p for p in prompts if not os.path.exists(os.path.join(ctx_dir, f"p{p:03d}.pt"))]
    assert not missing, (f"no context for {len(missing)} prompt(s) in {ctx_dir}, e.g. {missing[:3]}; "
                         "run python -m h3.encode_prompts --set train first")
    teacher = H.load_arm("teacher", dev)
    sched = H.Schedule(C.STEPS["teacher"], H.load_ctx(os.path.join(ctx_dir, f"p{prompts[0]:03d}.pt")), dev)
    assert 1 <= a.k <= sched.n_fwd
    meta = dict(block=H.BLOCK, k=a.k, teacher_steps=C.STEPS["teacher"], n_fwd=sched.n_fwd,
                sigma_in=sched.sigmas[a.k - 1], sigma_out=sched.sigmas[a.k],
                prompts=list(prompts), n_latents=a.n_latents, frames=H.NUM_FRAMES,
                seed_rule="SEED0 + prompt*100003 + i", stream="video rows (tail slice)")
    if si == 0:
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump(meta, f, indent=1)

    work = [(p, i) for p in prompts for i in range(a.n_latents)]
    work = [w for n, w in enumerate(work) if n % sn == si]
    t0, done = time.time(), 0
    for p, i in work:
        f = os.path.join(out, f"p{p:03d}_s{i}.pt")
        if os.path.exists(f) or not claim(f):
            continue
        ctx = H.load_ctx(os.path.join(ctx_dir, f"p{p:03d}.pt"))
        x0 = H.train_latent(p, i).unsqueeze(0)
        acts, _ = H.teacher_states(teacher, x0, ctx, dev, [a.k])
        torch.save(dict(prompt_idx=p, seed=i, k=a.k, latent=x0[0].contiguous(),
                        target=acts[a.k][0].contiguous().cpu()), f + ".tmp")
        os.replace(f + ".tmp", f)
        done += 1
        el = time.time() - t0
        print(f"  [{done}/{len(work)}] p{p:03d} s{i}  {el/done:.0f}s/entry  "
              f"eta {(len(work)-done)*el/done/60:.0f}m", flush=True)
    print(f"[h3/bank] shard {si}/{sn} done -> {out}", flush=True)


if __name__ == "__main__":
    main()
