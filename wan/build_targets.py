#!/usr/bin/env python
"""Build the ReDiverse-FT target bank for Wan 2.1 (paper App. D, "Targets and loss").

For each training caption ci (0..99, context file p{10000+ci}.pt) and latent ni (0..15): the
initial latent is the seed rule's (rediverse.seeds.train_noise_seed(ci, ni)), the CFG-distilled
teacher (wan.cfgd_teacher) is rolled out conditional-only on its UniPC-50 schedule at flow shift
5 for 12 full forwards, and the blocks.28 output of forward 13 is stored with the latent
(wan.target_bank). One entry takes ~50 s on an H100 (13 forwards); the bank is 1,600 entries.

Entries are written atomically and existing ones are skipped, so a stopped builder resumes by
rerunning the same command; --shard / --num-shards split the (caption, latent) list across
processes writing into one directory.

  python -m wan.build_targets --device cuda:0 --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from rediverse import data, seeds
from wan import config as C
from wan import model as M
from wan import target_bank as bank


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--captions", default=f"0-{data.N_CAPTIONS - 1}",
                    help="training caption indices ci, e.g. '0-99' or '0,5,7'")
    ap.add_argument("--n-latents", type=int, default=C.NOISES_PER_CAPTION, help="latents per caption")
    ap.add_argument("--shard", type=int, default=0, help="this worker takes the entries n with n %% num-shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--bank-dir", default=None, help=f"default {C.target_dir()}")
    ap.add_argument("--ctx-dir", default=None, help=f"default {C.train_ctx_dir()}")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in 0..{a.num_shards - 1}")
    bdir = a.bank_dir or C.target_dir()
    ctx_dir = a.ctx_dir or C.train_ctx_dir()
    os.makedirs(bdir, exist_ok=True)

    cis = data.parse_ids(a.captions)
    work = [(data.ctx_index(ci), ci, ni) for ci in cis for ni in range(a.n_latents)]
    mine = [w for n, w in enumerate(work) if n % a.num_shards == a.shard]
    pending = [w for w in mine if not os.path.exists(bank.entry_path(bdir, w[0], w[2]))]
    missing = [p for p in sorted({w[0] for w in pending})
               if not os.path.exists(os.path.join(ctx_dir, f"p{p}.pt"))]
    assert not missing, f"no context files for {missing} in {ctx_dir}: run python -m wan.encode_prompts --set train"
    print(f"bank   : {bdir}\n"
          f"target : blocks[{C.ANCHOR_BLOCK}] at forward {C.FT_TARGET_FORWARD}/{C.TEACHER_STEPS} of the "
          f"CFG-distilled teacher, flow shift {C.CFGD_FLOW_SHIFT}, conditional only\n"
          f"work   : {len(cis)} captions x {a.n_latents} latents = {len(work)} entries; shard "
          f"{a.shard}/{a.num_shards} owns {len(mine)}, {len(pending)} to build", flush=True)
    if not pending:
        return

    teacher = M.load_transformer("teacher_cfgd", a.device)
    if a.shard == 0:
        bank.write_manifest(bdir, extra=dict(n_latents_per_prompt=a.n_latents,
                                             prompts=sorted({w[0] for w in work})))

    t0 = time.time()
    for done, (p, ci, ni) in enumerate(pending, 1):
        emb = M.load_ctx(os.path.join(ctx_dir, f"p{p}.pt"))["encoder_hidden_states"].to(a.device, M.DTYPE)
        lat = seeds.wan_latent(seeds.train_noise_seed(ci, ni), C.FRAMES)     # [1, 16, 21, 60, 104] fp32
        h = M.teacher_block_output(teacher, lat.to(a.device, torch.float32), emb, a.device,
                                   k=C.FT_TARGET_FORWARD, block=C.ANCHOR_BLOCK,
                                   flow_shift=C.CFGD_FLOW_SHIFT, truncate_last=True)
        assert h.shape == (1, C.TOKENS, C.DIM), tuple(h.shape)
        bank.save_entry(bank.entry_path(bdir, p, ni), h[0].cpu(), lat[0])
        del h
        rate = (time.time() - t0) / done
        print(f"[build] p{p} i{ni:03d}  {done}/{len(pending)}  {rate:.1f} s/entry  "
              f"eta {(len(pending) - done) * rate / 60:.0f} min", flush=True)
    print(f"[build] done: {len(pending)} entries in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
