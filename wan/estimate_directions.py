#!/usr/bin/env python
"""ReDiverse-TF directions for Wan 2.1 (paper Sec. 4.2, App. D; run in the `video` environment).

For one prompt the base teacher (no CFG-distillation LoRA) is rolled out conditional-only on its
repository UniPC-50 schedule (flow shift 3) from N=25 latents (seed rule
rediverse.seeds.wan_direction_seed(i, j), j = 100..124, disjoint from the evaluation clips), and
the blocks.28 output of forward 8 (32,760 x 5120 = 167.7M values) is kept per latent in bf16.
rediverse.directions.directions_from_samples turns the 25 x 167.7M matrix into the top K=8
unit directions u_k (bf16) and spreads s_k; one file per prompt holds them (2.7 GB).
A prompt takes ~15 min on an H100 (200 teacher forwards plus the CPU Gram eigendecomposition).

Benchmark mode (VBench-300; several workers may share the output directory):
  python -m wan.estimate_directions --device cuda:0 [--shard 0 --num-shards 4]
Single prompt (written to <REDIVERSE_OUT>/wan2.1/prompt/<slug>/directions.pt, where
`python -m wan.generate --prompt ...` with the same --prompt-index looks for it):
  python -m wan.estimate_directions --prompt "a corgi surfing a wave" [--prompt-index 0]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch

from rediverse import claims, data, directions, seeds
from wan import config as C
from wan import encode_prompts
from wan import model as M


def estimate(teacher, emb, prompt_idx: int, device, n_lat: int = C.N_DIRECTION_SAMPLES):
    """(U [8, D] bf16, s [8], eigvals [N], seeds, activation shape) of one prompt."""
    js = seeds.direction_sample_indices(n_lat)
    X, shape = None, None
    for n, j in enumerate(js):
        lat = seeds.wan_latent(seeds.wan_direction_seed(prompt_idx, j), C.FRAMES).to(device, torch.float32)
        h = M.teacher_block_output(teacher, lat, emb, device, k=C.TF_DIRECTION_FORWARD,
                                   block=C.ANCHOR_BLOCK)[0]                # [tokens, 5120]
        shape = tuple(h.shape)
        h = h.reshape(-1).to(torch.bfloat16).cpu()
        if X is None:
            X = torch.empty(n_lat, h.numel(), dtype=torch.bfloat16)
        X[n] = h
    U, S, evals = directions.directions_from_samples(X, C.N_DIR)
    return U, S, evals, js, shape


def save(path: str, prompt_idx: int, prompt: str, U, S, evals, js, shape):
    return directions.save_directions(path, U, S, evals, prompt_idx=prompt_idx, prompt=prompt,
                                      k=C.TF_DIRECTION_FORWARD, block=C.ANCHOR_BLOCK, frames=C.FRAMES,
                                      n=len(js), seeds=js, shape=shape)


def prompt_file(prompt: str) -> str:
    """Direction file of a free-text prompt: <REDIVERSE_OUT>/wan2.1/prompt/<slug>/directions.pt."""
    return os.path.join(data.prompt_dir(C.BACKBONE, prompt), "directions.pt")


def estimate_prompt(prompt: str, emb, prompt_index: int, device, path: str) -> dict:
    """Directions of one free-text prompt from its UMT5 context `emb`, with the seed rule of
    prompt index `prompt_index`, written to `path`. Loads the base teacher and frees it again."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    teacher = M.load_transformer("teacher", device)
    rec = save(path, prompt_index, prompt, *estimate(teacher, emb.to(device, M.DTYPE), prompt_index, device))
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[wan-tf] {path}: s = {[round(float(v), 1) for v in rec['s']]}, "
          f"pc1_share {rec['pc1_share']:.3f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompt-ids", default=None, help="VBench indices, e.g. '0-299' (default all)")
    ap.add_argument("--shard", type=int, default=0, help="this worker takes the prompts with index %% num-shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=None, help=f"default {C.direction_dir()}")
    ap.add_argument("--prompt", default=None, help="single-prompt mode: the text")
    ap.add_argument("--prompt-index", type=int, default=0, help="single-prompt mode: prompt index of the seed rule")
    ap.add_argument("--out", default=None,
                    help=f"single-prompt mode: output file (default {C.output_dir('prompt', '<slug>', 'directions.pt')})")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in 0..{a.num_shards - 1}")
    dev = a.device

    if a.prompt is not None:
        tok, enc = M.text_encoder(dev)
        emb = M.embed(tok, enc, a.prompt, dev)[0]
        del tok, enc
        torch.cuda.empty_cache()
        estimate_prompt(a.prompt, emb, a.prompt_index, dev, a.out or prompt_file(a.prompt))
        return

    texts = data.read_prompts(data.VBENCH_300)
    ids = data.parse_ids(a.prompt_ids) if a.prompt_ids else list(range(len(texts)))
    ids = [i for i in ids if i % a.num_shards == a.shard]
    out = a.out_dir or C.direction_dir()
    os.makedirs(out, exist_ok=True)
    present = [i for i in ids if os.path.exists(directions.direction_file(out, i, "v"))]
    foreign = [i for i in present if not directions.matches(directions.direction_file(out, i, "v"), texts[i], i)]
    if foreign:
        sys.exit(f"{out} holds {len(foreign)} direction files of other prompts, e.g. v{foreign[0]:03d}.pt; "
                 "pass another --out-dir")
    pending = [i for i in ids if i not in present]
    print(f"[wan-tf] blocks.{C.ANCHOR_BLOCK}, teacher forward {C.TF_DIRECTION_FORWARD}, {C.FRAMES} frames: "
          f"{len(pending)}/{len(ids)} prompts to do -> {out}", flush=True)
    if not pending:
        return
    encode_prompts.ensure(["vbench"], dev)
    teacher = M.load_transformer("teacher", dev)
    t0, done = time.time(), 0
    for i in pending:
        f = directions.direction_file(out, i, "v")
        if os.path.exists(f) or not claims.claim(f):
            continue
        emb = M.load_ctx(os.path.join(C.vbench_ctx_dir(), f"v{i:03d}.pt"))["encoder_hidden_states"].to(dev, M.DTYPE)
        rec = save(f, i, texts[i], *estimate(teacher, emb, i, dev))
        done += 1
        el = time.time() - t0
        print(f"  [{done}/{len(pending)}] v{i:03d} s1={float(rec['s'][0]):.1f} s8={float(rec['s'][-1]):.1f} "
              f"pc1_share={rec['pc1_share']:.3f} {el / done / 60:.1f} min/prompt "
              f"eta {(len(pending) - done) * el / done / 3600:.1f} h {texts[i][:40]!r}", flush=True)


if __name__ == "__main__":
    main()
