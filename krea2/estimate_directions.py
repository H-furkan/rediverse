"""ReDiverse-TF directions for Krea 2 (paper Sec. 4.2).

Per prompt, the top-8 principal directions of the teacher's (Krea-2-Raw) image rows at block l, teacher forward tau.

    python -m krea2.estimate_directions --blocks 14 --prompt-ids 0-99 --shard 0 --num-shards 2

For GenEval prompt i the teacher is rolled CONDITIONAL-ONLY (no CFG, as in every released port) on its own 52-step
schedule (mu from the 1024^2 resolution) from 25 latents [4096, 64] drawn on a CPU generator seeded i * 1000 + j,
j = 100..124 (disjoint from the evaluation samples): tau - 1 full forwards with Euler updates, then forward tau,
hooked at every requested block and stopped after the deepest one. Each block's image rows [4096, 6144] are flattened
(D = 25,165,824) and rediverse.directions.directions_from_samples gives U [8, D] bf16, s [8], eigenvalues.
One rollout serves several blocks (--blocks 10,14,18): files <out-dir>/b{l:02d}/p{i:03d}.pt (~400 MB each).
Contexts are the compacted conditional text taps (krea2.model.encode), cached in <REDIVERSE_DATA>/krea2/geneval_ctx.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from krea2 import config as C
from krea2 import model as M
from rediverse import data as D
from rediverse.directions import N_DIR, direction_file, directions_from_samples, matches, save_directions
from rediverse.seeds import N_DIRECTION_SAMPLES, cpu_randn, direction_sample_indices, image_direction_seed

ROLL_BATCH = 5


class _Stop(Exception):
    pass


def ctx_dir() -> str:
    """Cached conditional contexts by prompt index (KREA2_CTX_DIR overrides it for another prompt file)."""
    return os.environ.get("KREA2_CTX_DIR") or C.data_dir("geneval_ctx")


def ensure_contexts(texts: dict, cdir: str, device) -> None:
    """Write <cdir>/p{idx:03d}.pt (compacted conditional context) for every prompt of {idx: text} not cached."""
    os.makedirs(cdir, exist_ok=True)
    missing = [i for i in sorted(texts) if not os.path.exists(os.path.join(cdir, f"p{i:03d}.pt"))]
    if not missing:
        return
    ctxs = M.encode_many([texts[i] for i in missing], device)
    for i, c in zip(missing, ctxs):
        f = os.path.join(cdir, f"p{i:03d}.pt")
        torch.save(dict(ctx=c, prompt=texts[i], index=i), f + f".tmp{os.getpid()}")
        os.replace(f + f".tmp{os.getpid()}", f)


def load_context(cdir: str, i: int, text: str) -> torch.Tensor:
    c = torch.load(os.path.join(cdir, f"p{i:03d}.pt"), map_location="cpu")
    assert c["prompt"] == text, f"ctx p{i:03d} holds {c['prompt']!r}, expected {text!r}"
    return c["ctx"]


@torch.no_grad()
def teacher_states(teacher, x0: torch.Tensor, ctx: torch.Tensor, blocks: list[int], k: int) -> dict:
    """{block: image rows [B, 4096, 6144]} of the teacher's conditional forward k from x0 [B, 4096, 64]."""
    sg = M.schedule("teacher")["sigmas"]
    grab, last_b = {}, max(blocks)
    x = x0
    for i in range(k - 1):                       # forwards 1..k-1: full, conditional, Euler updates
        v = M.forward(teacher, x, ctx, sg[i])
        x = M.euler_step(x, v, sg[i], sg[i + 1])
    hs = []                                      # forward k: hooked, stopped after the deepest block
    for b in blocks:
        def hook(_m, _i, out, _b=b):
            grab[_b] = M.extract(f"block{_b}", out).detach()
            if _b == last_b:
                raise _Stop
        hs.append(teacher.transformer_blocks[b].register_forward_hook(hook))
    try:
        M.forward(teacher, x, ctx, sg[k - 1])
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    assert set(grab) == set(blocks), f"hooks fired for {sorted(grab)}, wanted {blocks}"
    return grab


@torch.no_grad()
def estimate(teacher, ctx: torch.Tensor, prompt_idx: int, blocks: list[int], device, k: int = C.DIRECTION_STEP,
             n_lat: int = N_DIRECTION_SAMPLES, n_dir: int = N_DIR) -> dict:
    """{block: (U, s, eigvals)} plus seeds, from n_lat conditional teacher rollouts of one prompt."""
    seeds = direction_sample_indices(n_lat)
    X = {b: torch.empty(n_lat, C.D_BLOCK, dtype=torch.bfloat16) for b in blocks}
    ctx = ctx.to(device, C.DTYPE)
    for j0 in range(0, n_lat, ROLL_BATCH):
        js = seeds[j0:j0 + ROLL_BATCH]
        x0 = torch.stack([cpu_randn(image_direction_seed(prompt_idx, j), C.LATENT_SHAPE) for j in js]).to(device, C.DTYPE)
        st = teacher_states(teacher, x0, ctx, blocks, k)
        for b in blocks:
            X[b][j0:j0 + len(js)] = st[b].reshape(len(js), -1).to(torch.bfloat16).cpu()
    out = {b: directions_from_samples(X[b], n_dir) for b in blocks}
    return dict(dirs=out, seeds=seeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blocks", default=None, help="comma list of blocks (default: config ANCHOR_BLOCK)")
    ap.add_argument("--k", type=int, default=C.DIRECTION_STEP, help="teacher forward (tau)")
    ap.add_argument("--prompts", default=D.GENEVAL_553)
    ap.add_argument("--prompt-ids", default=D.GENEVAL_SUBSET100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=None, help="default <REDIVERSE_DATA>/krea2/directions_k{k}")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    blocks = [int(b) for b in a.blocks.split(",")] if a.blocks else [C.ANCHOR_BLOCK]
    assert all(0 <= b < C.N_BLOCKS for b in blocks), f"blocks {blocks}: set --blocks or KREA2_ANCHOR_BLOCK"
    texts = D.read_prompts(a.prompts)
    ids = [i for i in D.parse_ids(a.prompt_ids) if i % a.num_shards == a.shard]
    out_root = a.out_dir or C.data_dir(f"directions_k{a.k}")
    bdirs = {b: os.path.join(out_root, f"b{b:02d}") for b in blocks}
    for d in bdirs.values():
        os.makedirs(d, exist_ok=True)
    todo = [i for i in ids if not all(os.path.exists(direction_file(bdirs[b], i)) for b in blocks)]
    for i in ids:
        for b in blocks:
            f = direction_file(bdirs[b], i)
            if os.path.exists(f) and not matches(f, texts[i]):
                sys.exit(f"{f} holds another prompt's directions")
    ts = M.teacher_target_sigmas(a.k)
    print(f"[krea2-dirs] teacher forward {a.k} (sigma {ts['sigma_in']:.5f} -> {ts['sigma_out']:.5f}; student sigma1 "
          f"{ts['student_sigma1']:.5f}), blocks {blocks}, {N_DIRECTION_SAMPLES} latents: {len(todo)}/{len(ids)} "
          f"prompts -> {out_root}", flush=True)
    if not todo:
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    cdir = ctx_dir()
    ensure_contexts({i: texts[i] for i in todo}, cdir, a.device)
    teacher = M.load_transformer("teacher", a.device)
    t0, done = time.time(), 0
    for i in todo:
        r = estimate(teacher, load_context(cdir, i, texts[i]), i, blocks, a.device, k=a.k)
        for b in blocks:
            U, s, ev = r["dirs"][b]
            rec = save_directions(direction_file(bdirs[b], i), U, s, ev, prompt_idx=i, prompt=texts[i], k=a.k,
                                  block=b, stream="image", arm="teacher", n=len(r["seeds"]), seeds=r["seeds"],
                                  shape=(C.N_IMG_TOKENS, C.D_MODEL), res=C.RES, teacher=C.TEACHER_ID,
                                  teacher_revision=C.TEACHER_REVISION, sigma_in=ts["sigma_in"],
                                  sigma_out=ts["sigma_out"])
        done += 1
        el = time.time() - t0
        print(f"  [{done}/{len(todo)}] p{i:03d} s1(b{blocks[0]})={float(r['dirs'][blocks[0]][1][0]):.3f} "
              f"pc1={rec['pc1_share']:.3f} {el / done:.1f}s/prompt eta {(len(todo) - done) * el / done / 3600:.2f}h",
              flush=True)
    gaps = [(b, i) for i in ids for b in blocks if not os.path.exists(direction_file(bdirs[b], i))]
    if gaps:
        sys.exit(f"[krea2-dirs] shard {a.shard}/{a.num_shards} INCOMPLETE: {len(gaps)} files missing, e.g. {gaps[:3]}")
    print(f"[krea2-dirs] shard {a.shard}/{a.num_shards} done", flush=True)


if __name__ == "__main__":
    main()
