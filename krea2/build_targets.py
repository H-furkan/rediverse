"""ReDiverse-FT target bank for Krea 2 (offline teacher targets, the Wan / H3 pattern).

For caption ci (prompts/train_captions.txt, DiffusionDB lines 0-99) and noise ni (0..15), the paired initial latent
is rediverse.seeds.train_noise(ci, ni, [4096, 64]) (CPU float32 -> bf16), the same tensor krea2.train feeds the
student. The teacher (Krea-2-Raw) runs tau - 1 = 4 forwards on its own 52-step schedule with Krea guidance g
(default 3.5 = conventional CFG 4.5; cond and uncond as two forwards, since compacted contexts differ in length),
v = v_c + g (v_c - v_u), Euler updates in fp32; then the conditional forward tau = 5, truncated after the deepest
requested block. Its image rows [4096, 6144] bf16 at each block are the targets.

    python -m krea2.build_targets --blocks 12,14,16 --shard 0 --num-shards 2

Files: <REDIVERSE_DATA>/krea2/bank_k{tau}_g{g}/b{l:02d}/c{ci:03d}.pt = dict(h [16, 4096, 6144] bf16, ...),
~805 MB per caption and block (~80 GB per block for the 100 captions). Contexts: <REDIVERSE_DATA>/krea2/ctx/.
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
from rediverse.seeds import train_noise

NOISES = D.NOISES_PER_CAPTION
ROLL_BATCH = 4


def train_ctx_dir() -> str:
    return C.data_dir("ctx")


def ensure_train_contexts(caps: list[str], ids, device) -> None:
    d = train_ctx_dir()
    os.makedirs(d, exist_ok=True)
    missing = [ci for ci in ids if not os.path.exists(os.path.join(d, f"c{ci:03d}.pt"))]
    if missing:
        ctxs = M.encode_many([caps[ci] for ci in missing], device)
        for ci, c in zip(missing, ctxs):
            f = os.path.join(d, f"c{ci:03d}.pt")
            torch.save(dict(ctx=c, prompt=caps[ci], index=ci), f + f".tmp{os.getpid()}")
            os.replace(f + f".tmp{os.getpid()}", f)


def load_train_context(ci: int, text: str) -> torch.Tensor:
    c = torch.load(os.path.join(train_ctx_dir(), f"c{ci:03d}.pt"), map_location="cpu")
    assert c["prompt"] == text, (ci, c["prompt"][:60], text[:60])
    return c["ctx"]


def bank_dir(k: int, g: float) -> str:
    return C.data_dir(f"bank_k{k}_g{g:g}")


@torch.no_grad()
def teacher_targets(teacher, x0, ctx, uctx, blocks, k: int, g: float) -> dict:
    sg = M.schedule("teacher")["sigmas"]
    x = x0
    for i in range(k - 1):
        v = M.forward(teacher, x, ctx, sg[i])
        if g > 0:
            vu = M.forward(teacher, x, uctx, sg[i])
            v = v + g * (v - vu)
        x = M.euler_step(x, v, sg[i], sg[i + 1])
    return M.forward_to_blocks(teacher, x, ctx, sg[k - 1], blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blocks", required=True, help="comma list of blocks to store")
    ap.add_argument("--k", type=int, default=C.TEACHER_TARGET_STEP)
    ap.add_argument("--guidance", type=float, default=C.GUIDANCE["teacher"], help="Krea convention")
    ap.add_argument("--captions", default=None)
    ap.add_argument("--caption-ids", default="0-99")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    blocks = sorted(int(b) for b in a.blocks.split(","))
    caps = D.captions(a.captions)
    ids = [ci for ci in D.parse_ids(a.caption_ids) if ci % a.num_shards == a.shard]
    root = bank_dir(a.k, a.guidance)
    bd = {b: os.path.join(root, f"b{b:02d}") for b in blocks}
    for d in bd.values():
        os.makedirs(d, exist_ok=True)
    todo = [ci for ci in ids if not all(os.path.exists(os.path.join(bd[b], f"c{ci:03d}.pt")) for b in blocks)]
    ts = M.teacher_target_sigmas(a.k)
    print(f"[krea2-bank] teacher forward {a.k} (sigma {ts['sigma_in']:.5f} -> {ts['sigma_out']:.5f}, student "
          f"{ts['student_sigma1']:.5f}), guidance {a.guidance} (Krea), blocks {blocks}: {len(todo)}/{len(ids)} "
          f"captions x {NOISES} -> {root}", flush=True)
    if not todo:
        return
    ensure_train_contexts(caps, todo, a.device)
    from krea2.generate import uncond_context
    uctx = uncond_context(a.device).to(a.device)
    teacher = M.load_transformer("teacher", a.device)
    rel = M.check_truncation(teacher, load_train_context(todo[0], caps[todo[0]]).to(a.device), a.device)
    print(f"[krea2-bank] truncated forward vs hooked full forward: max rel diff {rel:.2e}", flush=True)
    assert rel < 1e-6, rel
    t0, done = time.time(), 0
    for ci in todo:
        ctx = load_train_context(ci, caps[ci]).to(a.device)
        H = {b: torch.empty(NOISES, C.N_IMG_TOKENS, C.D_MODEL, dtype=C.DTYPE) for b in blocks}
        for n0 in range(0, NOISES, ROLL_BATCH):
            ns = list(range(n0, min(NOISES, n0 + ROLL_BATCH)))
            x0 = torch.stack([train_noise(ci, ni, C.LATENT_SHAPE) for ni in ns]).to(a.device, C.DTYPE)
            out = teacher_targets(teacher, x0, ctx, uctx, blocks, a.k, a.guidance)
            for b in blocks:
                H[b][n0:n0 + len(ns)] = out[b].cpu()
        for b in blocks:
            f = os.path.join(bd[b], f"c{ci:03d}.pt")
            torch.save(dict(h=H[b], ci=ci, caption=caps[ci], noises=list(range(NOISES)), block=b, k=a.k,
                            guidance=a.guidance, sigma_in=ts["sigma_in"], sigma_out=ts["sigma_out"],
                            teacher=C.TEACHER_ID, revision=C.TEACHER_REVISION), f + f".tmp{os.getpid()}")
            os.replace(f + f".tmp{os.getpid()}", f)
        done += 1
        el = time.time() - t0
        print(f"  [{done}/{len(todo)}] c{ci:03d} h_std(b{blocks[0]})={float(H[blocks[0]].float().std()):.3f} "
              f"{el / done:.0f}s/caption eta {(len(todo) - done) * el / done / 3600:.2f}h", flush=True)
    gaps = [(b, ci) for ci in ids for b in blocks if not os.path.exists(os.path.join(bd[b], f"c{ci:03d}.pt"))]
    if gaps:
        sys.exit(f"[krea2-bank] shard {a.shard}/{a.num_shards} INCOMPLETE: {len(gaps)} files missing, e.g. {gaps[:3]}")
    print(f"[krea2-bank] shard {a.shard}/{a.num_shards} done", flush=True)


if __name__ == "__main__":
    main()
