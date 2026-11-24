"""ReDiverse-FT target bank for a Qwen-Image family (offline teacher targets, the Krea / Wan / H3 pattern).

For caption ci (prompts/train_captions.txt, DiffusionDB lines 0-99) and noise ni (0..15) the paired initial latent is
rediverse.seeds.train_noise(ci, ni, [4096, 64]) (CPU float32 -> bf16), the tensor qwenimage.train feeds the student.
The teacher runs tau - 1 forwards on its own schedule (q21: conditional only, as it samples; q20/q2512: true CFG 4 with the
pipeline's norm rescale), with model.step (model.EULER, default the scheduler-exact update), then the conditional
forward tau, truncated after the deepest requested block. Each file records euler, cfg and negative. Its image rows [4096, d_model] bf16 at each block are the targets.

    python -m qwenimage.build_targets --student lightning --tau 10 --blocks 29,21 --shard 0 --num-shards 2

Files: <REDIVERSE_DATA>/<backbone>/bank_k{tau}/b{l:02d}/c{ci:03d}.pt = dict(h [16, 4096, D] bf16, ...); q21 ~537 MB per
caption and block (~54 GB per block). Train contexts: <REDIVERSE_DATA>/<backbone>/ctx/c{ci:03d}.pt.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from qwenimage import config as C
from qwenimage import model as M
from rediverse import data as D
from rediverse.seeds import train_noise

NOISES = D.NOISES_PER_CAPTION
ROLL_BATCH = 4


def train_ctx_dir(fam: str) -> str:
    return C.data_dir(fam, "ctx")


def ensure_train_contexts(fam: str, caps: list[str], ids, device) -> None:
    d = train_ctx_dir(fam)
    os.makedirs(d, exist_ok=True)
    missing = [ci for ci in ids if not os.path.exists(os.path.join(d, f"c{ci:03d}.pt"))]
    if missing:
        for ci, c in zip(missing, M.encode_many(fam, [caps[ci] for ci in missing], device)):
            f = os.path.join(d, f"c{ci:03d}.pt")
            torch.save(dict(ctx=c, prompt=caps[ci], index=ci, family=fam), f + f".tmp{os.getpid()}")
            os.replace(f + f".tmp{os.getpid()}", f)


def load_train_context(fam: str, ci: int, text: str) -> torch.Tensor:
    c = torch.load(os.path.join(train_ctx_dir(fam), f"c{ci:03d}.pt"), map_location="cpu")
    assert c["prompt"] == text, (ci, c["prompt"][:60], text[:60])
    return c["ctx"]


def bank_dir(fam: str, k: int) -> str:
    return C.data_dir(fam, f"bank_k{k}")


@torch.no_grad()
def teacher_rollout(fam: str, teacher, x0, ctx, neg, k: int):
    """Latents after the teacher's first k - 1 sampling steps from x0 (its own schedule and guidance)."""
    return M.rollout(f"teacher:{fam}", teacher, x0, ctx, k - 1, neg=neg, cfg=C.family(fam)["teacher"]["cfg"])


@torch.no_grad()
def teacher_targets(fam: str, teacher, x0, ctx, neg, blocks, k: int) -> dict:
    x = teacher_rollout(fam, teacher, x0, ctx, neg, k)
    t = M.model_timestep(f"teacher:{fam}", k - 1, x.shape[0], x.device)
    return M.forward_to_blocks(fam, teacher, x, ctx, t, blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", required=True, help="the student the bank is for (family, tau check)")
    ap.add_argument("--tau", type=int, required=True, help="teacher forward of the targets (1-based)")
    ap.add_argument("--allow-tau-mismatch", action="store_true", help="ablations: tau other than the sigma match")
    ap.add_argument("--blocks", required=True, help="comma list of blocks to store")
    ap.add_argument("--captions", default=None)
    ap.add_argument("--caption-ids", default="0-99")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    fam = C.family_of(a.student)
    f = C.family(fam)
    tsig = M.teacher_target_sigmas(a.student, a.tau)
    assert a.allow_tau_mismatch or a.tau == tsig["nearest_k"], \
        f"--tau {a.tau} but the teacher forward landing nearest {a.student}'s sigma after step 1 is {tsig['nearest_k']}"
    prov = M.provenance(f"teacher:{fam}", guided=True)
    blocks = sorted(int(b) for b in a.blocks.split(","))
    caps = D.captions(a.captions)
    ids = [ci for ci in D.parse_ids(a.caption_ids) if ci % a.num_shards == a.shard]
    root = bank_dir(fam, a.tau)
    bd = {b: os.path.join(root, f"b{b:02d}") for b in blocks}
    for d in bd.values():
        os.makedirs(d, exist_ok=True)
    todo = [ci for ci in ids if not all(os.path.exists(os.path.join(bd[b], f"c{ci:03d}.pt")) for b in blocks)]
    sg = M.schedule(f"teacher:{fam}")["sigmas"]
    print(f"[qwen-bank/{fam}] teacher forward {a.tau} (sigma {sg[a.tau - 1]:.5f} -> {sg[a.tau]:.5f}; {a.student} sigma1 "
          f"{tsig['student_sigma1']:.5f}), blocks {blocks}, {prov}: {len(todo)}/{len(ids)} captions x {NOISES} -> {root}",
          flush=True)
    for ci in ids:                     # existing files must come from the same arithmetic and guidance
        for b in blocks:
            fn = os.path.join(bd[b], f"c{ci:03d}.pt")
            if os.path.exists(fn):
                M.check_provenance(torch.load(fn, map_location="cpu", mmap=True), prov, fn)
    if not todo:
        return
    ensure_train_contexts(fam, caps, todo, a.device)
    from qwenimage.generate import negative_context
    neg = negative_context(fam, a.device).to(a.device) if prov["negative"] is not None else None
    teacher = M.load_transformer(f"teacher:{fam}", a.device)
    rel = M.check_truncation(fam, teacher, load_train_context(fam, todo[0], caps[todo[0]]).to(a.device), a.device)
    print(f"[qwen-bank/{fam}] truncated forward vs hooked full forward: max rel diff {rel:.2e}", flush=True)
    assert rel < 1e-6, rel
    t0, done = time.time(), 0
    for ci in todo:
        ctx = load_train_context(fam, ci, caps[ci]).to(a.device)
        H = {b: torch.empty(NOISES, f["n_img"], f["d_model"], dtype=C.DTYPE) for b in blocks}
        for n0 in range(0, NOISES, ROLL_BATCH):
            ns = list(range(n0, min(NOISES, n0 + ROLL_BATCH)))
            x0 = torch.stack([train_noise(ci, ni, f["latent_shape"]) for ni in ns]).to(a.device, C.DTYPE)
            out = teacher_targets(fam, teacher, x0, ctx, neg, blocks, a.tau)
            for b in blocks:
                H[b][n0:n0 + len(ns)] = out[b].cpu()
        for b in blocks:
            fn = os.path.join(bd[b], f"c{ci:03d}.pt")
            torch.save(dict(h=H[b], ci=ci, caption=caps[ci], noises=list(range(NOISES)), block=b, k=a.tau,
                            sigma_in=sg[a.tau - 1], sigma_out=sg[a.tau], teacher=f["repo"], revision=f["revision"],
                            student=a.student, **prov),
                       fn + f".tmp{os.getpid()}")
            os.replace(fn + f".tmp{os.getpid()}", fn)
        done += 1
        el = time.time() - t0
        print(f"  [{done}/{len(todo)}] c{ci:03d} h_std(b{blocks[0]})={float(H[blocks[0]].float().std()):.3f} "
              f"{el / done:.0f}s/caption eta {(len(todo) - done) * el / done / 3600:.2f}h", flush=True)
    gaps = [(b, ci) for ci in ids for b in blocks if not os.path.exists(os.path.join(bd[b], f"c{ci:03d}.pt"))]
    if gaps:
        sys.exit(f"[qwen-bank] shard {a.shard}/{a.num_shards} INCOMPLETE: {len(gaps)} files missing, e.g. {gaps[:3]}")
    print(f"[qwen-bank] shard {a.shard}/{a.num_shards} done", flush=True)


if __name__ == "__main__":
    main()
