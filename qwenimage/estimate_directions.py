"""ReDiverse-TF directions for a Qwen-Image family (paper Sec. 4.2).

Per prompt, the top-8 principal directions of the teacher's image rows at block l, teacher forward tau.

    python -m qwenimage.estimate_directions --student lightning --tau 10 --blocks 29,21 --shard 0 --num-shards 2

For GenEval prompt i the teacher is rolled CONDITIONAL-ONLY (as in every released port) on its own schedule from 25
latents [4096, 64] drawn on a CPU generator seeded i * 1000 + j, j = 100..124 (disjoint from the evaluation samples):
tau - 1 full forwards with model.step (model.EULER), then forward tau, truncated after the deepest requested block. Each block's
image rows are flattened (D = 4096 x d_model) and rediverse.directions.directions_from_samples gives U [8, D] bf16,
s [8] and eigenvalues. Files: <out-dir>/b{l:02d}/p{i:03d}.pt (q21 ~270 MB each).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from qwenimage import config as C
from qwenimage import model as M
from qwenimage.generate import ctx_dir, ensure_contexts, load_context
from rediverse import data as D
from rediverse.directions import N_DIR, direction_file, directions_from_samples, matches, save_directions
from rediverse.seeds import N_DIRECTION_SAMPLES, cpu_randn, direction_sample_indices, image_direction_seed

ROLL_BATCH = 5


@torch.no_grad()
def teacher_states(fam: str, teacher, x0, ctx, blocks, k: int) -> dict:
    who = f"teacher:{fam}"
    x = M.rollout(who, teacher, x0, ctx, k - 1)                 # conditional only
    t = M.model_timestep(who, k - 1, x.shape[0], x.device)
    return M.forward_to_blocks(fam, teacher, x, ctx, t, blocks)


@torch.no_grad()
def estimate(fam: str, teacher, ctx, prompt_idx: int, blocks, device, k: int, n_lat: int = N_DIRECTION_SAMPLES,
             n_dir: int = N_DIR) -> dict:
    f = C.family(fam)
    seeds = direction_sample_indices(n_lat)
    X = {b: torch.empty(n_lat, f["d_block"], dtype=torch.bfloat16) for b in blocks}
    ctx = ctx.to(device, C.DTYPE)
    for j0 in range(0, n_lat, ROLL_BATCH):
        js = seeds[j0:j0 + ROLL_BATCH]
        x0 = torch.stack([cpu_randn(image_direction_seed(prompt_idx, j), f["latent_shape"]) for j in js]).to(device, C.DTYPE)
        st = teacher_states(fam, teacher, x0, ctx, blocks, k)
        for b in blocks:
            X[b][j0:j0 + len(js)] = st[b].reshape(len(js), -1).to(torch.bfloat16).cpu()
    return dict(dirs={b: directions_from_samples(X[b], n_dir) for b in blocks}, seeds=seeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", required=True, help="the student the directions are for (family, tau check)")
    ap.add_argument("--tau", type=int, required=True)
    ap.add_argument("--allow-tau-mismatch", action="store_true")
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--prompts", default=D.GENEVAL_553)
    ap.add_argument("--prompt-ids", default=D.GENEVAL_SUBSET100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=None, help="default <REDIVERSE_DATA>/<backbone>/directions_k{tau}")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    fam = C.family_of(a.student)
    f = C.family(fam)
    tsig = M.teacher_target_sigmas(a.student, a.tau)
    assert a.allow_tau_mismatch or a.tau == tsig["nearest_k"], (a.tau, tsig)
    prov = M.provenance(f"teacher:{fam}", guided=False)          # conditional-only rollouts
    blocks = [int(b) for b in a.blocks.split(",")]
    assert all(0 <= b < f["n_blocks"] for b in blocks), blocks
    texts = D.read_prompts(a.prompts)
    ids = [i for i in D.parse_ids(a.prompt_ids) if i % a.num_shards == a.shard]
    out_root = a.out_dir or C.data_dir(fam, f"directions_k{a.tau}")
    bdirs = {b: os.path.join(out_root, f"b{b:02d}") for b in blocks}
    for d in bdirs.values():
        os.makedirs(d, exist_ok=True)
    for i in ids:
        for b in blocks:
            fn = direction_file(bdirs[b], i)
            if os.path.exists(fn):
                if not matches(fn, texts[i]):
                    sys.exit(f"{fn} holds another prompt's directions")
                M.check_provenance(torch.load(fn, map_location="cpu", mmap=True), prov, fn)
    todo = [i for i in ids if not all(os.path.exists(direction_file(bdirs[b], i)) for b in blocks)]
    sg = M.schedule(f"teacher:{fam}")["sigmas"]
    print(f"[qwen-dirs/{fam}] {prov}; teacher forward {a.tau} (sigma {sg[a.tau - 1]:.5f} -> {sg[a.tau]:.5f}), blocks {blocks}, "
          f"{N_DIRECTION_SAMPLES} latents: {len(todo)}/{len(ids)} prompts -> {out_root}", flush=True)
    if not todo:
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    cdir = ctx_dir(fam)
    ensure_contexts(fam, {i: texts[i] for i in todo}, cdir, a.device)
    teacher = M.load_transformer(f"teacher:{fam}", a.device)
    t0, done = time.time(), 0
    for i in todo:
        r = estimate(fam, teacher, load_context(cdir, i, texts[i]), i, blocks, a.device, k=a.tau)
        for b in blocks:
            U, s, ev = r["dirs"][b]
            rec = save_directions(direction_file(bdirs[b], i), U, s, ev, prompt_idx=i, prompt=texts[i], k=a.tau,
                                  block=b, stream="image", arm="teacher", n=len(r["seeds"]), seeds=r["seeds"],
                                  shape=(f["n_img"], f["d_model"]), res=C.RES, teacher=f["repo"],
                                  teacher_revision=f["revision"], sigma_in=sg[a.tau - 1], sigma_out=sg[a.tau],
                                  student=a.student, **prov)
        done += 1
        el = time.time() - t0
        print(f"  [{done}/{len(todo)}] p{i:03d} s1(b{blocks[0]})={float(r['dirs'][blocks[0]][1][0]):.3f} "
              f"pc1={rec['pc1_share']:.3f} {el / done:.1f}s/prompt eta {(len(todo) - done) * el / done / 3600:.2f}h",
              flush=True)
    gaps = [(b, i) for i in ids for b in blocks if not os.path.exists(direction_file(bdirs[b], i))]
    if gaps:
        sys.exit(f"[qwen-dirs] shard {a.shard}/{a.num_shards} INCOMPLETE: {len(gaps)} files missing, e.g. {gaps[:3]}")
    print(f"[qwen-dirs] shard {a.shard}/{a.num_shards} done", flush=True)


if __name__ == "__main__":
    main()
