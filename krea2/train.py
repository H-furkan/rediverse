"""ReDiverse-FT training for Krea 2 (paper Sec. 4.1, App. D.2).

Student Krea-2-Turbo with a LoRA (r 32, alpha 32, B = 0) on every linear of transformer_blocks 0..l. Loss on the
image rows of block l: the student's step-1 forward (sigma 1) from the paired initial latent vs the bank target
(krea2.build_targets: teacher forward tau after tau - 1 guided steps from the same latent and caption).

  loss = mse   : MSE(h_s, h_t)                                   (paper Eq. 1; SDXL / SD3 / Wan ports)
  loss = norm  : 0.5 * MSE(h_s, h_t) / Var(h_t), per latent       (MiniMax-H3 port)

Recipe (rediverse.data): 100 captions x 16 noises, 4 captions x 8 noises = 32 pairs per step, 200 steps,
AdamW 1e-4 (wd 0.01), grads nan_to_num'd and clipped at 10, fp32 LoRA on the bf16 model. Data parallel under
torchrun (each rank takes pairs[rank::world] of every step, LoRA grads averaged), world in {1, 2, 4, 8}.
The student forward stops after block l (krea2.model.forward_to_blocks) with per-block gradient checkpointing.

  torchrun --nproc_per_node 4 -m krea2.train --block 14 --run b14_mse
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from krea2 import config as C
from krea2 import model as M
from krea2.build_targets import bank_dir, load_train_context
from rediverse import data as D
from rediverse.seeds import train_noise


class Bank:
    """Per-caption target files; rows indexed by noise. With preload, every caption's [16, 4096, 6144] bf16 target is
    read sequentially into host RAM at start (~76 GB per block; mmap page faults on Lustre ran at ~20 MB/s and made
    training ~35% I/O-bound)."""

    def __init__(self, d: str, cis=None, preload: bool = False, log=print):
        self.d = d
        self._open = {}
        if preload and cis:
            from concurrent.futures import ThreadPoolExecutor
            t0 = time.time()
            with ThreadPoolExecutor(8) as ex:
                for ci, rec in zip(cis, ex.map(lambda c: torch.load(os.path.join(d, f"c{c:03d}.pt"), map_location="cpu"), cis)):
                    self._open[ci] = rec
            log(f"[krea2-train] preloaded {len(cis)} target files in {time.time() - t0:.0f}s")

    def rows(self, ci: int, nis: list[int]) -> torch.Tensor:
        if ci not in self._open:
            self._open[ci] = torch.load(os.path.join(self.d, f"c{ci:03d}.pt"), map_location="cpu", mmap=True)
        rec = self._open[ci]
        return rec["h"][nis]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", type=int, required=True, help="collapsed block l (loss site, last LoRA block)")
    ap.add_argument("--lora-last", type=int, default=None, help="last LoRA block (default --block)")
    ap.add_argument("--k", type=int, default=C.TEACHER_TARGET_STEP)
    ap.add_argument("--guidance", type=float, default=C.GUIDANCE["teacher"])
    ap.add_argument("--loss", choices=("mse", "norm"), default="mse")
    ap.add_argument("--rank", type=int, default=D.LORA_RANK)
    ap.add_argument("--steps", type=int, default=D.STEPS)
    ap.add_argument("--lr", type=float, default=D.LR)
    ap.add_argument("--chunk", type=int, default=4, help="noises of one caption per forward/backward")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-preload", action="store_true", help="memory-map the bank instead of reading it into RAM")
    a = ap.parse_args()
    lora_last = a.block if a.lora_last is None else a.lora_last
    out = C.output_dir("train", a.run)
    adapters = os.path.join(out, "adapters")
    bdir = os.path.join(bank_dir(a.k, a.guidance), f"b{a.block:02d}")

    rank, world = D.dist_init()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device("cuda", torch.cuda.current_device())
    assert D.SAMPLES_PER_STEP % world == 0, f"world {world} must divide {D.SAMPLES_PER_STEP}"
    is_main = rank == 0
    if is_main and os.path.isdir(adapters) and os.listdir(adapters) and not a.overwrite:
        sys.exit(f"{adapters} is not empty; pass --overwrite or another --run")
    os.makedirs(adapters, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    caps = D.captions()
    sampler = D.StepSampler(a.seed)
    plan = list(sampler.steps(a.steps))
    cis = sorted({ci for st in plan for ci, _ in D.flatten(st)})
    missing = [ci for ci in cis if not os.path.exists(os.path.join(bdir, f"c{ci:03d}.pt"))]
    assert not missing, f"bank {bdir} lacks captions {missing[:5]} ({len(missing)})"
    bank = Bank(bdir, cis, preload=not a.no_preload, log=lambda m: print(m, flush=True) if is_main else None)
    ctxs = {ci: load_train_context(ci, caps[ci]) for ci in cis}

    model = M.load_transformer("student", dev)
    # gate: truncated forward == hooked full forward, before the LoRA exists
    rel = M.check_truncation(model, ctxs[cis[0]].to(dev), dev)
    assert rel < 1e-6, f"truncation check failed: {rel}"
    # pre-LoRA residual on one pair
    with torch.no_grad():
        x0 = train_noise(cis[0], 0, C.LATENT_SHAPE)[None].to(dev, C.DTYPE)
        h0 = M.forward_to_blocks(model, x0, ctxs[cis[0]].to(dev), 1.0, [a.block])[a.block].float()
        t0_ = bank.rows(cis[0], [0]).to(dev).float()
        residual = dict(mse=float(F.mse_loss(h0, t0_)), var_t=float(t0_.var()), var_s=float(h0.var()),
                        cos=float(F.cosine_similarity(h0.flatten(), t0_.flatten(), dim=0)))
    info = M.attach_lora(model, lora_last, rank=a.rank, alpha=a.rank)
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        import torch.distributed as dist
        for p in params:
            dist.broadcast(p.data, src=0)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=D.WEIGHT_DECAY, betas=D.BETAS, eps=D.EPS)
    meta = D.run_meta("krea2", dict(
        backbone=C.BACKBONE, teacher=C.TEACHER_ID, teacher_revision=C.TEACHER_REVISION, student=C.STUDENT_ID,
        student_revision=C.STUDENT_REVISION, block=a.block, lora_last=lora_last, k=a.k, guidance_krea=a.guidance,
        loss=a.loss, lr=a.lr, steps=a.steps, rank=a.rank, chunk=a.chunk, world=world, seed=a.seed, lora=info,
        residual_pre_lora=residual, res=C.RES, bank=bdir, gate="adapter active in student step 1 only"))
    if is_main:
        D.write_run_meta(os.path.join(out, "config.json"), meta)
        print(f"[krea2-train] block {a.block}, LoRA 0..{lora_last}: {info['n_modules']} modules, "
              f"{info['n_params'] / 1e6:.1f}M params | loss {a.loss} | world {world} | residual {residual}", flush=True)

    def save(step: int):
        if is_main:
            from safetensors.torch import save_file
            path = os.path.join(adapters, f"rediverse_ft_{step:07d}.safetensors")
            save_file(M.lora_state_dict(model), path + ".tmp", metadata={"step": str(step), "block": str(a.block),
                                                                        "lora_last": str(lora_last), "rank": str(a.rank)})
            os.replace(path + ".tmp", path)
            print(f"[save] {path}", flush=True)

    model.train()
    n_step = D.SAMPLES_PER_STEP
    t_start = time.time()
    for step, step_batches in enumerate(plan, 1):
        opt.zero_grad(set_to_none=True)
        acc, n_mine, t0 = 0.0, 0, time.time()
        sp_s, sp_t = 0.0, 0.0
        for ci, nis in D.group_by_caption(D.shard_for_rank(step_batches, rank, world)):
            ctx = ctxs[ci].to(dev)
            for j in range(0, len(nis), a.chunk):
                idx = nis[j:j + a.chunk]
                x = torch.stack([train_noise(ci, ni, C.LATENT_SHAPE) for ni in idx]).to(dev, C.DTYPE)
                h_t = bank.rows(ci, idx).to(dev, non_blocking=True).float()
                h_s = M.forward_to_blocks(model, x, ctx, 1.0, [a.block], checkpoint=True)[a.block].float()
                if a.loss == "mse":
                    loss = F.mse_loss(h_s, h_t)
                else:
                    per = ((h_s - h_t) ** 2).flatten(1).mean(1) / h_t.flatten(1).var(1).clamp_min(1e-8)
                    loss = 0.5 * per.mean()
                (loss * (len(idx) * world / n_step)).backward()
                acc += float(loss) * len(idx)
                n_mine += len(idx)
                if len(idx) > 1:
                    with torch.no_grad():
                        sp_s += float(h_s.var(0).mean()) * len(idx)
                        sp_t += float(h_t.var(0).mean()) * len(idx)
                del h_s, h_t, loss, x
        assert n_mine == n_step // world, (n_mine, n_step, world)
        D.dist_allreduce_grads(params, world)
        gnorm, n_bad = D.clip_and_step(params, opt)
        dt = time.time() - t0
        if step % 5 == 0 or step == 1:
            reg = D.dist_allreduce_scalar(acc / n_mine, world)
            ratio = D.dist_allreduce_scalar(sp_s / max(sp_t, 1e-12), world)
            if is_main:
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"iter {step:4d}/{a.steps}  loss {reg:.6g}  |g| {gnorm:.3g}{'  NONFINITE ' + str(n_bad) if n_bad else ''}"
                      f"  spread s/t {ratio:.3f}  {dt:.1f}s/step  peak {peak:.1f} GiB  eta {(a.steps - step) * dt / 60:.0f}m",
                      flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save(step)
    D.dist_barrier(world)
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    if is_main:
        print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
