"""ReDiverse-FT training for a Qwen-Image student (paper Sec. 4.1, App. D.2; the Krea 2 recipe).

The student (base transformer + its card LoRA, or its own full weights) gets a second LoRA, "rediverse" (r 32,
alpha 32, B = 0), on the attention and MLP linears of transformer_blocks 0..l. Loss on the image rows of block l: the
student's step-1 forward (sigma 1) from the paired initial latent vs the bank target (qwenimage.build_targets:
teacher forward tau after tau - 1 teacher steps from the same latent and caption).

  loss = mse : MSE(h_s, h_t)  (paper Eq. 1)

Recipe (rediverse.data): 100 captions x 16 noises, 4 captions x 8 noises = 32 pairs per step, 200 steps,
AdamW 1e-4 (wd 0.01), grads nan_to_num'd and clipped at 10, fp32 LoRA on the bf16 model, data parallel under torchrun.

  torchrun --nproc_per_node 4 -m qwenimage.train --student lightning --block 20 --tau 10 --run lightning_b20
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from qwenimage import config as C
from qwenimage import model as M
from qwenimage.build_targets import bank_dir, load_train_context
from rediverse import data as D
from rediverse.seeds import train_noise


class Bank:
    """Per-caption target files, rows indexed by noise; preloaded into host RAM (mmap on Lustre is I/O-bound).
    Every file's stored teacher forward k, sigmas and block are checked against the run's (expect dict)."""

    def __init__(self, d: str, expect: dict, cis=None, preload: bool = False, log=print):
        self.d, self.expect, self._open = d, expect, {}
        if preload and cis:
            from concurrent.futures import ThreadPoolExecutor
            t0 = time.time()
            with ThreadPoolExecutor(8) as ex:
                for ci, rec in zip(cis, ex.map(lambda c: torch.load(os.path.join(d, f"c{c:03d}.pt"), map_location="cpu"), cis)):
                    self._check(ci, rec)
                    self._open[ci] = rec
            log(f"[qwen-train] preloaded {len(cis)} target files in {time.time() - t0:.0f}s")

    def _check(self, ci: int, rec: dict):
        e = self.expect
        assert rec["ci"] == ci and rec["k"] == e["k"] and rec["block"] == e["block"], (ci, rec["k"], rec["block"], e)
        assert abs(rec["sigma_in"] - e["sigma_in"]) < 1e-6 and abs(rec["sigma_out"] - e["sigma_out"]) < 1e-6, \
            (ci, rec["sigma_in"], rec["sigma_out"], e)
        M.check_provenance(rec, e["prov"], f"{self.d}/c{ci:03d}.pt")

    def rows(self, ci: int, nis: list[int]) -> torch.Tensor:
        if ci not in self._open:
            rec = torch.load(os.path.join(self.d, f"c{ci:03d}.pt"), map_location="cpu", mmap=True)
            self._check(ci, rec)
            self._open[ci] = rec
        return self._open[ci]["h"][nis]


def check_checkpointing(fam: str, model, x, ctx, t, block: int) -> dict:
    """Grad-enabled forward to `block` with per-block checkpointing vs without (run before the adapter exists):
    outputs must agree exactly; input gradients of a fixed random projection must agree up to the backward's own
    run-to-run noise, measured as plain vs plain (SDPA backward kernels accumulate nondeterministically in bf16)."""
    def run(ck: bool):
        xx = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            h = M.forward_to_blocks(fam, model, xx, ctx, t, [block], checkpoint=ck)[block]
            w = torch.randn(h.shape, generator=torch.Generator("cpu").manual_seed(1)).to(h.device, h.dtype)
            (h.float() * w.float()).sum().backward()
        out = (h.detach().float(), xx.grad.detach().float())
        del h, xx
        torch.cuda.empty_cache()
        return out
    plain, plain2, ckpt = run(False), run(False), run(True)
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-12))
    return dict(out_rel=rel(ckpt[0], plain[0]), grad_rel=rel(ckpt[1], plain[1]),
                grad_floor=rel(plain2[1], plain[1]), out_floor=rel(plain2[0], plain[0]))


def checkpointing_ok(ck: dict) -> bool:
    return ck["out_rel"] < 1e-6 and ck["grad_rel"] <= max(3.0 * ck["grad_floor"], 1e-3)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", required=True)
    ap.add_argument("--block", type=int, required=True, help="collapsed block l (loss site, last LoRA block)")
    ap.add_argument("--tau", type=int, required=True)
    ap.add_argument("--allow-tau-mismatch", action="store_true", help="ablations: tau other than the sigma match")
    ap.add_argument("--lora-last", type=int, default=None)
    ap.add_argument("--rank", type=int, default=D.LORA_RANK)
    ap.add_argument("--steps", type=int, default=D.STEPS)
    ap.add_argument("--lr", type=float, default=D.LR)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-preload", action="store_true")
    a = ap.parse_args()
    fam = C.family_of(a.student)
    f = C.family(fam)
    lora_last = a.block if a.lora_last is None else a.lora_last
    out = C.output_dir(fam, a.student, "train", a.run)
    adapters = os.path.join(out, "adapters")
    bdir = os.path.join(bank_dir(fam, a.tau), f"b{a.block:02d}")
    tsig = M.teacher_target_sigmas(a.student, a.tau)
    assert a.allow_tau_mismatch or a.tau == tsig["nearest_k"], \
        f"--tau {a.tau} but the teacher forward landing nearest {a.student}'s sigma after step 1 is {tsig['nearest_k']}"

    # every rank decides the same way (nothing writes adapters before training starts), so no rank is left waiting
    if any(f.endswith(".safetensors") for f in (os.listdir(adapters) if os.path.isdir(adapters) else [])) and not a.overwrite:
        sys.exit(f"{adapters} already holds adapters; pass --overwrite or another --run")
    rank, world = D.dist_init()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device("cuda", torch.cuda.current_device())
    assert D.SAMPLES_PER_STEP % world == 0, world
    is_main = rank == 0
    os.makedirs(adapters, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    caps = D.captions()
    plan = list(D.StepSampler(a.seed).steps(a.steps))
    cis = sorted({ci for st in plan for ci, _ in D.flatten(st)})
    missing = [ci for ci in cis if not os.path.exists(os.path.join(bdir, f"c{ci:03d}.pt"))]
    assert not missing, f"bank {bdir} lacks captions {missing[:5]} ({len(missing)})"
    prov = M.provenance(f"teacher:{fam}", guided=True)
    # host RAM: every rank preloads the whole bank; fall back to mmap when world x bank would exceed ~70% of the job
    bank_bytes = sum(os.path.getsize(os.path.join(bdir, f"c{ci:03d}.pt")) for ci in cis)
    mem_mb = float(os.environ.get("SLURM_MEM_PER_NODE", "0") or 0)
    preload = not a.no_preload and (mem_mb == 0 or world * bank_bytes < 0.7 * mem_mb * 2 ** 20)
    if is_main:
        print(f"[qwen-train] bank {bank_bytes / 2 ** 30:.0f} GiB x {world} ranks vs job memory {mem_mb / 1024:.0f} GiB -> "
              f"{'preload' if preload else 'mmap'}; teacher provenance {prov}; sampling EULER={M.EULER}", flush=True)
    bank = Bank(bdir, dict(k=a.tau, block=a.block, sigma_in=tsig["sigma_in"], sigma_out=tsig["sigma_out"], prov=prov),
                cis, preload=preload, log=lambda m: print(m, flush=True) if is_main else None)
    ctxs = {ci: load_train_context(fam, ci, caps[ci]) for ci in cis}

    model = M.load_transformer(a.student, dev)
    rel = M.check_truncation(fam, model, ctxs[cis[0]].to(dev), dev)
    assert rel < 1e-6, f"truncation check failed: {rel}"
    t1 = lambda b: M.model_timestep(a.student, 0, b, dev)
    with torch.no_grad():
        x0 = train_noise(cis[0], 0, f["latent_shape"])[None].to(dev, C.DTYPE)
        h0 = M.forward_to_blocks(fam, model, x0, ctxs[cis[0]].to(dev), t1(1), [a.block])[a.block].float()
        tt = bank.rows(cis[0], [0]).to(dev).float()
        residual = dict(mse=float(F.mse_loss(h0, tt)), var_t=float(tt.var()), var_s=float(h0.var()),
                        cos=float(F.cosine_similarity(h0.flatten(), tt.flatten(), dim=0)))
    ck = check_checkpointing(fam, model, x0, ctxs[cis[0]].to(dev), t1(1), a.block)
    if is_main:
        print(f"[qwen-train] checkpointed vs plain forward to block {a.block}: {ck}", flush=True)
    assert checkpointing_ok(ck), f"checkpointed backward differs beyond the plain-vs-plain noise: {ck}"
    info = M.attach_lora(fam, model, lora_last, rank=a.rank, alpha=a.rank)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.dtype == torch.float32 for p in params)
    if world > 1:
        import torch.distributed as dist
        for p in params:
            dist.broadcast(p.data, src=0)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=D.WEIGHT_DECAY, betas=D.BETAS, eps=D.EPS)
    meta = D.run_meta("qwenimage", dict(
        family=fam, backbone=f["backbone"], teacher=f["repo"], teacher_revision=f["revision"], student=a.student,
        student_cfg=C.student(a.student), block=a.block, lora_last=lora_last, tau=a.tau, loss="mse", lr=a.lr,
        steps=a.steps, rank=a.rank, chunk=a.chunk, world=world, seed=a.seed, lora=info, residual_pre_lora=residual,
        checkpoint_check=ck, teacher_sigmas=tsig, bank_provenance=prov, euler=M.EULER,
        res=C.RES, bank=bdir, student_lora=model._qwen_info.get("student_lora"),
        gate="adapter active in student step 1 only"))
    if is_main:
        D.write_run_meta(os.path.join(out, "config.json"), meta)
        print(f"[qwen-train] {a.student}: block {a.block}, tau {a.tau}, LoRA 0..{lora_last}: {info['n_modules']} modules, "
              f"{info['n_params'] / 1e6:.1f}M params | world {world} | residual {residual}", flush=True)

    def save(step: int):
        if is_main:
            from safetensors.torch import save_file
            path = os.path.join(adapters, f"rediverse_ft_{step:07d}.safetensors")
            save_file(M.lora_state_dict(model), path + ".tmp", metadata={
                "step": str(step), "block": str(a.block), "lora_last": str(lora_last), "rank": str(a.rank),
                "tau": str(a.tau), "student": a.student, "family": fam})
            os.replace(path + ".tmp", path)
            print(f"[save] {path}", flush=True)

    model.train()
    losses = []
    n_step = D.SAMPLES_PER_STEP
    t_start = time.time()
    for step, step_batches in enumerate(plan, 1):
        opt.zero_grad(set_to_none=True)
        acc, n_mine, t0, sp_s, sp_t = 0.0, 0, time.time(), 0.0, 0.0
        for ci, nis in D.group_by_caption(D.shard_for_rank(step_batches, rank, world)):
            ctx = ctxs[ci].to(dev)
            for j in range(0, len(nis), a.chunk):
                idx = nis[j:j + a.chunk]
                x = torch.stack([train_noise(ci, ni, f["latent_shape"]) for ni in idx]).to(dev, C.DTYPE)
                h_t = bank.rows(ci, idx).to(dev, non_blocking=True).float()
                h_s = M.forward_to_blocks(fam, model, x, ctx, t1(len(idx)), [a.block], checkpoint=True)[a.block].float()
                loss = F.mse_loss(h_s, h_t)
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
        if step == 1:              # the stacked student LoRA + FT adapter must actually receive gradient
            assert gnorm > 0 and n_bad == 0, f"step 1: |g| {gnorm}, non-finite {n_bad}"
        losses.append(D.dist_allreduce_scalar(acc / n_mine, world))
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
        first, last = sum(losses[:5]) / len(losses[:5]), sum(losses[-10:]) / len(losses[-10:])
        print(f"loss: first 5 steps {first:.6g} -> last 10 steps {last:.6g} ({100 * (last / first - 1):+.1f}%)", flush=True)
        if last >= first:
            print("WARNING: the loss did not fall", flush=True)
        print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
