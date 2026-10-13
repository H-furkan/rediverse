#!/usr/bin/env python
"""ReDiverse-FT training for Wan 2.1 (paper Sec. 4.1, App. D).

    loss = MSE(h_student, h_teacher)    at blocks.28, over all 32,760 video tokens

h_student is the lightx2v student's blocks.28 output at its first step (timestep 999, the
initial unit-noise latent) with a rank-32 LoRA on the attention projections and feed-forward
linears of blocks 0..28; h_teacher is the bank entry of the same (caption, latent) pair
(wan.build_targets). The student forward is truncated after block 28 and checkpointed per block.
Each optimizer step draws 4 captions x 8 of their 16 noises (rediverse.data.StepSampler);
200 steps visit every pair 4 times. AdamW 1e-4, weight decay 0.01, gradient clipping at 10,
LoRA kept in float32 on the bf16 student.

Data parallel under torchrun: every rank builds the same sampler and takes pairs [rank::world]
of each step's 32; a chunk's loss is weighted by n_chunk * world / 32 and LoRA gradients are
averaged across ranks, so a step's gradient is the mean over its 32 pairs at any world size.
The released adapter was trained at world size 8, where every forward holds one sample.

  torchrun --standalone --nproc_per_node 8 -m wan.train
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time


def _pin_local_gpu():
    """Under torchrun each rank sees only its own card (CUDA_VISIBLE_DEVICES[LOCAL_RANK]), so
    every rank trains on cuda:0. Must run before CUDA is initialised."""
    if "LOCAL_RANK" in os.environ:
        pool = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",") if g]
        lr = int(os.environ["LOCAL_RANK"])
        assert lr < len(pool), f"LOCAL_RANK {lr} but only {len(pool)} GPUs in CUDA_VISIBLE_DEVICES"
        os.environ["CUDA_VISIBLE_DEVICES"] = pool[lr]


_pin_local_gpu()

import torch                                        # noqa: E402
import torch.nn.functional as F                     # noqa: E402

from rediverse import data, seeds                   # noqa: E402
from wan import config as C                         # noqa: E402
from wan import model as M                          # noqa: E402
from wan import target_bank as bank                 # noqa: E402

N_BLOCKS_RUN = C.ANCHOR_BLOCK + 1                   # blocks 0..28 are executed, the rest never


def lora_target_modules() -> list[str]:
    """blocks 0..28: attn1/attn2 x to_q/to_k/to_v/to_out.0, then ffn.net.0.proj / ffn.net.2."""
    out = [f"blocks.{i}.{a}.{p}" for i in range(C.LORA_LAST_BLOCK + 1)
           for a in C.LORA_ATTENTIONS for p in C.LORA_PROJECTIONS]
    out += [f"blocks.{i}.{m}" for i in range(C.LORA_LAST_BLOCK + 1) for m in C.LORA_FFN]
    return out


class Student:
    """The LoRA'd student and the capture of its blocks.28 output.

    The adapter is enabled once and never toggled: the trainer's only student forward is the
    first sampling step, so this is the step-1 gate. The forward hook is gated by
    `_capture`, because the checkpointed blocks run their forward again during backward."""

    def __init__(self, device: str, verbose: bool = True):
        self.device = torch.device(device)
        self.verbose = verbose
        self.model = M.load_transformer("student", self.device)
        self._h, self._capture = {}, False

        def hook(_m, _i, o):
            if self._capture:
                self._h["h"] = o
        self.model.blocks[C.ANCHOR_BLOCK].register_forward_hook(hook)
        self.lora_params: list[torch.nn.Parameter] = []
        self._graph_checked = False

    def say(self, msg: str):
        if self.verbose:
            print(f"[wan-train] {msg}", flush=True)

    def emb(self, ctx: dict, k: int) -> torch.Tensor:
        e = ctx.get("_emb")
        if e is None:
            e = ctx["_emb"] = ctx["encoder_hidden_states"].to(self.device, M.DTYPE)
        return e.expand(k, -1, -1)

    def student_activation(self, x, emb, checkpoint: bool = True) -> torch.Tensor:
        """Step-1 student forward truncated after blocks.28; returns the block's output."""
        self._h.clear()
        self._capture = True
        try:
            t = torch.full((x.shape[0],), C.TRAIN_TIMESTEP, dtype=torch.float32, device=x.device)
            M.forward_to_block(self.model, x, t, emb, N_BLOCKS_RUN, checkpoint=checkpoint)
        finally:
            self._capture = False
        return self._h.pop("h")

    # ------------------------------------------------------------------ gates
    def verify_truncation(self, ctx: dict) -> float:
        """The truncated forward equals the model's own forward at blocks.28 (on a fresh unit-noise
        latent, before any LoRA exists)."""
        x = torch.randn(1, *C.LATENT_SHAPE, dtype=torch.float32, device=self.device).to(M.DTYPE)
        t = torch.full((1,), C.TRAIN_TIMESTEP, dtype=torch.float32, device=self.device)
        d = M.assert_truncation_matches(self.model, x, t, self.emb(ctx, 1), N_BLOCKS_RUN)
        self.say(f"truncated forward == model.forward at blocks[{C.ANCHOR_BLOCK}] (max|diff| = {d:.3e})")
        return d

    def residual(self, ctx: dict, x, target) -> dict:
        """Pre-LoRA residual between the student and a bank target of the same pair."""
        with torch.no_grad():
            h_s = self.student_activation(x, self.emb(ctx, 1), checkpoint=False)
            h_t = target.to(self.device, M.DTYPE)
            mse = F.mse_loss(h_s.float(), h_t.float()).item()
            rel = ((h_s.float() - h_t.float()).norm() / h_t.float().norm()).item()
            t_rms = h_t.float().pow(2).mean().sqrt().item()
            s_rms = h_s.float().pow(2).mean().sqrt().item()
        assert mse > 0.0, "student and teacher activations are identical before training: check the arms"
        assert tuple(h_s.shape) == (1, C.TOKENS, C.DIM), tuple(h_s.shape)
        self.say(f"pre-LoRA residual at blocks[{C.ANCHOR_BLOCK}]: MSE={mse!r}  rel-L2={rel!r}  "
                 f"rms teacher={t_rms:.4g} student={s_rms:.4g}")
        del h_s, h_t
        torch.cuda.empty_cache()
        return dict(mse=mse, rel_l2=rel, rms_teacher=t_rms, rms_student=s_rms)

    # ------------------------------------------------------------------- LoRA
    def attach_lora(self) -> list[str]:
        """peft LoRA rank 32, alpha 32, on blocks 0..28, cast to float32 (peft creates it in the
        base dtype; bf16 would round small AdamW updates away)."""
        from peft import LoraConfig

        targets = lora_target_modules()
        mods = dict(self.model.named_modules())
        for name in targets:
            assert isinstance(mods.get(name), torch.nn.Linear), f"LoRA target {name} is not an nn.Linear of the student"
        self.model.requires_grad_(False)
        self.model.add_adapter(LoraConfig(r=C.LORA_RANK, lora_alpha=C.LORA_ALPHA, target_modules=targets))
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
        for n, p in self.model.named_parameters():
            p.requires_grad_("lora_" in n)
        self.lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n]
        n_par = sum(p.numel() for p in self.lora_params)
        self.say(f"LoRA: rank={C.LORA_RANK} modules={len(targets)} tensors={len(self.lora_params)} "
                 f"trainable {n_par / 1e6:.1f}M dtype={self.lora_params[0].dtype}")
        self.model.enable_adapters()
        for n, p in self.model.named_parameters():      # enable_adapters resets requires_grad
            p.requires_grad_("lora_" in n)
        return targets

    def lora_state_dict(self) -> dict:
        return {n: p.detach().to(torch.float32).cpu()
                for n, p in self.model.named_parameters() if "lora_" in n}

    # ------------------------------------------------------------------- step
    def loss(self, ctx: dict, x, target):
        """MSE between the student's blocks.28 output and the bank targets, K pairs of one caption."""
        k = x.shape[0]
        x = x.to(self.device, M.DTYPE)
        h_s = self.student_activation(x, self.emb(ctx, k))
        if not self._graph_checked:
            assert h_s.requires_grad and h_s.grad_fn is not None, \
                "the blocks.28 capture is detached: the gradient cannot reach the adapter"
            self._graph_checked = True
        h_t = target.to(self.device, M.DTYPE, non_blocking=True)
        reg_loss = F.mse_loss(h_s.float(), h_t.float())
        return data.REG_LOSS_WEIGHT * reg_loss, reg_loss.item()


def load_contexts(ctx_dir: str, cis, caps: list[str]) -> dict:
    """{ctx_index(ci): context} of the captions `cis`; each file must hold its caption's text."""
    out = {}
    for ci in cis:
        p = data.ctx_index(ci)
        f = os.path.join(ctx_dir, f"p{p}.pt")
        d = out[p] = M.load_ctx(f)
        assert d.get("prompt") == caps[ci], (
            f"{f} holds {d.get('prompt')!r}, caption {ci} is {caps[ci]!r}; pass --captions with the file "
            "the contexts were encoded from, or rerun python -m wan.encode_prompts --set train --force")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank-dir", default=None, help=f"default {C.target_dir()}")
    ap.add_argument("--ctx-dir", default=None, help=f"default {C.train_ctx_dir()}")
    ap.add_argument("--captions", default=None,
                    help=f"caption file the contexts were encoded from (default {data.TRAIN_CAPTIONS})")
    ap.add_argument("--out-dir", default=None, help=f"default {C.train_dir()}")
    ap.add_argument("--steps", type=int, default=data.STEPS, help=f"optimizer steps (default {data.STEPS})")
    ap.add_argument("--chunk", type=int, default=C.CHUNK,
                    help="noises of one caption per forward/backward (at world 8 a rank holds one)")
    ap.add_argument("--save-every", type=int, default=C.SAVE_EVERY)
    ap.add_argument("--log-every", type=int, default=C.LOG_EVERY)
    ap.add_argument("--seed", type=int, default=C.TRAIN_SEED)
    ap.add_argument("--overwrite", action="store_true", help="allow an output dir that already holds adapters")
    a = ap.parse_args()
    bdir = a.bank_dir or C.target_dir()
    ctx_dir = a.ctx_dir or C.train_ctx_dir()
    out = a.out_dir or C.train_dir()
    adapters = os.path.join(out, "adapters")

    rank, world = data.dist_init()
    if world > 1:
        torch.cuda.set_device(0)
    assert data.SAMPLES_PER_STEP % world == 0, f"world size {world} must divide {data.SAMPLES_PER_STEP}"
    is_main = rank == 0
    if os.path.isdir(adapters) and os.listdir(adapters) and not a.overwrite:
        sys.exit(f"{adapters} already holds {sorted(os.listdir(adapters))}; pass --overwrite or another --out-dir")
    os.makedirs(adapters, exist_ok=True)

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    cache = bank.RamCache(bdir, max_gib=min(400.0, 1400.0 / world))    # ranks share the host RAM
    man = cache.manifest
    sampler = data.StepSampler(a.seed)
    plan = list(sampler.steps(a.steps))
    need = sorted({(data.ctx_index(ci), ni) for st in plan for ci, ni in data.flatten(st)})
    absent = [pn for pn in need if pn[1] not in cache.index.get(pn[0], [])]
    assert not absent, f"the bank at {bdir} lacks {len(absent)} of the {len(need)} pairs this run draws, e.g. {absent[:3]}"
    if is_main:
        print(f"bank      : {bdir} ({cache.stats()}); teacher forward {man['teacher_step']} "
              f"(sigma {man['sigma_in']:.4f} -> {man['sigma_out']:.4f})\n"
              f"budget    : {a.steps} steps x {data.SAMPLES_PER_STEP} pairs over {len(need)} distinct pairs, "
              f"world {world}, chunk {a.chunk}", flush=True)

    student = Student("cuda:0", verbose=is_main)

    # ---- gates, before any LoRA exists ---------------------------------------------
    caps = data.captions(a.captions)
    p0 = data.ctx_index(0)
    ctx0 = load_contexts(ctx_dir, [0], caps)[p0]
    student.verify_truncation(ctx0)
    i0 = cache.index[p0][0]
    lat0, tgt0 = cache.batch(p0, [i0])
    assert torch.equal(lat0[0], seeds.wan_latent(seeds.prompt_seed(p0, i0), C.FRAMES)[0]), \
        "the bank's stored latent is not the seed rule's latent"
    residual = student.residual(ctx0, lat0.to("cuda:0", M.DTYPE), tgt0)
    targets = student.attach_lora()
    del lat0, tgt0
    torch.cuda.empty_cache()

    if world > 1:
        import torch.distributed as dist
        for p_ in student.lora_params:              # identical adapter init on every rank
            dist.broadcast(p_.data, src=0)
    opt = data.make_optimizer(student.lora_params)

    ctxs = load_contexts(ctx_dir, sorted({ci for st in plan for ci, _ in data.flatten(st)}), caps)
    meta = data.run_meta("wan", dict(
        backbone=C.BACKBONE, teacher=C.TEACHER_REPO, teacher_revision=C.TEACHER_REVISION,
        target_teacher=f"{C.TEACHER_REPO} + {C.CFGD_REPO} (LoRA scale 1.0, flow shift {C.CFGD_FLOW_SHIFT}, guidance 1)",
        student=C.STUDENT_REPO, student_revision=C.STUDENT_REVISION, gate="step1_only",
        frames=C.FRAMES, latent_shape=list(C.LATENT_SHAPE), tokens=C.TOKENS, train_timestep=C.TRAIN_TIMESTEP,
        site=f"blocks.{C.ANCHOR_BLOCK}", teacher_target_step=C.FT_TARGET_FORWARD,
        bank_manifest=man, lora_rank=C.LORA_RANK, lora_alpha=C.LORA_ALPHA,
        lora_blocks=f"0..{C.LORA_LAST_BLOCK}", lora_modules=len(targets), lora_module_list=targets,
        world_size=world, chunk=a.chunk, steps=a.steps, seed=a.seed, residual_pre_lora=residual,
        loss=f"{data.REG_LOSS_WEIGHT} * MSE(h_student_step1, h_teacher_forward{C.FT_TARGET_FORWARD}) at "
             f"blocks[{C.ANCHOR_BLOCK}], mean over the {data.SAMPLES_PER_STEP} pairs of a step"))
    if is_main:
        data.write_run_meta(os.path.join(out, "config.json"), meta)

    def save(step: int):
        if is_main:
            path = os.path.join(adapters, f"rediverse_ft_{step:07d}.pth")
            torch.save(dict(lora=student.lora_state_dict(), step=step, meta=data.portable(meta)), path)
            print(f"[save] {path}", flush=True)

    t_start = time.time()
    n_step = data.SAMPLES_PER_STEP
    for step, step_batches in enumerate(plan, 1):
        opt.zero_grad(set_to_none=True)
        log = step % a.log_every == 0 or step == 1
        acc, t0, n_mine = 0.0, time.time(), 0
        mine = data.shard_for_rank(step_batches, rank, world)
        for ci, nis in data.group_by_caption(mine):
            p = data.ctx_index(ci)
            for j in range(0, len(nis), a.chunk):
                idxs = nis[j:j + a.chunk]
                lats, tgts = cache.batch(p, idxs)
                loss, reg = student.loss(ctxs[p], lats, tgts)
                (loss * (len(idxs) * world / n_step)).backward()
                acc += reg * len(idxs)
                n_mine += len(idxs)
                del lats, tgts, loss
        assert n_mine == n_step // world, (n_mine, n_step, world)
        data.dist_allreduce_grads(student.lora_params, world)
        gnorm, n_bad = data.clip_and_step(student.lora_params, opt)
        dt = time.time() - t0
        if log:
            reg = data.dist_allreduce_scalar(acc / n_mine, world)
            if is_main:
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"iter {step:4d}/{a.steps}  reg_loss {reg:.6g}  |g| {gnorm:.3g}"
                      f"{'  NONFINITE ' + str(n_bad) if n_bad else ''}  {dt:.1f}s/step  peak {peak:.1f} GiB"
                      f"  eta {(a.steps - step) * dt / 60.0:.0f}m  bank {cache.hits}h/{cache.misses}m  world {world}",
                      flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save(step)
    data.dist_barrier(world)
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    if is_main:
        print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
