#!/usr/bin/env python
"""ReDiverse-FT training on MiniMax-H3 (paper Sec. 4.1, App. D).

The student is Turbo (teacher + rank-128 LoRA). ReDiverse-FT is a rank-32 LoRA on the attention
and feed-forward projections of transformer_blocks 0..45, trained so that the student's block-45
video rows at its first forward match the teacher's at forward tau = 16 from the same latent:

    loss = 0.5 * MSE(h_student, h_teacher) / Var(h_teacher)

with Var taken over the target of each forward (--micro-chunk latents, 1 in the recipe, so every
latent is normalised by its own target). Targets come from the bank of h3.build_targets (the
teacher is never loaded here). One optimizer step = 4 micro-batches, each holding 2 latents of one
training prompt; prompts are visited round-robin with a rotating latent window; 200 steps, AdamW
1e-4, gradient clipping at 10.

Memory: the 124-frame sequence does not fit one 80 GB GPU with gradients, so Turbo is fused into
the base weights (exact: it is frozen), blocks 0..22 stay on --device, blocks 23..45 move to
--dev1 and blocks 46..49 are dropped (the loss reads block 45, and the forward stops there).

  python -m h3.train --device cuda:0 --dev1 cuda:1

Checkpoints go to <outputs>/train/adapters/rediverse_ft_{iter:07d}.pth (every 50 steps and the
last); h3.convert_weights turns one into the adapter file that h3.generate --adapter reads. A run
directory that already holds checkpoints needs --overwrite.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

import torch

from h3 import config as C
from h3 import model as H
from h3.split2 import forward_to_block, split_model
from rediverse import data


@dataclass
class TrainConfig:
    anchor_block: int = C.ANCHOR_BLOCK
    lora_last_block: int = C.ANCHOR_BLOCK
    reg_loss_weight: float = C.REG_LOSS_WEIGHT
    loss_norm: str = "target_var"              # MSE / Var(target) of each forward chunk
    teacher_target_step: int = C.TAU
    lora_rank: int = C.LORA_RANK
    lora_alpha: int = C.LORA_ALPHA
    samples_per_prompt: int = C.SAMPLES_PER_PROMPT
    micro_chunk: int = C.MICRO_CHUNK           # latents per forward/backward and per Var normaliser
    gradient_checkpointing: bool = True
    lr: float = data.LR
    weight_decay: float = data.WEIGHT_DECAY
    betas: tuple = data.BETAS
    eps: float = data.EPS
    grad_clip: float = data.GRAD_CLIP
    batch_size_global: int = C.CAPTIONS_PER_STEP
    max_iter: int = C.MAX_ITER
    save_ckpt_iter: int = C.SAVE_EVERY
    logging_iter: int = C.LOG_EVERY
    seed: int = C.TRAIN_SEED
    dev1: str = "cuda:1"
    split_cut: int = C.SPLIT_CUT
    bank_dir: str = ""
    run_dir: str = field(default_factory=lambda: C.output_dir("train"))


class _Stop(Exception):
    pass


def prepare_student_for_training(student, cfg, dev0):
    """Fuse Turbo into the weights, then blocks [cut, 46) to dev1 and drop blocks 46..49."""
    student.fuse_lora(lora_scale=1.0)
    student.unload_lora()
    dropped = split_model(student, dev0, cfg.dev1, cfg.split_cut, cfg.anchor_block + 1)
    print(f"[h3] Turbo fused; blocks 0..{cfg.split_cut-1} on {dev0}, "
          f"{cfg.split_cut}..{cfg.anchor_block} on {cfg.dev1}; dropped {dropped}", flush=True)
    return dict(mode="fused_split", dropped=dropped, cut=cfg.split_cut)


def lora_target_modules(model, cfg) -> list[str]:
    """attn.to_q/k/v/out.0 and ff.net.0.proj/ff.net.2 of transformer_blocks 0..lora_last_block."""
    pat = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
    names = []
    for n, m in model.named_modules():
        base = getattr(m, "base_layer", None)
        lin = base if base is not None else m
        if not isinstance(lin, torch.nn.Linear):
            continue
        mm = pat.match(n)
        if mm and int(mm.group(1)) <= cfg.lora_last_block and mm.group(2) in C.LORA_SUFFIXES:
            names.append(n)
    expect = (cfg.lora_last_block + 1) * len(C.LORA_SUFFIXES)
    assert len(names) == expect, f"{len(names)} targets, expected {expect}"
    return names


def attach_rediverse_ft(student, cfg) -> dict:
    """The ReDiverse-FT LoRA (B = 0 at init), trainable and kept in float32; every other
    parameter is frozen. Turbo is already fused, so this is the only adapter."""
    from peft import LoraConfig
    assert not getattr(student, "peft_config", None), "fuse the Turbo LoRA before attaching ReDiverse-FT"
    targets = lora_target_modules(student, cfg)
    student.add_adapter(LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                   target_modules=targets, init_lora_weights=True),
                        adapter_name=C.ADAPTER_NAME)
    active = set(student.active_adapters())
    n_train = 0
    for n, p in student.named_parameters():
        is_ft = f".{C.ADAPTER_NAME}." in n
        p.requires_grad_(is_ft)
        if is_ft:
            if p.dtype != torch.float32:
                p.data = p.data.float()
            n_train += p.numel()
    assert n_train > 0, "no trainable parameters"
    print(f"[h3] ReDiverse-FT LoRA r={cfg.lora_rank} on {len(targets)} modules "
          f"({n_train/1e6:.1f}M trainable); active adapters {sorted(active)}", flush=True)
    return dict(n_target_modules=len(targets), n_trainable=n_train)


class ActivationRegression:
    """Student block-45 video rows at forward 1 (the forward stops at block 45) and the loss."""

    def __init__(self, student, cfg, device):
        assert cfg.loss_norm == "target_var", cfg.loss_norm
        self.s, self.cfg, self.dev = student, cfg, device
        self.cap = {}
        blk = student.transformer_blocks[cfg.anchor_block]

        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            self.cap["h"] = o[:, -H.NUM_VIDEO_ROWS:, :]
            # Stops the forward after block 45. Under non-reentrant gradient checkpointing the
            # recomputation in backward ends before this hook would fire again.
            if self.cap.get("stop"):
                raise _Stop
            return None
        blk.register_forward_hook(hook)
        if cfg.gradient_checkpointing:
            student.enable_gradient_checkpointing()
        self._sched = {}

    def _plan(self, ctx, device):
        key = id(ctx)
        if key not in self._sched:
            self._sched[key] = (H.dev_ctx(ctx, device), H.Schedule(C.STEPS["student"], ctx, device))
        return self._sched[key]

    def student_activation(self, x, ctx):
        d, sched = self._plan(ctx, self.dev)
        rows = H.pack(x.to(self.dev, H.DTYPE))
        audio = d["audio_rows"].unsqueeze(0).expand(rows.shape[0], -1, -1)
        self.cap.clear(); self.cap["stop"] = True
        uniq, idx = sched.plan[0]
        try:
            forward_to_block(
                self.s, hidden_states=rows, audio_hidden_states=audio,
                encoder_hidden_states=d["prompt_embeds"].expand(rows.shape[0], -1, -1),
                timestep=uniq, timestep_indices=idx, token_tags=d["token_tags"],
                position_ids=d["position_ids"], video_indices=d["video_indices"],
                audio_indices=d["audio_indices"], text_indices=d["text_indices"])
        except _Stop:
            pass
        assert "h" in self.cap, f"the block-{self.cfg.anchor_block} hook never fired"
        return self.cap["h"]

    def step(self, x, ctx, tgt, backward_scale=1.0) -> dict:
        """x [K, 24, 37, 48, 84], tgt [K, 37296, 5376] from the bank; micro_chunk latents per
        forward/backward, gradients accumulated."""
        cfg = self.cfg
        K, c = x.shape[0], cfg.micro_chunk
        assert K % c == 0
        n = K // c
        acc = {}
        for j in range(n):
            h = self.student_activation(x[j*c:(j+1)*c], ctx)
            t = tgt[j*c:(j+1)*c].to(h.device)
            assert h.shape == t.shape, (tuple(h.shape), tuple(t.shape))
            # Block-45 activations are O(1e3) per element; dividing by the target variance (a
            # per-chunk constant; one latent at --micro-chunk 1) puts the loss on the scale of rel^2.
            raw = torch.nn.functional.mse_loss(h.float(), t.float())
            reg = raw / t.float().var().clamp_min(1e-6)
            (cfg.reg_loss_weight * reg * backward_scale / n).backward()
            with torch.no_grad():
                d = dict(reg_loss=reg, raw_mse=raw, total_loss=cfg.reg_loss_weight * reg,
                         rel=(h.float()-t.float()).pow(2).mean().sqrt() / t.float().std(),
                         h_std_student=h.float().std(), h_std_teacher=t.float().std())
            for k, v in d.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach()) / n
            del h, t, reg
        return acc


class Bank:
    """Target bank entries {bank_dir}/p{idx:03d}_s{i}.pt, read from disk on use."""

    def __init__(self, bank_dir, K_needed):
        self.files = sorted(glob.glob(os.path.join(bank_dir, "p*_s*.pt")))
        assert self.files, f"empty bank {bank_dir}"
        self.by_prompt = {}
        for f in self.files:
            m = re.search(r"p(\d+)_s(\d+)\.pt$", f)
            self.by_prompt.setdefault(int(m.group(1)), []).append((int(m.group(2)), f))
        self.prompts = sorted(self.by_prompt)
        for p in self.prompts:
            self.by_prompt[p].sort()
        self.K = K_needed
        self.i = 0

    def __len__(self):
        return sum(len(v) for v in self.by_prompt.values())

    def next(self):
        """K latents + targets of one prompt; prompts round-robin, latent window rotating by K
        per pass."""
        p = self.prompts[self.i % len(self.prompts)]
        rot = (self.i // len(self.prompts)) * self.K
        self.i += 1
        entries = self.by_prompt[p]
        pick = [entries[(rot + j) % len(entries)] for j in range(self.K)]
        xs, ts = [], []
        for s, f in pick:
            d = torch.load(f, map_location="cpu")
            xs.append(d["latent"]); ts.append(d["target"])
        return p, torch.stack(xs), torch.stack(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=C.TAU, help="teacher forward of the targets (bank b45_k<k>)")
    ap.add_argument("--device", default="cuda:0", help="GPU of the embedders and blocks 0..22")
    ap.add_argument("--dev1", default="cuda:1", help="GPU of blocks 23..45")
    ap.add_argument("--split-cut", type=int, default=C.SPLIT_CUT)
    ap.add_argument("--samples-per-prompt", type=int, default=C.SAMPLES_PER_PROMPT)
    ap.add_argument("--micro-chunk", type=int, default=C.MICRO_CHUNK,
                    help="latents per forward/backward; the Var normaliser is taken per chunk, "
                         "so 1 reproduces the adapter")
    ap.add_argument("--steps", type=int, default=C.MAX_ITER - 1, help=f"optimizer steps (default {C.MAX_ITER - 1})")
    ap.add_argument("--bank-dir", default="", help="target bank (default <data>/targets/b45_k<k>)")
    ap.add_argument("--ctx-dir", default="", help="training context directory (default <data>/ctx/train)")
    ap.add_argument("--out-dir", default="", help="run directory (default <outputs>/train)")
    ap.add_argument("--overwrite", action="store_true", help="allow an output dir that already holds adapters")
    ap.add_argument("--dry-run", type=int, default=0,
                    help=f"N steps of one micro-batch each, at its 1/{C.CAPTIONS_PER_STEP} weight in a full step; "
                         "logged to dry_run.jsonl, no checkpoint (memory and speed check)")
    a = ap.parse_args()

    cfg = TrainConfig(teacher_target_step=a.k, samples_per_prompt=a.samples_per_prompt,
                       micro_chunk=a.micro_chunk, max_iter=a.steps + 1, dev1=a.dev1, split_cut=a.split_cut)
    cfg.bank_dir = a.bank_dir or C.target_dir(a.k)
    cfg.run_dir = a.out_dir or C.output_dir("train")
    ctx_dir = a.ctx_dir or C.train_ctx_dir()
    adapters = os.path.join(cfg.run_dir, "adapters")
    if not a.dry_run and os.path.isdir(adapters) and os.listdir(adapters) and not a.overwrite:
        sys.exit(f"{adapters} already holds {sorted(os.listdir(adapters))}; pass --overwrite or another --out-dir")
    os.makedirs(cfg.run_dir if a.dry_run else adapters, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dev = a.device

    bank = Bank(cfg.bank_dir, cfg.samples_per_prompt)
    want = [(p, i) for p in C.ft_prompt_ids() for i in range(C.FT_LATENTS_PER_PROMPT)]
    have = {(p, s) for p in bank.prompts for s, _ in bank.by_prompt[p]}
    absent, extra = [w for w in want if w not in have], sorted(have - set(want))
    assert not absent, (f"the bank at {cfg.bank_dir} lacks {len(absent)} of its {len(want)} entries, e.g. "
                        f"{absent[:3]}; run python -m h3.build_targets to complete it")
    assert not extra, f"the bank at {cfg.bank_dir} holds {len(extra)} entries outside the recipe, e.g. {extra[:3]}"
    with open(os.path.join(cfg.bank_dir, "manifest.json")) as f:
        man = json.load(f)
    assert man["k"] == a.k and man["block"] == C.ANCHOR_BLOCK, man
    student = H.load_arm("student", dev)
    info = prepare_student_for_training(student, cfg, dev)
    info.update(attach_rediverse_ft(student, cfg))
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                            weight_decay=cfg.weight_decay)
    tr = ActivationRegression(student, cfg, dev)

    loss_desc = (f"{cfg.reg_loss_weight} * MSE(student blocks[{C.ANCHOR_BLOCK}] video rows at step 1, "
                 f"teacher's at forward {a.k}) / Var(teacher's) per forward of {cfg.micro_chunk} latent(s), "
                 f"{cfg.samples_per_prompt} latents of one prompt per micro-batch, "
                 f"{cfg.batch_size_global} micro-batches per step")
    if not a.dry_run:
        meta = dict(backbone=C.BACKBONE, teacher=f"{C.TEACHER_REPO}@{C.TEACHER_REVISION}",
                    student=f"{C.STUDENT_REPO}@{C.STUDENT_REVISION}/{C.STUDENT_FILE}",
                    cfg=vars(cfg), lora=info, bank=man, n_bank=len(bank), loss=loss_desc)
        data.write_run_meta(os.path.join(cfg.run_dir, "config.json"), meta)
    print(f"[train] {loss_desc}", flush=True)
    print(f"[train] bank {len(bank)} entries over {len(bank.prompts)} prompts; "
          f"K={cfg.samples_per_prompt} x {cfg.batch_size_global} captions/step, "
          f"{cfg.max_iter-1} steps; sigma_in {man['sigma_in']:.4f} -> {man['sigma_out']:.4f}",
          flush=True)

    ctxs = {p: H.load_ctx(os.path.join(ctx_dir, f"p{p:03d}.pt")) for p in bank.prompts}
    n_iter = a.dry_run or a.steps
    log_path = os.path.join(cfg.run_dir, "dry_run.jsonl" if a.dry_run else "train_log.jsonl")
    open(log_path, "w").close()
    B = cfg.batch_size_global
    t0 = time.time()
    for it in range(1, n_iter + 1):
        opt.zero_grad(set_to_none=True)
        acc = {}
        for _ in range(1 if a.dry_run else B):
            p, x, tgt = bank.next()
            lm = tr.step(x, ctxs[p], tgt, backward_scale=1.0 / B)
            for k, v in lm.items():
                acc[k] = acc.get(k, 0.0) + v / B
        for q in params:
            if q.grad is not None and not q.grad.isfinite().all():
                torch.nan_to_num(q.grad, nan=0.0, out=q.grad)
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        opt.step()
        el = time.time() - t0
        row = dict(it=it, dry_run=bool(a.dry_run), grad_norm=gnorm, **acc)
        if a.dry_run:
            row["scale"] = 1.0 / B
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if a.dry_run:
            print(f"dry {it}/{a.dry_run} reg/{B} {acc['reg_loss']:.4f} |g|/{B} {gnorm:.3f} "
                  f"rel/{B} {acc['rel']:.3f} peak {torch.cuda.max_memory_allocated(dev)/2**30:.1f}G "
                  f"{el/it:.0f}s/micro-batch", flush=True)
            continue
        if it % cfg.logging_iter == 0 or it == 1:
            print(f"it {it:4d} reg_loss {acc['reg_loss']:8.4f} |g| {gnorm:7.3f} rel {acc['rel']:.4f} "
                  f"h_std s/t {acc['h_std_student']:.3f}/{acc['h_std_teacher']:.3f} "
                  f"[{el/it:.0f}s/it eta {(n_iter-it)*el/it/60:.0f}m "
                  f"peak {torch.cuda.max_memory_allocated(dev)/2**30:.1f}G]", flush=True)
        if it % cfg.save_ckpt_iter == 0 or it == n_iter:
            sd = {n: p.detach().cpu() for n, p in student.named_parameters() if p.requires_grad}
            f = os.path.join(cfg.run_dir, "adapters", f"rediverse_ft_{it:07d}.pth")
            torch.save(sd, f + ".tmp"); os.replace(f + ".tmp", f)
            print(f"[train] saved {f}", flush=True)
    print(f"[train] {'dry ' if a.dry_run else ''}done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
