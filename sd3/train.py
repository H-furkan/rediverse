"""Train the SD3 ReDiverse-FT adapter (paper Sec. 4.1, App. D).

    python -m sd3.train                       # 200 optimizer steps on one 80 GB GPU (about 1 h)

The recipe has no options: the 100 training captions (prompts/train_captions.txt, embedded by
sd3.encode_prompts), 16 paired noises per caption, 4 captions x 8 noises per optimizer step as
four gradient-accumulation micro-batches of 8, 200 steps (every pair used exactly 4 times),
plain MSE between the student's step-1 block-12 image stream and the teacher's at forward 9 of
its CFG-7 trajectory, LoRA rank 32 / alpha 32 on blocks 0..12, AdamW 1e-4 (weight decay 0.01,
constant, no warm-up), gradients nan_to_num'd and clipped to norm 10, seed 0.

Writes <out-dir>/config.json, timing.json and adapters/rediverse_ft_{0000100,0000200}.pth (the
adapter tensors only, fp32). sd3.convert_weights turns the last one into the released
safetensors file; sd3.generate also reads the .pth directly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

from rediverse import data as D
from sd3.config import DTYPE, LATENT_SHAPE, ctx_dir, output_dir, uncond_path
from sd3.model import SD3ReDiverseFT, premise

SEED = 0
SAVE_EVERY = 100
LOG_EVERY = 5


def load_ctx(ctx: str, captions: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(prompt_embeds [1,333,4096], pooled [1,2048]) of caption ci from <ctx>/p{ci:06d}.pt,
    verified against the caption text."""
    out = []
    for ci, text in enumerate(captions):
        f = os.path.join(ctx, f"p{ci:06d}.pt")
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found; run: python -m sd3.encode_prompts")
        d = torch.load(f, map_location="cpu")
        assert d.get("prompt", "") == text, f"{f} holds {d.get('prompt')!r}, caption {ci} is {text!r}"
        out.append((d["prompt_embeds"], d["pooled_projections"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None, help="default <REDIVERSE_OUT>/sd3/train")
    ap.add_argument("--ctx-dir", default=None, help="default <REDIVERSE_DATA>/sd3/ctx")
    ap.add_argument("--uncond", default=None, help="default <REDIVERSE_DATA>/sd3/uncond.pt")
    ap.add_argument("--captions", default=None, help="default prompts/train_captions.txt")
    ap.add_argument("--steps", type=int, default=D.STEPS, help="optimizer steps (default 200)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing a finished run")
    a = ap.parse_args()

    run_dir = a.out_dir or output_dir("train")
    ctx = a.ctx_dir or ctx_dir()
    unc = a.uncond or uncond_path()
    if not os.path.exists(unc):
        raise FileNotFoundError(f"{unc} not found; run: python -m sd3.encode_prompts")
    max_iter = a.steps + 1
    ckpt_dir = os.path.join(run_dir, "adapters")
    final = os.path.join(ckpt_dir, f"rediverse_ft_{max_iter - 1:07d}.pth")
    assert a.overwrite or not os.path.exists(final), f"{final} exists; pass --overwrite to replace it"
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.manual_seed(SEED)
    random.seed(SEED)

    model = SD3ReDiverseFT(unc, device=a.device)
    params = model.adapter_parameters()
    opt = D.make_optimizer(params)
    texts = D.captions(a.captions)
    ctx_cache = load_ctx(ctx, texts)
    sampler = D.StepSampler(seed=SEED)
    print(f"[train] {len(texts)} captions verified against {ctx}; {sampler.caps_per_step} captions x "
          f"{sampler.noises_per_draw} noises per step, seed rule {D.SEED0} + (10000+ci)*100003 + ni",
          flush=True)

    meta = model.config_dict()
    meta.update(steps=a.steps, seed=SEED, save_every=SAVE_EVERY,
                optimizer="AdamW", lr=D.LR, weight_decay=D.WEIGHT_DECAY, betas=list(D.BETAS), eps=D.EPS,
                grad_clip=D.GRAD_CLIP, lr_schedule="constant, no warm-up",
                micro_batches_per_step=D.CAPTIONS_PER_STEP,
                loss=(f"MSE(h_img_student_step1[i], h_img_teacher_forward{model.k_teacher}[i]) averaged "
                      f"over the micro-batch; teacher trajectory guided with CFG {model.teacher_guidance} "
                      f"on forwards 1..{model.k_teacher - 1}"))
    meta.update(D.run_meta("sd3", dict(ctx_dir=ctx, uncond_ctx=unc, run_dir=run_dir)))
    D.write_run_meta(os.path.join(run_dir, "config.json"), meta)

    print(json.dumps(premise(), indent=2), flush=True)
    print(f"[train] run_dir={run_dir}", flush=True)
    print(f"[train] {sampler.noises_per_draw} x {sampler.caps_per_step} captions/step = "
          f"{D.SAMPLES_PER_STEP} samples/step, {a.steps} steps", flush=True)
    print(f"[train] trainable {sum(p.numel() for p in params) / 1e6:.1f}M in {len(params)} tensors on "
          f"{meta['lora_info']['n_target_modules']} of {meta['lora_info']['n_target_modules_unpruned']} "
          f"modules", flush=True)
    print(f"[train] loss = {meta['loss']}", flush=True)

    t0 = time.time()
    for it in range(1, max_iter):
        opt.zero_grad(set_to_none=True)
        acc = {}
        step_plan = sampler.step()
        for ci, nis in step_plan:
            pe, pooled = (v.to(a.device, DTYPE) for v in ctx_cache[ci])
            x = D.noise_batch([(ci, ni) for ni in nis], LATENT_SHAPE).to(a.device, DTYPE)
            lm = model.step(x, pe, pooled)
            (lm["total_loss"] / D.CAPTIONS_PER_STEP).backward()
            for k, v in lm.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach()) / D.CAPTIONS_PER_STEP

        gnorm, n_bad = D.clip_and_step(params, opt)

        if it % LOG_EVERY == 0 or it == 1:
            print(f"it {it:5d}  reg_loss {acc['reg_loss']:9.4f}  total {acc['total_loss']:9.4f}  "
                  f"|g| {gnorm:8.4f}  h_std s/t {acc['h_std_student']:.3f}/{acc['h_std_teacher']:.3f}  "
                  f"spread_ratio {acc['spread_ratio']:.4f}"
                  f"{'  NONFINITE ' + str(n_bad) if n_bad else ''}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
        if it % SAVE_EVERY == 0 or it == max_iter - 1:
            path = os.path.join(ckpt_dir, f"rediverse_ft_{it:07d}.pth")
            torch.save(dict(state_dict=model.adapter_state_dict(), iteration=it,
                            config=D.portable(meta)), path)
            print(f"[train] saved {path} ({os.path.getsize(path) / 2**20:.0f} MB)", flush=True)

    torch.cuda.synchronize()
    tm = model.timing()
    tm["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    tm["wall_min"] = (time.time() - t0) / 60
    with open(os.path.join(run_dir, "timing.json"), "w") as f:
        json.dump(tm, f, indent=2)
    print(f"[train] per micro-batch: student {tm['s_student'] * 1e3:.0f} ms, teacher "
          f"{tm['s_teacher'] * 1e3:.0f} ms ({tm['teacher_forwards_per_micro']:.0f} forwards); "
          f"peak GPU memory {tm['peak_mem_gib']:.2f} GiB", flush=True)
    print(f"[train] done in {tm['wall_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
