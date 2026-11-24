#!/usr/bin/env python
"""Qwen-Image-Lightning ReDiverse: diverse samples from Qwen-Image-Lightning (8 steps, diffusers only).

  # adapter + steering (default; estimates the prompt's teacher directions with the Qwen-Image base once, under a minute on an H100)
  python generate_qwen_image_lightning.py --prompt "a photo of a dog" --seeds 0-7 --out out --compare
  # adapter only
  python generate_qwen_image_lightning.py --prompt "a photo of a dog" --seeds 0-7 --out out --mode ft --compare
  # the paper's steering strength
  python generate_qwen_image_lightning.py --prompt "a photo of a dog" --seeds 0-7 --out out --band 2

Writes out/rediverse/<seed>.png and, with --compare, seed-matched out/lightning/<seed>.png from Qwen-Image-Lightning alone.

What it does (ReDiverse, arXiv:2611.18550):
  The student is Qwen/Qwen-Image with the lightx2v Qwen-Image-Lightning 8-step V2.0 LoRA (8 steps, shift 3, no CFG, as on
  its model card). The teacher is Qwen/Qwen-Image itself (50 steps, true CFG 4), i.e. the same weights with the Lightning
  LoRA turned off, so one copy of the model serves both.
  ReDiverse-FT  a rank-32 LoRA on transformer_blocks 0..57 (of 60), trained to match the teacher's block-57 image-stream
                activations at its 4th sampling step; active in the first of the 8 sampling steps only.
  ReDiverse-TF  for each prompt, the top-8 principal directions of the teacher's block-57 image activations over 25 noises
                (conditional, at its 4th step); every sample adds sum_k alpha_k s_k u_k to block 57 in the first step,
                alpha_k ~ U[-band, band] (default band 4; the paper uses 2).
Images are 1024x1024 (the adapter and the directions are tied to this resolution).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np
import torch

BASE, BASE_REV = "Qwen/Qwen-Image", "75e0b4be04f60ec59a75f475837eced720f823b6"
LIGHTNING_REPO, LIGHTNING_REV = "lightx2v/Qwen-Image-Lightning", "d468c534b360e0754c6bf9e38ed49e1e09cd2e2a"
LIGHTNING_FILE = "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
ADAPTER_REPO, ADAPTER_FILE = "ReDiverse/Qwen-Image-Lightning-ReDiverse", "qwen-image-lightning-rediverse.safetensors"
BLOCK, TAU, N_ROLL, K, RES, STEPS = 57, 4, 25, 8, 1024, 8
TEACHER_STEPS = 50
N_IMG = (RES // 16) ** 2                      # 4096 image tokens at 1024^2 (VAE /8, patch 2)
D_MODEL = 3072


def parse_seeds(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        if "-" in tok:
            lo, hi = tok.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif tok:
            out.append(int(tok))
    return out


def packed_noise(seed: int) -> torch.Tensor:
    """[1, 4096, 64] standard normal on a CPU generator (packed latent layout of QwenImagePipeline)."""
    return torch.randn((1, N_IMG, 64), generator=torch.Generator("cpu").manual_seed(seed))


def lightning_scheduler(config):
    """Qwen-Image-Lightning's scheduler (model card): exponential shift with mu = log 3, no terminal stretch."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_config(config, base_shift=math.log(3), max_shift=math.log(3),
                                                       shift_terminal=None, use_dynamic_shifting=True)


def teacher_schedule(config) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen-Image's own 50-step schedule at 1024^2 (base scheduler config, resolution-dependent mu), as QwenImagePipeline."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    sch = FlowMatchEulerDiscreteScheduler.from_config(config)
    c = sch.config
    m = (c.max_shift - c.base_shift) / (c.max_image_seq_len - c.base_image_seq_len)
    mu = N_IMG * m + (c.base_shift - m * c.base_image_seq_len)
    sch.set_timesteps(sigmas=np.linspace(1.0, 1 / TEACHER_STEPS, TEACHER_STEPS), mu=mu)
    return sch.sigmas.clone(), sch.timesteps.clone()


@torch.no_grad()
def estimate_directions(pipe, prompt: str, device, cache_dir: str, seed_base: int = 100) -> dict:
    """Top-K principal directions of the teacher's (Qwen-Image, LoRAs off) block-57 image activations at its forward 4
    (25 conditional rollouts from noises seed_base .. seed_base + 24)."""
    key = hashlib.sha1(f"{prompt}|{seed_base}".encode()).hexdigest()[:16]
    f = os.path.join(cache_dir, f"directions_{key}.pt")
    if os.path.exists(f):
        d = torch.load(f, map_location="cpu")
        if d.get("prompt") == prompt and d.get("seed_base") == seed_base:
            return d
    print("[rediverse] estimating teacher directions with the Qwen-Image base (25 rollouts) ...", flush=True)
    tr = pipe.transformer
    pipe.disable_lora()                            # the teacher is the base model
    pe, pm = pipe.encode_prompt(prompt=[prompt], device=device)
    sg, ts = teacher_schedule(pipe.scheduler.config)
    shapes = [[(1, RES // 16, RES // 16)]]
    grab = {}

    class _Stop(Exception):
        pass

    def hook(_m, _i, out):
        grab["h"] = out[1].detach().float().cpu().reshape(-1)
        raise _Stop
    X = torch.empty(N_ROLL, N_IMG * D_MODEL)
    for j in range(N_ROLL):
        x = packed_noise(seed_base + j).to(device, torch.bfloat16)
        for i in range(TAU):
            t = ts[i].to(device).expand(1).to(torch.bfloat16) / 1000
            kw = dict(hidden_states=x, encoder_hidden_states=pe, encoder_hidden_states_mask=pm, timestep=t,
                      img_shapes=shapes, return_dict=False)
            if i < TAU - 1:                       # conditional teacher steps (Euler, as FlowMatchEulerDiscreteScheduler)
                v = tr(**kw)[0]
                x = (x.float() + (sg[i + 1] - sg[i]).to(device) * v).to(torch.bfloat16)
            else:
                h = tr.transformer_blocks[BLOCK].register_forward_hook(hook)
                try:
                    tr(**kw)
                except _Stop:
                    pass
                finally:
                    h.remove()
        X[j] = grab["h"]
    pipe.enable_lora()
    Xc = X - X.mean(0, keepdim=True)
    evals, evecs = torch.linalg.eigh((Xc @ Xc.T).double())
    evals, evecs = evals.flip(0), evecs.flip(1)
    U = torch.stack([(evecs[:, k].float() @ Xc) / (evecs[:, k].float() @ Xc).norm() for k in range(K)]).to(torch.bfloat16)
    s = torch.tensor([float((evals[k].clamp_min(0) ** 0.5 * evecs[:, k]).std(unbiased=False)) for k in range(K)])
    d = dict(prompt=prompt, U=U, s=s, block=BLOCK, tau=TAU, res=RES, seed_base=seed_base)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(d, f)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--out", default="out")
    ap.add_argument("--mode", choices=("rediverse", "ft"), default="rediverse",
                    help="rediverse = adapter + steering; ft = adapter only")
    ap.add_argument("--band", type=float, default=4.0, help="steering strength: alpha ~ U[-band, band] (paper: 2)")
    ap.add_argument("--base", default=BASE, help="Qwen-Image repo id or a local diffusers folder")
    ap.add_argument("--adapter", default=None, help="local qwen-image-lightning-rediverse.safetensors (default: the Hub)")
    ap.add_argument("--lightning", default=None, help=f"local {LIGHTNING_FILE} (default: the Hub)")
    ap.add_argument("--compare", action="store_true", help="also render Qwen-Image-Lightning alone from the same noises")
    ap.add_argument("--cache", default=".rediverse_cache")
    ap.add_argument("--device", default="cuda")
    # reproduction of the paper-style evaluation seeds (noise seed = seed, alpha seed = alpha_seed_base + seed)
    ap.add_argument("--alpha-seed-base", type=int, default=7_000_000)
    ap.add_argument("--direction-seed-base", type=int, default=100)
    a = ap.parse_args()
    from diffusers import QwenImagePipeline
    dev = a.device
    kw = dict(revision=BASE_REV) if a.base == BASE else {}
    pipe = QwenImagePipeline.from_pretrained(a.base, torch_dtype=torch.bfloat16, **kw).to(dev)
    teacher_config = dict(pipe.scheduler.config)
    if a.lightning:
        from safetensors.torch import load_file
        pipe.load_lora_weights(load_file(a.lightning), adapter_name="lightning")
    else:
        pipe.load_lora_weights(LIGHTNING_REPO, revision=LIGHTNING_REV, weight_name=LIGHTNING_FILE, adapter_name="lightning")
    pipe.set_adapters(["lightning"], adapter_weights=[1.0])
    pipe.scheduler = lightning_scheduler(teacher_config)
    seeds = parse_seeds(a.seeds)
    os.makedirs(os.path.join(a.out, "rediverse"), exist_ok=True)

    def render(latents, cb=None):
        return pipe(prompt=a.prompt, height=RES, width=RES, num_inference_steps=STEPS, true_cfg_scale=1.0,
                    latents=latents.to(dev, torch.bfloat16), callback_on_step_end=cb).images[0]

    if a.compare:                                  # Lightning alone first, before the ReDiverse adapter or hook exist
        os.makedirs(os.path.join(a.out, "lightning"), exist_ok=True)
        for sd in seeds:
            render(packed_noise(sd)).save(os.path.join(a.out, "lightning", f"{sd}.png"))

    tf = None
    if a.mode == "rediverse":
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(teacher_config)
        d = estimate_directions(pipe, a.prompt, dev, a.cache, a.direction_seed_base)
        pipe.scheduler = lightning_scheduler(teacher_config)
        tf = dict(U=d["U"].to(dev), s=d["s"].to(dev, torch.float32), alpha=None, step=0)

        def tf_hook(_m, _i, out):                  # block outputs are (text stream, image stream)
            if tf["step"] != 0 or tf["alpha"] is None:
                return None
            delta = ((tf["alpha"] * tf["s"]).to(tf["U"].dtype) @ tf["U"]).reshape(1, N_IMG, D_MODEL)
            return out[0], out[1] + delta.to(out[1].dtype)
        pipe.transformer.transformer_blocks[BLOCK].register_forward_hook(tf_hook)

    if a.adapter:
        from safetensors.torch import load_file
        pipe.load_lora_weights(load_file(a.adapter), adapter_name="rediverse")
    else:
        pipe.load_lora_weights(ADAPTER_REPO, weight_name=ADAPTER_FILE, adapter_name="rediverse")

    def step_cb(p, i, t, kw):                      # ReDiverse acts in the first step only
        if i == 0:
            p.set_adapters(["lightning"], adapter_weights=[1.0])
            if tf is not None:
                tf["step"] = 1
        return kw

    meta = {}
    for sd in seeds:
        pipe.set_adapters(["lightning", "rediverse"], adapter_weights=[1.0, 1.0])
        if tf is not None:
            g = torch.Generator("cpu").manual_seed(a.alpha_seed_base + sd)
            tf["alpha"] = torch.empty(K).uniform_(-a.band, a.band, generator=g).to(dev)
            tf["step"] = 0
            meta[sd] = [round(float(x), 5) for x in tf["alpha"]]
        render(packed_noise(sd), step_cb).save(os.path.join(a.out, "rediverse", f"{sd}.png"))
    pipe.set_adapters(["lightning"], adapter_weights=[1.0])
    with open(os.path.join(a.out, "rediverse", "settings.json"), "w") as f:
        json.dump(dict(prompt=a.prompt, mode=a.mode, band=a.band, seeds=seeds, alphas=meta), f, indent=1)
    print(f"[rediverse] wrote {len(seeds)} images to {a.out}/rediverse" + (f" and {a.out}/lightning" if a.compare else ""))


if __name__ == "__main__":
    main()
