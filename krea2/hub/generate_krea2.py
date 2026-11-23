#!/usr/bin/env python
"""Krea 2 ReDiverse: diverse samples from Krea 2 Turbo (diffusers only).

  # adapter + steering (recommended; needs access to krea/Krea-2-Raw for the prompt's directions, ~1 min per prompt)
  python generate_krea2.py --prompt "a photo of a dog" --seeds 0-7 --out out --compare
  # adapter only (no teacher needed)
  python generate_krea2.py --prompt "a photo of a dog" --seeds 0-7 --out out --mode ft --compare
  # paper setting of the steering strength
  python generate_krea2.py --prompt "a photo of a dog" --seeds 0-7 --out out --band 2

Writes out/rediverse/<seed>.png and, with --compare, seed-matched out/turbo/<seed>.png from Krea 2 Turbo alone.

What it does (ReDiverse, arXiv:2611.18550):
  ReDiverse-FT  a rank-32 LoRA on transformer_blocks 0..26 of Krea 2 Turbo, trained to match Krea 2 Raw's block-26 activations;
                active in the first of the 8 sampling steps only.
  ReDiverse-TF  for each prompt, the top-8 principal directions of Krea 2 Raw's block-26 image activations over 25 noises
                (conditional, at Raw's 5th step); every sample adds sum_k alpha_k s_k u_k to block 26 in the first step,
                alpha_k ~ U[-band, band] (default band 4; the paper uses 2).
Images are 1024x1024 (the directions are tied to this resolution); Turbo runs 8 steps without guidance, as on its model card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

TURBO, RAW = "krea/Krea-2-Turbo", "krea/Krea-2-Raw"
ADAPTER_REPO, ADAPTER_FILE = "ReDiverse/Krea-2-ReDiverse", "krea-2-rediverse.safetensors"
BLOCK, TAU, N_ROLL, K, RES = 26, 5, 25, 8, 1024
N_IMG = (RES // 16) ** 2                      # 4096 image tokens at 1024^2 (VAE /8, patch 2)
D_MODEL = 6144


def parse_seeds(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            out += list(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def packed_noise(seed: int) -> torch.Tensor:
    """[1, 4096, 64] standard normal on a CPU generator (packed latent layout of Krea2Pipeline)."""
    return torch.randn((1, N_IMG, 64), generator=torch.Generator("cpu").manual_seed(seed))


def raw_sigmas() -> tuple[list[float], float]:
    """Krea 2 Raw's own 52-step schedule at 1024^2 (resolution-dependent shift, as Krea2Pipeline computes it)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    m = (1.15 - 0.5) / (6400 - 256)
    mu = N_IMG * m + (0.5 - m * 256)
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(TURBO, subfolder="scheduler")
    sch.set_timesteps(sigmas=np.linspace(1.0, 1 / 52, 52), mu=mu)
    return [float(s) for s in sch.sigmas], mu


@torch.no_grad()
def estimate_directions(pipe, prompt: str, device, cache_dir: str) -> dict:
    """Top-K principal directions of Raw's block-26 image activations at its forward 5 (25 conditional rollouts)."""
    key = hashlib.sha1(prompt.encode()).hexdigest()[:16]
    f = os.path.join(cache_dir, f"directions_{key}.pt")
    if os.path.exists(f):
        d = torch.load(f, map_location="cpu")
        if d.get("prompt") == prompt:
            return d
    from diffusers import Krea2Transformer2DModel
    print(f"[rediverse] estimating teacher directions with {RAW} (25 rollouts) ...", flush=True)
    from huggingface_hub import snapshot_download
    try:                                           # transformer files only; also works offline from the local cache
        path = snapshot_download(RAW, allow_patterns=["transformer/*"])
    except Exception:
        path = snapshot_download(RAW, allow_patterns=["transformer/*"], local_files_only=True)
    raw = Krea2Transformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=torch.bfloat16).to(device).eval()
    pe, pm = pipe.encode_prompt(prompt=[prompt], device=device)
    pos = pipe.prepare_position_ids(pe.shape[1], RES // 16, RES // 16, device)
    sg, _ = raw_sigmas()
    grab = {}

    class _Stop(Exception):
        pass

    def hook(_m, _i, out):
        grab["h"] = out[:, -N_IMG:].detach().float().cpu().reshape(-1)
        raise _Stop
    X = torch.empty(N_ROLL, N_IMG * D_MODEL)
    for j in range(N_ROLL):
        x = packed_noise(100 + j).to(device, torch.bfloat16)
        for i in range(TAU):
            t = torch.tensor([sg[i]], device=device).to(torch.bfloat16)
            kw = dict(hidden_states=x, encoder_hidden_states=pe, timestep=t, position_ids=pos,
                      encoder_attention_mask=pm, return_dict=False)
            if i < TAU - 1:
                v = raw(**kw)[0]
                x = (x.float() + (sg[i + 1] - sg[i]) * v.float()).to(torch.bfloat16)
            else:
                h = raw.transformer_blocks[BLOCK].register_forward_hook(hook)
                try:
                    raw(**kw)
                except _Stop:
                    pass
                finally:
                    h.remove()
        X[j] = grab["h"]
    del raw
    torch.cuda.empty_cache()
    Xc = X - X.mean(0, keepdim=True)
    evals, evecs = torch.linalg.eigh((Xc @ Xc.T).double())
    evals, evecs = evals.flip(0), evecs.flip(1)
    U = torch.stack([(evecs[:, k].float() @ Xc) / (evecs[:, k].float() @ Xc).norm() for k in range(K)]).to(torch.bfloat16)
    s = torch.tensor([float((evals[k].clamp_min(0) ** 0.5 * evecs[:, k]).std(unbiased=False)) for k in range(K)])
    d = dict(prompt=prompt, U=U, s=s, block=BLOCK, tau=TAU, res=RES)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(d, f)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seeds", default="0-3")
    ap.add_argument("--out", default="out")
    ap.add_argument("--mode", choices=("rediverse", "ft"), default="rediverse",
                    help="rediverse = adapter + steering (needs krea/Krea-2-Raw); ft = adapter only")
    ap.add_argument("--band", type=float, default=4.0, help="steering strength: alpha ~ U[-band, band] (paper: 2)")
    ap.add_argument("--adapter", default=None, help="local krea-2-rediverse.safetensors (default: download from the Hub)")
    ap.add_argument("--compare", action="store_true", help="also render Krea 2 Turbo alone from the same noises")
    ap.add_argument("--cache", default=".rediverse_cache")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from diffusers import Krea2Pipeline
    dev = a.device
    pipe = Krea2Pipeline.from_pretrained(TURBO, torch_dtype=torch.bfloat16).to(dev)
    seeds = parse_seeds(a.seeds)
    os.makedirs(os.path.join(a.out, "rediverse"), exist_ok=True)

    def render(latents, cb=None):
        return pipe(prompt=a.prompt, height=RES, width=RES, num_inference_steps=8, guidance_scale=0.0,
                    latents=latents.to(dev, torch.bfloat16), callback_on_step_end=cb).images[0]

    if a.compare:                                  # Turbo alone first, before any adapter or hook exists
        os.makedirs(os.path.join(a.out, "turbo"), exist_ok=True)
        for sd in seeds:
            render(packed_noise(sd)).save(os.path.join(a.out, "turbo", f"{sd}.png"))

    tf = None
    if a.mode == "rediverse":
        d = estimate_directions(pipe, a.prompt, dev, a.cache)
        tf = dict(U=d["U"].to(dev), s=d["s"].to(dev, torch.float32), alpha=None, step=0)

        def tf_hook(_m, _i, out):
            if tf["step"] != 0 or tf["alpha"] is None:
                return None
            delta = ((tf["alpha"] * tf["s"]).to(tf["U"].dtype) @ tf["U"]).reshape(1, N_IMG, D_MODEL)
            o = out.clone()
            o[:, -N_IMG:] += delta.to(out.dtype)
            return o
        pipe.transformer.transformer_blocks[BLOCK].register_forward_hook(tf_hook)

    if a.adapter:                                  # a local .safetensors file
        from safetensors.torch import load_file
        pipe.load_lora_weights(load_file(a.adapter), adapter_name="rediverse")
    else:
        pipe.load_lora_weights(ADAPTER_REPO, weight_name=ADAPTER_FILE, adapter_name="rediverse")

    def step_cb(p, i, t, kw):                      # ReDiverse acts in the first step only
        if i == 0:
            p.transformer.disable_adapters()
            if tf is not None:
                tf["step"] = 1
        return kw

    meta = {}
    for sd in seeds:
        pipe.transformer.enable_adapters()
        if tf is not None:
            g = torch.Generator("cpu").manual_seed(7_000_000 + sd)
            tf["alpha"] = torch.empty(K).uniform_(-a.band, a.band, generator=g).to(dev)
            tf["step"] = 0
            meta[sd] = [round(float(x), 5) for x in tf["alpha"]]
        render(packed_noise(sd), step_cb).save(os.path.join(a.out, "rediverse", f"{sd}.png"))
    pipe.transformer.disable_adapters()
    with open(os.path.join(a.out, "rediverse", "settings.json"), "w") as f:
        json.dump(dict(prompt=a.prompt, mode=a.mode, band=a.band, seeds=seeds, alphas=meta), f, indent=1)
    print(f"[rediverse] wrote {len(seeds)} images to {a.out}/rediverse" + (f" and {a.out}/turbo" if a.compare else ""))


if __name__ == "__main__":
    main()
