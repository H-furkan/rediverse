#!/usr/bin/env python
"""ReDiverse-TF directions for SDXL: per-prompt top-8 teacher principal directions at
up_blocks.0.attentions.0, teacher step 7 (paper Sec. 4.2 and App. D).

For GenEval prompt i the fp16 SDXL-base teacher is rolled conditional-only (no CFG) along its
50-step Euler trajectory from N = 25 latents, randn(1, 4, 128, 128) on a CPU generator seeded
i*1000 + j, j = 100..124 (disjoint from the evaluation seeds i*1000 + 0..99). The step-7 activation
[1280, 32, 32] of each rollout is flattened (D = 1,310,720, stored bfloat16) and decomposed by
rediverse.directions.directions_from_samples. Output: <out-dir>/p{i:03d}.pt with U [8, D] bf16,
s [8], eigvals [25], pc1_share and the estimation metadata (~21 MB per prompt, ~14 s per prompt on
an H100). Several workers may share an output directory (per-file claims).

  python -m sdxl.estimate_directions                       # all 553 GenEval prompts
  python -m sdxl.estimate_directions --shard 0 --num-shards 4 --device cuda:0
  python -m sdxl.estimate_directions --prompt-ids 27

With --prompt the same estimation runs for one free-text prompt with the seed rules of prompt
index 0 and writes outputs/sdxl/prompt/<prompt>/directions.pt, the file that
`sdxl.generate --prompt` reads (and writes itself when it is missing or holds another prompt):

  python -m sdxl.estimate_directions --prompt "a photo of a fox in the snow"
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from rediverse.claims import claim
from rediverse.data import GENEVAL_553, parse_ids, prompt_dir, read_prompts
from rediverse.directions import N_DIR, direction_file, directions_from_samples, matches, save_directions
from rediverse.seeds import cpu_randn, direction_sample_indices, image_direction_seed
from sdxl import config as C
from sdxl.teacher_rollout import teacher_states

DTYPE = torch.float16


def load_text_encoders(device):
    """SDXL's two text encoders in float16 (pipeline without UNet and VAE)."""
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(C.SDXL_ID, revision=C.SDXL_REVISION, unet=None, vae=None,
                                                     torch_dtype=DTYPE, add_watermarker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def load_teacher(device):
    """The SDXL-base UNet, float16 weights, frozen."""
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(C.SDXL_ID, subfolder="unet", revision=C.SDXL_REVISION,
                                                torch_dtype=DTYPE, variant="fp16")
    unet.eval().requires_grad_(False)
    return unet.to(device)


@torch.no_grad()
def encode(pipe, prompt: str, device) -> dict:
    """pe [1, 77, 2048], pool [1, 1280], tid [1, 6]: conditional embeddings and the time ids of an
    uncropped 1024 x 1024 render (original size + crop top-left + target size)."""
    pe, _, pool, _ = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1,
                                        do_classifier_free_guidance=False)
    tid = torch.tensor([[C.HEIGHT, C.WIDTH, 0, 0, C.HEIGHT, C.WIDTH]], dtype=DTYPE)
    return dict(pe=pe.cpu(), pool=pool.cpu(), tid=tid)


@torch.no_grad()
def estimate(unet, ctx: dict, prompt_idx: int, device, n: int = C.N_DIRECTION_SAMPLES, k_step: int = C.DIRECTION_STEP):
    """(U, s, eigvals, seeds, shape) for one prompt from n conditional-only teacher rollouts."""
    site = unet.up_blocks[0].attentions[0]
    cond = {key: ctx[key].to(device, DTYPE) for key in ("pe", "pool", "tid")}
    seeds = direction_sample_indices(n)
    X, shape = None, None
    for j, s in enumerate(seeds):
        x0 = cpu_randn(image_direction_seed(prompt_idx, s), (1, *C.LATENT_SHAPE)).to(device, DTYPE)
        out = teacher_states(unet, {C.SITE: site}, x0, cond, k_step, keep={k_step}, steps=C.TEACHER_STEPS,
                             model_id=C.SDXL_ID, dtype=DTYPE, device=device)
        h = out[k_step][C.SITE][0]
        shape = tuple(h.shape)
        h = h.reshape(-1).to(torch.bfloat16).cpu()
        if X is None:
            X = torch.empty(n, h.numel(), dtype=torch.bfloat16)
        X[j] = h
    U, S, evals = directions_from_samples(X, N_DIR)
    return U, S, evals, seeds, shape


def prompt_directions_file(prompt: str, out_dir: str = None) -> str:
    """<out_dir>/directions.pt, by default in the prompt's output directory."""
    return os.path.join(out_dir or prompt_dir(C.BACKBONE, prompt), "directions.pt")


def estimate_prompt(prompt: str, path: str, device, n: int = C.N_DIRECTION_SAMPLES) -> dict:
    """Estimate and save the directions of one free-text prompt (seed rules of prompt index
    C.PROMPT_SEED_INDEX); the teacher is released before returning."""
    pipe = load_text_encoders(device)
    unet = load_teacher(device)
    ctx = encode(pipe, prompt, device)
    U, S, evals, seeds, shape = estimate(unet, ctx, C.PROMPT_SEED_INDEX, device, n=n)
    del pipe, unet
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rec = save_directions(path, U, S, evals, prompt_idx=C.PROMPT_SEED_INDEX, prompt=prompt, k=C.DIRECTION_STEP,
                          site=C.SITE, n=n, seeds=seeds, shape=shape)
    print(f"[sdxl/directions] {prompt!r}: s1={float(S[0]):.2f} pc1_share={rec['pc1_share']:.3f} -> {path}",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=GENEVAL_553, help="prompt file; line index = prompt index")
    ap.add_argument("--prompt-ids", default=None, help="e.g. '27' or '0-9,27' (default: all, sharded)")
    ap.add_argument("--prompt", default=None, help="one free-text prompt instead of the prompt file")
    ap.add_argument("--out-dir", default=None, help=f"default {C.directions_dir()}; with --prompt the prompt's "
                                                    "output directory")
    ap.add_argument("--n-lat", type=int, default=C.N_DIRECTION_SAMPLES)
    ap.add_argument("--shard", type=int, default=0, help="this worker takes prompts with index %% num-shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.prompt:
        path = prompt_directions_file(a.prompt, a.out_dir)
        if matches(path, a.prompt, C.PROMPT_SEED_INDEX):
            print(f"[sdxl/directions] {path} holds this prompt's directions", flush=True)
            return
        estimate_prompt(a.prompt, path, a.device, n=a.n_lat)
        return
    out_dir = a.out_dir or C.directions_dir()
    texts = read_prompts(a.prompts)
    idxs = parse_ids(a.prompt_ids) if a.prompt_ids else [i for i in range(len(texts)) if i % a.num_shards == a.shard]
    os.makedirs(out_dir, exist_ok=True)
    pending, other = [], []
    for i in idxs:
        f = direction_file(out_dir, i)
        if not os.path.exists(f):
            pending.append(i)
        elif not matches(f, texts[i]):
            other.append(f)
    if other:
        raise ValueError(f"{out_dir} holds the directions of other prompts than {a.prompts} for {len(other)} prompt "
                         f"indices (first {other[0]}); pass --out-dir to estimate a separate direction bank for this "
                         "prompt file")
    print(f"[sdxl/directions] {C.SITE} step {C.DIRECTION_STEP}: {len(pending)}/{len(idxs)} prompts to estimate",
          flush=True)
    if not pending:
        return
    pipe = load_text_encoders(a.device)
    unet = load_teacher(a.device)
    t0, done = time.time(), 0
    for i in pending:
        f_out = direction_file(out_dir, i)
        if os.path.exists(f_out) or not claim(f_out):
            continue
        ctx = encode(pipe, texts[i], a.device)
        U, S, evals, seeds, shape = estimate(unet, ctx, i, a.device, n=a.n_lat)
        rec = save_directions(f_out, U, S, evals, prompt_idx=i, prompt=texts[i], k=C.DIRECTION_STEP, site=C.SITE,
                              n=a.n_lat, seeds=seeds, shape=shape)
        done += 1
        el = time.time() - t0
        if done % 10 == 0 or done == 1:
            print(f"  [{done}/{len(pending)}] p{i:03d} s1={float(S[0]):.2f} pc1_share={rec['pc1_share']:.3f} "
                  f"{el/done:.1f}s/prompt eta {(len(pending)-done)*el/done/60:.0f}m", flush=True)
    print(f"[sdxl/directions] shard {a.shard}/{a.num_shards} done", flush=True)


if __name__ == "__main__":
    main()
