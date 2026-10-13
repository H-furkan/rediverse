#!/usr/bin/env python
"""Generate SDXL images with the five arms of the paper.

  teacher       SDXL-base-1.0, 50-step EulerDiscrete, CFG 8 (no negative prompt)
  student       DMD2 4-step UNet, LCMScheduler, timesteps 999/749/499/249, no CFG
  rediverse_ft  student + the ReDiverse-FT LoRA, enabled for the first step only
  rediverse_tf  student + sum_k alpha_k s_k u_k added to up_blocks.0.attentions.0 in step 1
                (K = 8 teacher directions per prompt, alpha_k ~ U[-2, 2] per sample)
  rediverse     ReDiverse-FT + ReDiverse-TF

All arms run the SDXL-base fp16 pipeline at 1024 x 1024 with the float16 text encoders and
the fp32-upcast VAE, in batches of 10 samples of one prompt (results depend on the batch size).
Sample s of prompt i starts from the CUDA draw seeded i*1000 + s (the noise bank stores exactly
that draw; one draw per generator is consumed before sampling so that the student's between-step
re-noising continues from the same generator state as without the bank) and receives the
ReDiverse-TF coefficients seeded 7_000_000 + i*1009 + s.

Benchmark mode (GenEval-553 x 100, resumable, shardable):
  python -m sdxl.generate --arm rediverse
  python -m sdxl.generate --arm student --shard 0 --num-shards 4 --device cuda:0
  -> outputs/sdxl/<arm>/images/{i:03d}_s{s}.png

Single prompt (sample k follows the seed rules of prompt index 0):
  python -m sdxl.generate --arm rediverse --prompt "a photo of a fox" --num-samples 4
  -> outputs/sdxl/prompt/<prompt>/<arm>/{k:02d}.png
  The TF arms read --directions, else outputs/sdxl/prompt/<prompt>/directions.pt; when that file
  is missing or holds another prompt, the directions are estimated first
  (sdxl.estimate_directions --prompt, ~15 s) and saved there.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

from rediverse import hub
from rediverse.data import GENEVAL_553, parse_ids, prompt_dir, read_prompts
from rediverse.directions import N_DIR, direction_file, load_directions, matches, tf_offset
from rediverse.seeds import TF_BAND, alpha_record, image_eval_generator, tf_alphas_batch
from sdxl import config as C

ARMS = ("teacher", "student", "rediverse_ft", "rediverse_tf", "rediverse")
FT_ARMS = ("rediverse_ft", "rediverse")
TF_ARMS = ("rediverse_tf", "rediverse")
DTYPE = torch.float16


# ---------------------------------------------------------------- pipelines --
def build_teacher_pipe(device):
    """SDXL-base fp16 pipeline with its default EulerDiscreteScheduler."""
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(C.SDXL_ID, revision=C.SDXL_REVISION, torch_dtype=DTYPE,
                                                     variant="fp16", add_watermarker=False)
    pipe.text_encoder.to(DTYPE); pipe.text_encoder_2.to(DTYPE)
    pipe.unet.to(DTYPE)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def build_student_pipe(device):
    """SDXL-base fp16 pipeline with the DMD2 fp16 UNet (strict load) and an LCMScheduler built
    from the SDXL scheduler config."""
    from diffusers import LCMScheduler, StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download
    pipe = StableDiffusionXLPipeline.from_pretrained(C.SDXL_ID, revision=C.SDXL_REVISION, torch_dtype=DTYPE,
                                                     variant="fp16", add_watermarker=False)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    sd = torch.load(hf_hub_download(C.DMD2_ID, C.DMD2_UNET_FP16, revision=C.DMD2_REVISION),
                    map_location="cpu", weights_only=True)
    missing, unexpected = pipe.unet.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    pipe.text_encoder.to(DTYPE); pipe.text_encoder_2.to(DTYPE)
    pipe.unet.to(DTYPE)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def attach_adapter(unet, path: str):
    """Add the released ReDiverse-FT LoRA to the student UNet (cast to float16, as the UNet).
    Target modules and rank are read off the tensors; alpha from the file's metadata."""
    from peft import LoraConfig
    rel, meta = hub.load_lora_safetensors(path)
    # released '<module>.lora_X.weight' -> peft's '<module>.lora_X.default.weight'
    sd = {re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", k): v.to(DTYPE) for k, v in rel.items()}
    targets = sorted({k.split(".lora_")[0] for k in sd})
    ranks = {v.shape[0] for k, v in sd.items() if ".lora_A" in k}
    assert len(ranks) == 1, f"mixed LoRA ranks in {path}: {sorted(ranks)}"
    rank = ranks.pop()
    alpha = int(meta.get("alpha", rank))
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets))
    _, unexpected = unet.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected adapter keys: {unexpected[:5]}"
    n_lora = sum(1 for n, _ in unet.named_parameters() if "lora_" in n)
    assert len(sd) == n_lora, f"{len(sd)} adapter tensors for {n_lora} LoRA parameters"
    print(f"adapter attached: r={rank} alpha={alpha}, {len(targets)} modules, {len(sd)} tensors from {path}",
          flush=True)


class Sampler:
    """One arm: the pipeline plus the step-1 adapter gate and the ReDiverse-TF injection hook."""

    def __init__(self, arm: str, device: str, adapter: str = None, n_dir: int = N_DIR, band: float = TF_BAND):
        self.arm, self.device, self.band, self.n_dir = arm, device, band, n_dir
        self.use_ft, self.use_tf = arm in FT_ARMS, arm in TF_ARMS
        self.tf = None
        if arm == "teacher":
            self.pipe = build_teacher_pipe(device)
            return
        self.pipe = build_student_pipe(device)
        if self.use_ft:
            attach_adapter(self.pipe.unet, hub.resolve_adapter(adapter, C.BACKBONE))
        if self.use_tf:
            self.tf = dict(U=None, s=None, alpha=None, step=0)
            self.pipe.unet.up_blocks[0].attentions[0].register_forward_hook(self._tf_hook)

    def _tf_hook(self, _m, _i, out):
        """Step 1 only: h <- h + sum_k alpha_k s_k u_k (per image in the batch)."""
        tf = self.tf
        if tf["step"] != 0 or tf["U"] is None:
            return None
        o = out[0] if isinstance(out, tuple) else out                    # [B, 1280, 32, 32]
        o = o + tf_offset(tf["alpha"], tf["s"], tf["U"]).reshape(o.shape).to(o.dtype)
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o

    def _gate(self, pipe, i, t, cb_kwargs):
        """callback_on_step_end: after step 1 the adapter is disabled and the injection stops."""
        if self.tf is not None:
            self.tf["step"] = i + 1
        if i == 0 and self.use_ft:
            pipe.unet.disable_adapters()
        return cb_kwargs

    def set_directions(self, U: torch.Tensor, s: torch.Tensor):
        self.tf["U"], self.tf["s"] = U.to(self.device), s.to(self.device, torch.float32)

    def __call__(self, prompt: str, prompt_idx: int, samples, latents: torch.Tensor = None):
        """Images (uint8 [B, H, W, 3]) and the TF coefficients of samples `samples` of a prompt."""
        gens = [image_eval_generator(prompt_idx, s, self.device) for s in samples]
        kwargs = dict(height=C.HEIGHT, width=C.WIDTH, generator=gens, output_type="np")
        if self.arm == "teacher":
            kwargs.update(num_inference_steps=C.TEACHER_STEPS, guidance_scale=C.TEACHER_CFG)
        else:
            kwargs.update(num_inference_steps=len(C.STUDENT_TIMESTEPS), timesteps=C.STUDENT_TIMESTEPS,
                          guidance_scale=0.0)
        if latents is not None:
            # the pipeline skips its own initial draw when latents are given: consume it here
            for g in gens:
                torch.randn((1, *C.LATENT_SHAPE), generator=g, device=self.device, dtype=DTYPE)
            kwargs["latents"] = latents.to(self.device)
        if self.use_ft:
            self.pipe.unet.enable_adapters()
        if self.use_ft or self.use_tf:
            kwargs["callback_on_step_end"] = self._gate
        alphas = None
        if self.use_tf:
            alphas = tf_alphas_batch(prompt_idx, samples, k=self.tf["U"].shape[0], a=self.band)   # [B, K]
            self.tf["alpha"], self.tf["step"] = alphas.to(self.device), 0
        images = self.pipe([prompt] * len(gens), **kwargs).images
        assert np.isfinite(images).all(), f"non-finite decode at prompt {prompt_idx}"
        return (images * 255).round().astype("uint8"), alphas


# ------------------------------------------------------------------- modes --
def run_benchmark(a):
    prompts = read_prompts(a.prompts)
    out_dir = a.out_dir or C.output_dir(a.arm, "images")
    os.makedirs(out_dir, exist_ok=True)
    only = set(parse_ids(a.prompt_ids)) if a.prompt_ids else None
    work = [(i, p) for i, p in enumerate(prompts)
            if (only is None or i in only) and i % a.num_shards == a.shard
            and not all(os.path.exists(os.path.join(out_dir, f"{i:03d}_s{s}.png")) for s in range(a.num_samples))]
    print(f"[sdxl/{a.arm} shard {a.shard}/{a.num_shards}] {len(work)} prompts -> {out_dir}", flush=True)
    if not work:
        return
    if a.arm in TF_ARMS:
        missing = [direction_file(a.directions_dir, i) for i, _ in work
                   if not os.path.exists(direction_file(a.directions_dir, i))]
        if missing:
            sys.exit(f"direction files missing for {len(missing)} of {len(work)} prompts (first {missing[0]}); "
                     "run python -m sdxl.estimate_directions first")
    sampler = Sampler(a.arm, a.device, adapter=a.adapter, n_dir=a.n_dir, band=a.tf_band)
    bank = None
    if a.noise != "none" and os.path.exists(a.noise):
        bank = np.load(a.noise)
        assert bank[f"{work[0][0]:03d}"].shape[0] >= a.num_samples, f"noise bank narrower than {a.num_samples}"
        print(f"initial latents from the noise bank {a.noise}", flush=True)
    else:
        print("initial latents drawn from the seeded generators (identical to the noise bank)", flush=True)
    scales_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), f"scales_shard{a.shard}of{a.num_shards}.json")
    scales = {}
    if sampler.use_tf and os.path.exists(scales_path):
        with open(scales_path) as f:
            scales = json.load(f)
    t0 = time.time()
    for n, (i, prompt) in enumerate(work):
        if sampler.use_tf:
            sampler.set_directions(*load_directions(direction_file(a.directions_dir, i), a.n_dir, prompt=prompt))
        lat_i = bank[f"{i:03d}"] if bank is not None else None
        for s0 in range(0, a.num_samples, a.batch_size):
            ss = range(s0, min(s0 + a.batch_size, a.num_samples))
            lat = torch.from_numpy(lat_i[ss.start:ss.stop]) if lat_i is not None else None
            images, alphas = sampler(prompt, i, ss, lat)
            for j, s in enumerate(ss):
                Image.fromarray(images[j]).save(os.path.join(out_dir, f"{i:03d}_s{s}.png"))
                if alphas is not None:
                    scales[f"{i:03d}_s{s}"] = alpha_record(alphas[j])
        if scales:
            with open(scales_path + ".tmp", "w") as f:
                json.dump(scales, f, indent=0)
            os.replace(scales_path + ".tmp", scales_path)
        if (n + 1) % 20 == 0:
            rate = (n + 1) / (time.time() - t0)
            print(f"  {n+1}/{len(work)}, {rate*60:.1f} prompts/min, ETA {(len(work)-n-1)/rate/60:.0f} min", flush=True)
    print(f"[sdxl/{a.arm} shard {a.shard}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def prompt_directions(a) -> str:
    """Direction file of the --prompt run: --directions as given, else the prompt's directions.pt,
    which is estimated with the teacher (sdxl.estimate_directions) when it is missing or holds
    another prompt or prompt index. Runs before the student is loaded, so the teacher is freed
    by then."""
    if a.directions:
        return a.directions
    from sdxl import estimate_directions as E
    path = E.prompt_directions_file(a.prompt)
    if matches(path, a.prompt, C.PROMPT_SEED_INDEX):
        print(f"ReDiverse-TF directions from {path}", flush=True)
    else:
        print(f"estimating ReDiverse-TF directions for {a.prompt!r} -> {path}", flush=True)
        E.estimate_prompt(a.prompt, path, a.device)
    return path


def run_prompt(a):
    out_dir = a.out_dir or prompt_dir(C.BACKBONE, a.prompt, a.arm)
    os.makedirs(out_dir, exist_ok=True)
    directions = prompt_directions(a) if a.arm in TF_ARMS else None
    sampler = Sampler(a.arm, a.device, adapter=a.adapter, n_dir=a.n_dir, band=a.tf_band)
    if directions:
        sampler.set_directions(*load_directions(directions, a.n_dir))
    for k0 in range(0, a.num_samples, a.batch_size):
        ks = range(k0, min(k0 + a.batch_size, a.num_samples))
        images, _ = sampler(a.prompt, C.PROMPT_SEED_INDEX, ks)
        for j, k in enumerate(ks):
            Image.fromarray(images[j]).save(os.path.join(out_dir, f"{k:02d}.png"))
    print(f"[sdxl/{a.arm}] {a.num_samples} images -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--prompt", default=None, help="single-prompt mode: the text to render")
    ap.add_argument("--prompts", default=GENEVAL_553, help="benchmark prompt file; line index = prompt index")
    ap.add_argument("--prompt-ids", default=None, help="benchmark subset, e.g. '27' or '0-9,27'")
    ap.add_argument("--num-samples", type=int, default=None,
                    help=f"samples per prompt (default {C.NUM_SAMPLES} in benchmark mode, 4 with --prompt)")
    ap.add_argument("--batch-size", type=int, default=C.BATCH, help="samples per pipeline call")
    ap.add_argument("--noise", default=C.noise_bank_path(),
                    help="noise bank (sdxl/make_noise_bank.py); 'none' or a missing file = seeded generators")
    ap.add_argument("--adapter", default=None, help="ReDiverse-FT adapter: .safetensors file or a directory holding "
                                                    "rediverse_ft.safetensors (default: download from the Hub)")
    ap.add_argument("--directions-dir", default=C.directions_dir(), help="benchmark mode: direction files p{i:03d}.pt "
                                                                         "of the --prompts file")
    ap.add_argument("--directions", default=None, help="single-prompt mode: direction file (default: the prompt's "
                                                        "directions.pt, estimated if missing or of another prompt)")
    ap.add_argument("--n-dir", type=int, default=N_DIR, help="directions K")
    ap.add_argument("--tf-band", type=float, default=TF_BAND, help="alpha_k ~ U[-a, a]")
    ap.add_argument("--out-dir", default=None, help="default outputs/sdxl/<arm>/images, or "
                                                    "outputs/sdxl/prompt/<prompt>/<arm> with --prompt")
    ap.add_argument("--shard", type=int, default=0, help="benchmark mode: prompts with index %% num-shards == shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.num_samples is None:
        a.num_samples = 4 if a.prompt else C.NUM_SAMPLES

    if a.prompt:
        run_prompt(a)
    else:
        run_benchmark(a)


if __name__ == "__main__":
    main()
