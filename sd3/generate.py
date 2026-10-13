"""Generate SD3 images for the five arms of the paper.

    teacher        SD3-Medium, 28 FlowMatchEuler steps, CFG 7
    student        + TDM LoRA at 0.125, 4 steps, guidance 1.0
    rediverse_ft   student + the ReDiverse-FT adapter, active at sampling step 1 only
    rediverse_tf   student + ReDiverse-TF directions added at step 1
    rediverse      student + ReDiverse-FT adapter + ReDiverse-TF directions

Benchmark mode (GenEval-553, 100 samples per prompt, frozen noise bank, batch 10):

    python -m sd3.generate --arm rediverse
    python -m sd3.generate --arm teacher --prompt-ids 0-99 --num-samples 10

writes <REDIVERSE_OUT>/sd3/<arm>/images/{idx:03d}_s{s}.png. Several workers (one per GPU) can
share an arm: each takes per-prompt claims under <out-dir>/.claims. The TF arms read the direction
files p{idx:03d}.pt of sd3.estimate_directions, each of which must hold its prompt's directions.

Prompt mode:

    python -m sd3.generate --arm rediverse --prompt "a photo of a fox" --num-samples 4

writes <REDIVERSE_OUT>/sd3/prompt/<slug>/<arm>/{k:02d}.png for k = 0..N-1. A custom prompt is
treated as prompt index 0 of every seed rule (sample k: latent seed k, TF coefficients seeded
7_000_000 + k). The TF arms read --directions, else <REDIVERSE_OUT>/sd3/prompt/<slug>/directions.pt;
when that file is missing or holds the directions of another prompt it is estimated first with
sd3.estimate_directions (the teacher is freed before the student loads) and kept for later calls.

Inference mechanics of the adapter arms, as trained:
  * step-1 gate: the ReDiverse-FT adapter is switched off after sampling step 1; the TDM adapter
    stays on at every step;
  * stream restriction: at step 1 a forward truncated after block 12 with the adapter off reads
    the student's block-12 text stream, and the adapted forward has its block-12 text output
    replaced by it, so the correction acts on the image stream only (the stream the loss
    supervised);
  * ReDiverse-TF: at step 1, h_img += sum_k alpha_k s_k u_k on the block-12 image stream, alpha_k ~
    U[-2, 2] seeded per (prompt, sample) (rediverse.seeds.tf_alphas), identical in rediverse_tf and
    rediverse; in rediverse it is added after the stream restriction.
The initial latent is the whole stochastic input (the FlowMatchEuler sampler draws nothing), and
results are bit-reproducible only at the same batch size (10) and library versions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from rediverse import data as D
from rediverse.claims import claim_tag, release_tag
from rediverse.directions import N_DIR, direction_file, load_directions, matches, tf_offset
from rediverse.hub import resolve_adapter
from rediverse.seeds import TF_BAND, alpha_record, image_eval_noise, tf_alphas_batch
from sd3.config import (ANCHOR_BLOCK, BACKBONE, BATCH, DTYPE, FT_ADAPTER, GUIDANCE, LATENT_SHAPE,
                        MODEL_ID, MODEL_REVISION, N_BLOCKS, NUM_SAMPLES, PROMPT_NUM_SAMPLES,
                        PROMPT_SEED_INDEX, STEPS, TDM_ADAPTER, TDM_SCALE, directions_dir,
                        noise_bank_path, output_dir)
from sd3 import model as M

ARMS = ("teacher", "student", "rediverse_ft", "rediverse_tf", "rediverse")
FT_ARMS = ("rediverse_ft", "rediverse")
TF_ARMS = ("rediverse_tf", "rediverse")


def arm_schedule(arm: str) -> tuple[int, float]:
    """(steps, guidance) of an arm: the teacher's schedule for the teacher, else the student's."""
    base = "teacher" if arm == "teacher" else "student"
    return STEPS[base], GUIDANCE[base]


def build_transformer(arm: str, device: str, adapter: str = None):
    """The arm's denoiser: the teacher, the student, or the student with the ReDiverse-FT adapter."""
    if arm in FT_ARMS:
        return M.load_student_with_adapter(device, resolve_adapter(adapter, "sd3"))
    m = M.load_teacher(device) if arm == "teacher" else M.load_student(device)
    m.requires_grad_(False).eval()
    return m


def build_pipe(device: str):
    """StableDiffusion3Pipeline with the text encoders and the VAE; the transformer is set per arm."""
    from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, transformer=None, torch_dtype=DTYPE).to(device)
    pipe.set_progress_bar_config(disable=True)
    assert isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler), type(pipe.scheduler).__name__
    return pipe


def install_stream_restriction(model, block: int = ANCHOR_BLOCK, weight: float = 1.0):
    """At step 1, keep the ReDiverse-FT correction on the block-`block` image stream only.

    Two passes on step 1: one truncated after `block` with the adapter off captures the student's
    block text stream; the full pass with the adapter on has that block's text output replaced
    by it. Later steps run unchanged. Returns the state the caller resets per batch
    (state["step"] = 0, state["txt"] = None) and advances per step."""
    blk = model.transformer_blocks[block]
    assert not getattr(blk, "context_pre_only", False)
    st = {"step": 0, "mode": None, "txt": None}

    def hook(_m, _i, out):
        if st["mode"] == "capture":
            st["txt"] = out[0].detach()
            return None
        if st["mode"] == "replace":
            return (st["txt"], out[1])          # student's text, adapted image
        return None

    blk.register_forward_hook(hook)
    orig = model.forward

    def fwd(*args, **kw):
        if st["step"] != 0:
            return orig(*args, **kw)
        st["mode"] = "capture"
        try:
            model.set_adapters([TDM_ADAPTER], weights=[TDM_SCALE])
            orig(*args, **{**kw, "skip_layers": list(range(block + 1, N_BLOCKS))})
        finally:
            st["mode"] = None
        assert st["txt"] is not None, "capture pass did not reach the collapsed block"
        model.set_adapters([TDM_ADAPTER, FT_ADAPTER], weights=[TDM_SCALE, weight])
        st["mode"] = "replace"
        try:
            return orig(*args, **kw)
        finally:
            st["mode"] = None

    model.forward = fwd
    return st


def install_directions(model, block: int = ANCHOR_BLOCK):
    """ReDiverse-TF hook: at step 1 add tf_offset(alpha, s, U) to the block's image stream.
    Returns the state holding U, s, alpha [B, K] and the step counter."""
    pc = dict(U=None, s=None, alpha=None, step=0)
    blk = model.transformer_blocks[block]

    def pc_hook(_m, _i, out):
        if pc["step"] != 0 or pc["U"] is None:
            return None
        txt, img = out
        delta = tf_offset(pc["alpha"], pc["s"], pc["U"])                  # [B, D]
        return (txt, img + delta.reshape(img.shape).to(img.dtype))

    blk.register_forward_hook(pc_hook)
    return pc


def make_callback(arm: str, ft_state, pc):
    """callback_on_step_end of the adapter / direction arms: advance the step counters and
    switch the ReDiverse-FT adapter off after step 1 (TDM stays on)."""
    if arm not in FT_ARMS + TF_ARMS:
        return None

    def cb(pl, step, timestep, kw):
        if ft_state is not None:
            ft_state["step"] = step + 1
        if pc is not None:
            pc["step"] = step + 1
        if step == 0 and arm in FT_ARMS:
            pl.transformer.set_adapters([TDM_ADAPTER], weights=[TDM_SCALE])
        return kw

    return cb


class Generator:
    """One arm: pipeline, hooks and the per-batch call."""

    def __init__(self, arm: str, device: str, adapter: str = None):
        assert arm in ARMS, arm
        self.arm, self.device = arm, device
        self.steps, self.guidance = arm_schedule(arm)
        self.pipe = build_pipe(device)
        self.pipe.transformer = build_transformer(arm, device, adapter)
        self.ft_state = install_stream_restriction(self.pipe.transformer) if arm in FT_ARMS else None
        self.pc = install_directions(self.pipe.transformer) if arm in TF_ARMS else None
        self.cb = make_callback(arm, self.ft_state, self.pc)

    def set_directions(self, U, s):
        self.pc["U"], self.pc["s"] = U, s

    @torch.no_grad()
    def __call__(self, prompt: str, latents: torch.Tensor, alpha: torch.Tensor = None) -> np.ndarray:
        """[B, 1024, 1024, 3] float images in [0, 1] for B latents of one prompt."""
        lat = latents.to(self.device, DTYPE)
        assert lat.shape[1:] == LATENT_SHAPE, tuple(lat.shape)
        if self.arm in FT_ARMS:                  # re-arm both adapters before every batch
            self.ft_state["step"], self.ft_state["txt"] = 0, None
            self.pipe.transformer.set_adapters([TDM_ADAPTER, FT_ADAPTER], weights=[TDM_SCALE, 1.0])
        if self.pc is not None:
            assert alpha is not None and alpha.shape[0] == lat.shape[0]
            self.pc["alpha"], self.pc["step"] = alpha.to(self.device), 0
        images = self.pipe(prompt=[prompt] * lat.shape[0], num_inference_steps=self.steps,
                           guidance_scale=self.guidance, latents=lat,
                           callback_on_step_end=self.cb, output_type="np").images
        assert np.isfinite(images).all(), "non-finite decode"
        return images


def save_png(arr: np.ndarray, final: str):
    """Atomic PNG write of a [H, W, 3] image in [0, 1]."""
    tmp = final + f".tmp{os.getpid()}"
    Image.fromarray((arr * 255).round().astype("uint8")).save(tmp, "PNG")
    os.replace(tmp, final)


def prompt_directions(a) -> str:
    """Direction file of a --prompt run of a TF arm: --directions as given, else the prompt's
    directions.pt, estimated here (sd3.estimate_directions) unless it already holds this prompt's
    directions."""
    if a.directions:
        return a.directions
    from sd3.estimate_directions import estimate_prompt, prompt_directions_path
    path = prompt_directions_path(a.prompt)
    if matches(path, a.prompt, PROMPT_SEED_INDEX):
        print(f"[gen] ReDiverse-TF directions from {path}", flush=True)
    else:
        print(f"[gen] estimating ReDiverse-TF directions for the prompt -> {path}", flush=True)
        estimate_prompt(a.prompt, path, a.device)
    return path


def run_prompt_mode(a, gen: Generator, directions: str = None):
    """--prompt: samples 0..num_samples-1 of one custom prompt (prompt index 0 of the seed rules),
    batches of a.batch_size."""
    n = a.num_samples
    out_dir = a.out_dir or D.prompt_dir(BACKBONE, a.prompt, a.arm)
    os.makedirs(out_dir, exist_ok=True)
    if gen.pc is not None:
        gen.set_directions(*load_directions(directions, N_DIR, a.device))
    for k0 in range(0, n, a.batch_size):
        chunk = list(range(k0, min(k0 + a.batch_size, n)))
        lat = torch.cat([image_eval_noise(PROMPT_SEED_INDEX, k, (1,) + LATENT_SHAPE, DTYPE, a.device)
                         for k in chunk])
        alpha = (tf_alphas_batch(PROMPT_SEED_INDEX, chunk, k=gen.pc["U"].shape[0], a=TF_BAND)
                 if gen.pc is not None else None)
        images = gen(a.prompt, lat, alpha)
        for j, k in enumerate(chunk):
            save_png(images[j], os.path.join(out_dir, f"{k:02d}.png"))
    print(f"[gen] {n} images -> {out_dir}", flush=True)


def run_benchmark(a):
    """GenEval-553 x num_samples from the noise bank, per-prompt claims, atomic PNG writes. The TF
    arms stop before loading any model when a direction file is missing."""
    prompts = D.read_prompts(a.prompts)
    keep = None if a.prompt_ids is None else set(D.parse_ids(a.prompt_ids))
    out_dir = a.out_dir or output_dir(a.arm, "images")
    os.makedirs(out_dir, exist_ok=True)
    claim_dir = None if a.no_claim else (a.claim_dir or os.path.join(out_dir, ".claims"))

    def path(i, s):
        return os.path.join(out_dir, f"{i:03d}_s{s}.png")

    work = [(i, p) for i, p in enumerate(prompts)
            if (keep is None or i in keep) and i % a.num_shards == a.shard
            and not all(os.path.exists(path(i, s)) for s in range(a.num_samples))]
    steps, guidance = arm_schedule(a.arm)
    print(f"[sd3/{a.arm} shard {a.shard}/{a.num_shards}] {len(work)} prompts x {a.num_samples} samples "
          f"-> {out_dir}; steps={steps} guidance={guidance} batch={a.batch_size}", flush=True)
    if not work:
        return
    dirs = a.directions_dir or directions_dir()
    if a.arm in TF_ARMS:
        missing = [i for i, _ in work if not os.path.exists(direction_file(dirs, i))]
        if missing:
            sys.exit(f"{len(missing)} of {len(work)} direction files are missing in {dirs} (first: "
                     f"{os.path.basename(direction_file(dirs, missing[0]))}); "
                     "run python -m sd3.estimate_directions first")
    gen = Generator(a.arm, a.device, a.adapter)

    noise_file = a.noise or noise_bank_path()
    if os.path.exists(noise_file):
        from sd3.make_noise_bank import load_bank
        get_noise = load_bank(noise_file, a.num_samples)
        print(f"[gen] initial latents from {noise_file}", flush=True)
    else:
        print(f"[gen] {noise_file} not found: drawing the bank's seed rule directly on {a.device} "
              "(equal to the bank on the GPU type it was built on; see sd3.make_noise_bank)", flush=True)

        def get_noise(i, s0, s1):
            return torch.cat([image_eval_noise(i, s, (1,) + LATENT_SHAPE, DTYPE, a.device)
                              for s in range(s0, s1)])

    savers = ThreadPoolExecutor(max_workers=4)
    pending, scales = [], {}
    t0, made, taken = time.time(), 0, 0
    for n, (i, prompt) in enumerate(work):
        if claim_dir and not claim_tag(claim_dir, f"{i:03d}"):
            continue
        taken += 1
        try:
            if gen.pc is not None:
                gen.set_directions(*load_directions(direction_file(dirs, i), N_DIR, a.device, prompt=prompt))
            for s0 in range(0, a.num_samples, a.batch_size):
                s1 = min(s0 + a.batch_size, a.num_samples)
                if all(os.path.exists(path(i, s)) for s in range(s0, s1)):
                    continue
                alpha = None
                if gen.pc is not None:
                    alpha = tf_alphas_batch(i, range(s0, s1), k=gen.pc["U"].shape[0], a=TF_BAND)
                    for j, s in enumerate(range(s0, s1)):
                        scales[f"{i:03d}_s{s}"] = alpha_record(alpha[j])
                images = gen(prompt, get_noise(i, s0, s1), alpha)
                for j, s in enumerate(range(s0, s1)):
                    pending.append(savers.submit(save_png, images[j], path(i, s)))
                made += s1 - s0
        finally:
            for f in pending:
                f.result()
            pending = []
            if claim_dir:
                release_tag(claim_dir, f"{i:03d}")
        if taken % 5 == 0 and made:
            dt = time.time() - t0
            print(f"  {taken} prompts taken ({n + 1}/{len(work)} scanned), {made} images, "
                  f"{made / dt:.2f} img/s", flush=True)
    savers.shutdown(wait=True)
    if scales:
        sp = os.path.join(os.path.dirname(out_dir.rstrip("/")), f"scales_{os.getpid()}.json")
        with open(sp + ".tmp", "w") as f:
            json.dump(scales, f, indent=0)
        os.replace(sp + ".tmp", sp)
    dt = time.time() - t0
    print(f"[sd3/{a.arm} shard {a.shard}] {made} images in {dt / 60:.1f} min = "
          f"{made / max(dt, 1e-9):.3f} img/s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--adapter", default=None,
                    help="ReDiverse-FT adapter: a rediverse_ft.safetensors file, a directory holding it "
                         "(or laid out like the Hub repository), or a trainer .pth (default: the Hub)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=BATCH,
                    help=f"images per pipeline call (default {BATCH}; outputs depend on it)")
    ap.add_argument("--num-samples", type=int, default=None,
                    help=f"samples per prompt (default {NUM_SAMPLES}; {PROMPT_NUM_SAMPLES} with --prompt)")
    ap.add_argument("--out-dir", default=None,
                    help="default <REDIVERSE_OUT>/sd3/<arm>/images, or <REDIVERSE_OUT>/sd3/prompt/<slug>/<arm> "
                         "with --prompt")
    # benchmark mode
    ap.add_argument("--prompts", default=D.GENEVAL_553,
                    help="prompt file, line i = prompt index i (default GenEval-553); for another file also "
                         "set --out-dir and, for the TF arms, --directions-dir")
    ap.add_argument("--prompt-ids", default=None, help="subset, e.g. '27' or '0-99' (default: all)")
    ap.add_argument("--noise", default=None, help="default <REDIVERSE_DATA>/sd3/noise_bank_553x100.npz")
    ap.add_argument("--directions-dir", default=None, help="default <REDIVERSE_DATA>/sd3/directions")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker takes the prompts whose index modulo --num-shards equals --shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--claim-dir", default=None, help="default <out-dir>/.claims")
    ap.add_argument("--no-claim", action="store_true")
    # prompt mode
    ap.add_argument("--prompt", default=None, help="generate for this text instead of the benchmark")
    ap.add_argument("--directions", default=None,
                    help="--prompt: direction file (default <REDIVERSE_OUT>/sd3/prompt/<slug>/directions.pt, "
                         "estimated if missing or of another prompt)")
    a = ap.parse_args()
    if a.num_samples is None:
        a.num_samples = PROMPT_NUM_SAMPLES if a.prompt is not None else NUM_SAMPLES

    import diffusers
    import transformers
    print(f"[gen] arm {a.arm}; torch {torch.__version__} diffusers {diffusers.__version__} "
          f"transformers {transformers.__version__} ({sys.executable})", flush=True)
    if a.batch_size != BATCH:
        print(f"[gen] batch size {a.batch_size} != {BATCH}: images are not bit-comparable with the "
              "reference sets", flush=True)
    if a.prompt is not None:
        directions = prompt_directions(a) if a.arm in TF_ARMS else None
        run_prompt_mode(a, Generator(a.arm, a.device, a.adapter), directions)
    else:
        run_benchmark(a)


if __name__ == "__main__":
    main()
