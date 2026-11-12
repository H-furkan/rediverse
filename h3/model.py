"""MiniMax-H3 model plumbing and sampler: teacher, Turbo student, text contexts, rollout, decoding.

MiniMax-H3 runs one transformer (50 blocks, hidden 5376) over one packed sequence
[text | audio | video]. Video and audio step down their own schedules (shift 12 and shift 3)
inside the same forward, so every step carries a per-row timestep plan. The Turbo student is a
rank-128 LoRA on the teacher (312 modules). Both are guidance-distilled: no CFG is used.

Conventions of this model:
  * timesteps run t = 0 (noise) -> t = 1 (clean); sigma = 1 - t, and step 1 is t = 0 at sigma = 1.
    Everything here is derived from MiniMaxH3Scheduler and asserted on sigma.
  * set_timesteps(n) gives n - 1 forwards: the 4-step student makes 3 transformer calls and the
    50-step teacher makes 49.
  * the video rows are the tail of the packed sequence, so a block's video activation is
    out[:, -NUM_VIDEO_ROWS:, :].
  * the audio stream starts from a frozen per-prompt noise tensor stored in the text context and
    shared by all clips and arms of that prompt; audio is never decoded.

The sampler is the diffusers 0.40.0 modular MiniMax-H3 denoise loop unrolled (row plan,
scheduler step in closed form, unpatchify, video decode).
"""
# The sampler (Schedule, pack/unpack, forward_at, euler_blend, sample) is adapted from the
# huggingface/diffusers modular MiniMax-H3 pipeline, diffusers 0.40.0 (Apache-2.0, Copyright 2026 The
# MiniMax and HuggingFace Teams).
from __future__ import annotations

import functools
import gc
import os

import numpy as np
import torch

from h3 import config as C
from rediverse import seeds

DTYPE = torch.bfloat16

# ---------------------------------------------------------------- geometry --
HEIGHT, WIDTH, NUM_FRAMES = C.HEIGHT, C.WIDTH, C.NUM_FRAMES
LATENT_C = 24
LATENT_T = 5 * ((NUM_FRAMES - 5) // 17) + 2             # 5 latents per 17 frames, +2: 37
LATENT_H, LATENT_W = HEIGHT // 16, WIDTH // 16          # VAE /16 spatial: 48 x 84
PATCH = (1, 2, 2)
ROWS_PER_FRAME = (LATENT_H // 2) * (LATENT_W // 2)      # 1008
NUM_VIDEO_ROWS = LATENT_T * ROWS_PER_FRAME              # 37296
NUM_AUDIO_LATENTS = round(NUM_FRAMES / 24 * 40)         # 207
NUM_AUDIO_ROWS = NUM_AUDIO_LATENTS * 2                  # 414, stereo, channel-major
BLOCK = C.ANCHOR_BLOCK
assert NUM_FRAMES % 17 == 5 and (LATENT_T, NUM_VIDEO_ROWS, NUM_AUDIO_ROWS) == (37, 37296, 414)


def latent_shape() -> tuple:
    return (LATENT_C, LATENT_T, LATENT_H, LATENT_W)


def eval_latent(prompt_idx: int, k: int) -> torch.Tensor:
    """Initial video latent of evaluation clip k of VBench prompt prompt_idx (CPU float32)."""
    return seeds.cpu_randn(seeds.h3_eval_seed(prompt_idx, k), latent_shape())


def train_latent(prompt_idx: int, i: int) -> torch.Tensor:
    """Latent i of GenEval prompt prompt_idx in the ReDiverse-FT target bank (CPU float32)."""
    return seeds.cpu_randn(seeds.h3_train_latent_seed(prompt_idx, i), latent_shape())


def direction_latent(prompt_idx: int, j: int) -> torch.Tensor:
    """ReDiverse-TF estimation latent j (100..124) of VBench prompt prompt_idx (CPU float32)."""
    return seeds.cpu_randn(seeds.h3_direction_seed(prompt_idx, j), latent_shape())


# ------------------------------------------------------------------ models --

@functools.lru_cache(maxsize=None)
def snapshot(*subfolders: str) -> str:
    """Local directory of MiniMaxAI/MiniMax-H3 at the pinned revision, holding `subfolders`
    (fetched into the Hugging Face cache on first use; HF_HUB_OFFLINE=1 reads the cache only)."""
    from huggingface_hub import constants, snapshot_download
    return snapshot_download(C.TEACHER_REPO, revision=C.TEACHER_REVISION,
                             allow_patterns=[f"{s}/*" for s in subfolders] + ["model_index.json"],
                             local_files_only=constants.HF_HUB_OFFLINE)


def student_lora_file() -> str:
    """The Turbo 4-step 768p LoRA file at the pinned revision."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C.STUDENT_REPO, C.STUDENT_FILE, revision=C.STUDENT_REVISION)


def load_arm(arm: str, device) -> torch.nn.Module:
    """The teacher transformer, or the student = teacher + Turbo LoRA bound as a peft adapter."""
    from diffusers import MiniMaxH3Transformer3DModel

    assert arm in ("teacher", "student"), arm
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        snapshot("transformer"), subfolder="transformer", torch_dtype=DTYPE)
    assert model.config.num_layers == C.NUM_LAYERS and model.config.hidden_size == C.HIDDEN
    model.to(device)

    if arm == "student":
        from diffusers.loaders.lora_pipeline import MiniMaxH3LoraLoaderMixin
        from peft.tuners.tuners_utils import BaseTunerLayer

        path = student_lora_file()
        # lora_state_dict converts the file to 'transformer.'-prefixed diffusers keys and reads
        # the alpha from the file metadata; binding the raw file would match no module.
        sd, metadata = MiniMaxH3LoraLoaderMixin.lora_state_dict(
            os.path.dirname(path), weight_name=os.path.basename(path),
            local_files_only=True, return_lora_metadata=True)
        assert sd and all(k.startswith("transformer.") for k in sd), \
            f"unexpected converted-LoRA key layout: {sorted(sd)[:3]}"
        model.load_lora_adapter(sd, prefix="transformer", metadata=metadata)
        bound = sum(1 for m in model.modules() if isinstance(m, BaseTunerLayer))
        assert bound == C.STUDENT_LORA_MODULES, \
            f"Turbo LoRA bound {bound} modules, expected {C.STUDENT_LORA_MODULES}"
        print(f"[h3/student] {os.path.basename(path)}: {bound} LoRA modules bound", flush=True)

    model.eval()
    model.requires_grad_(False)
    return model


def free():
    """Collect garbage and return cached GPU memory (after the caller dropped its references)."""
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------- text ctx --

def _first_step_plan(num_text_tokens: int, layout: dict):
    """(timestep, timestep_indices, scheduler summary) of denoising step 1, built by the
    pipeline's own scheduler and row-plan code; asserted on sigma, never on t."""
    from diffusers import MiniMaxH3Scheduler
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

    snap = snapshot("scheduler", "audio_scheduler")
    pairs = {}
    for sub in ("scheduler", "audio_scheduler"):
        sch = MiniMaxH3Scheduler.from_pretrained(snap, subfolder=sub)
        first = {}
        for steps in (4, 50):   # the first-step pair is step-count-invariant
            sch.set_timesteps(steps)
            sigma0, t0 = float(sch.sigmas[0]), float(sch.timesteps[0])
            assert abs(sigma0 - 1.0) < 1e-6, f"{sub}: first sigma {sigma0} != 1"
            first[steps] = (t0, sigma0)
        assert first[4] == first[50]
        pairs[sub] = dict(cls=type(sch).__name__, shift=float(sch.shift),
                          sigmas3=[float(s) for s in sch.sigmas[:3]],
                          timesteps3=[float(t) for t in sch.timesteps[:3]],
                          t0=first[4][0], sigma0=first[4][1])

    timestep, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        layout["video_indices"], layout["audio_indices"], 0, 0,
        num_text_tokens,
        pairs["scheduler"]["t0"], pairs["audio_scheduler"]["t0"],
        max(pairs["scheduler"]["t0"], 0.999), 1.0)
    return timestep, timestep_indices, pairs


def _build_layout(num_text_tokens: int) -> dict:
    """Packed [text | audio | video] layout of a text-to-video+audio sample."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG)

    tags = torch.full((num_text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    (position_ids, token_tags, video_indices, audio_indices, text_indices,
     n_cond_v, n_cond_a) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        tags, LATENT_T, LATENT_H, LATENT_W, NUM_AUDIO_LATENTS, PATCH,
        2, MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG, ())
    assert n_cond_v == 0 and n_cond_a == 0
    S = position_ids.shape[0]
    assert S == num_text_tokens + NUM_AUDIO_ROWS + NUM_VIDEO_ROWS
    # the video rows must be the tail of the sequence
    assert torch.equal(video_indices, torch.arange(S - NUM_VIDEO_ROWS, S))
    return dict(position_ids=position_ids, token_tags=token_tags,
                video_indices=video_indices, audio_indices=audio_indices,
                text_indices=text_indices)


class _TextOnlyQwen3VLProcessor:
    """Stand-in exposing Qwen3VLProcessor.create_mm_token_type_ids, the only member diffusers'
    get_qwen3vl_prompt_embeds uses. The full processor needs torchvision for its video
    sub-processor, which a text-only encode never runs. Token ids are resolved as
    Qwen3VLProcessor.__init__ resolves them."""

    def __init__(self, tokenizer):
        from transformers import Qwen3VLProcessor
        self.create_mm_token_type_ids = Qwen3VLProcessor.create_mm_token_type_ids.__get__(self)
        image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.image_token_ids = [getattr(tokenizer, "image_token_id", None)
                                or tokenizer.convert_tokens_to_ids(image_token)]
        self.video_token_ids = [getattr(tokenizer, "video_token_id", None)
                                or tokenizer.convert_tokens_to_ids(video_token)]
        self.audio_token_ids = [None]


class TextEncoder:
    """Qwen3-VL conditioner (about 66 GB): runs alone on its GPU, before any transformer load.
    The prompt is tokenized verbatim (no chat template, no special tokens) and conditioned on
    hidden state 50."""

    def __init__(self, device):
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration
        snap = snapshot("tokenizer", "text_encoder")
        self.device = device
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(snap, subfolder="tokenizer")
        self.processor = _TextOnlyQwen3VLProcessor(self.tokenizer)
        self.enc = Qwen3VLForConditionalGeneration.from_pretrained(
            snap, subfolder="text_encoder", torch_dtype=DTYPE)
        self.enc.to(device)
        self.enc.eval()
        self.enc.requires_grad_(False)

    def context(self, prompt_idx: int, text: str, audio_seed: int) -> dict:
        """Text context of one prompt: Qwen3-VL embeddings, the packed layout, the step-1 row
        plan and the frozen audio-row noise (seed `audio_seed`)."""
        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        mm = self.processor.create_mm_token_type_ids([token_ids])[0]
        assert set(mm) == {0}, f"non-text mm token type ids {sorted(set(mm))}"
        with torch.no_grad():
            emb = get_qwen3vl_prompt_embeds(
                self.enc, self.processor, token_ids, {}, text_encoder_layer=C.TEXT_ENCODER_LAYER,
                device=self.device, dtype=self.enc.dtype)
        layout = _build_layout(len(token_ids))
        timestep, timestep_indices, pairs = _first_step_plan(len(token_ids), layout)
        audio_rows = seeds.h3_audio_noise(audio_seed, NUM_AUDIO_ROWS)
        return dict(prompt_idx=prompt_idx, prompt=text,
                    prompt_embeds=emb.cpu(), audio_rows=audio_rows,
                    timestep=timestep, timestep_indices=timestep_indices,
                    schedule=pairs, **layout)

    def close(self):
        self.enc = None
        free()


def load_ctx(path: str) -> dict:
    return torch.load(path, map_location="cpu")


# ----------------------------------------------------------------- sampler --

class Schedule:
    """Video and audio schedulers set to `steps`, plus the per-step row plan."""

    def __init__(self, steps: int, ctx: dict, device):
        from diffusers import MiniMaxH3Scheduler
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
        snap = snapshot("scheduler", "audio_scheduler")
        self.v = MiniMaxH3Scheduler.from_pretrained(snap, subfolder="scheduler")
        self.a = MiniMaxH3Scheduler.from_pretrained(snap, subfolder="audio_scheduler")
        self.v.set_timesteps(steps, device=device)
        self.a.set_timesteps(steps, device=device)
        # set_timesteps(n) yields n - 1 forwards: timesteps has n - 1 entries, sigmas n (last 0)
        self.steps = steps
        self.n_fwd = int(len(self.v.timesteps))
        self.sigmas = [float(s) for s in self.v.sigmas]          # len n_fwd + 1
        assert len(self.sigmas) == self.n_fwd + 1 and abs(self.sigmas[-1]) < 1e-9
        self.timesteps = self.v.timesteps
        self.audio_timesteps = self.a.timesteps
        n_txt = int(ctx["text_indices"].numel())
        self.plan = [
            tuple(t.to(device) for t in MiniMaxH3SetTimestepsStep.build_row_timesteps(
                ctx["video_indices"], ctx["audio_indices"], 0, 0, n_txt,
                float(tv), float(ta), max(float(tv), 0.999), 1.0))
            for tv, ta in zip(self.timesteps, self.audio_timesteps)]
        assert abs(self.sigmas[0] - 1.0) < 1e-6, "first sigma must be 1"


def dev_ctx(ctx: dict, device) -> dict:
    d = {"prompt_embeds": ctx["prompt_embeds"].to(device, DTYPE),
         "audio_rows": ctx["audio_rows"].to(device, DTYPE)}
    for k in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"):
        d[k] = ctx[k].to(device)
    return d


def pack(x: torch.Tensor) -> torch.Tensor:
    """[B, 24, 37, 48, 84] -> [B, 37296, C * prod(PATCH)], the pipeline's own patchify."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    return patchify_video_latents(x, PATCH).reshape(x.shape[0], NUM_VIDEO_ROWS, -1)


def unpack(rows: torch.Tensor) -> torch.Tensor:
    """Inverse of pack: [B, 37296, C * prod(PATCH)] -> [B, 24, 37, 48, 84]."""
    pt, ph, pw = PATCH
    B = rows.shape[0]
    r = rows.reshape(B, LATENT_T // pt, LATENT_H // ph, LATENT_W // pw, LATENT_C, pt, ph, pw)
    r = r.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return r.reshape(B, LATENT_C, LATENT_T, LATENT_H, LATENT_W).contiguous()


def forward_at(model, rows, audio, d, sched: Schedule, k: int):
    """One transformer call at step index k (0-based) with that step's row plan."""
    B = rows.shape[0]
    uniq, idx = sched.plan[k]
    v_pred, a_pred = model(
        hidden_states=rows, audio_hidden_states=audio.expand(B, -1, -1) if audio.shape[0] == 1 else audio,
        encoder_hidden_states=d["prompt_embeds"].expand(B, -1, -1),
        timestep=uniq, timestep_indices=idx,
        token_tags=d["token_tags"], position_ids=d["position_ids"],
        video_indices=d["video_indices"], audio_indices=d["audio_indices"],
        text_indices=d["text_indices"], return_dict=False)
    return v_pred, a_pred


def x0_hat(sample, model_output, t: float):
    """The scheduler's clean-sample estimate x0 = x + (1 - t) * v."""
    return (sample.float() + (1.0 - t) * model_output.float())


def euler_blend(sample, model_output, t: float, sigma: float, sigma_next: float):
    """MiniMaxH3Scheduler.step in closed form: x_next = r * x + (1 - r) * x0, r = sigma_next / sigma,
    computed in float32 and cast back."""
    x0 = x0_hat(sample, model_output, t)
    r = sigma_next / sigma
    return (r * sample.float() + (1.0 - r) * x0).to(sample.dtype)


@torch.no_grad()
def sample(model, x0: torch.Tensor, ctx: dict, device, steps: int, step_hook=None) -> torch.Tensor:
    """Full rollout from the initial video latent x0 [B, 24, 37, 48, 84]; returns the final video
    latent (unpacked, normalised space). step_hook(k) runs before forward k (0-based): the
    ReDiverse gating hangs off it. Audio starts from the context's frozen audio noise and steps
    down its own schedule alongside the video."""
    d = dev_ctx(ctx, device)
    sched = Schedule(steps, ctx, device)
    rows = pack(x0.to(device, DTYPE))
    B = rows.shape[0]
    audio = d["audio_rows"].unsqueeze(0).expand(B, -1, -1).clone()
    for k in range(sched.n_fwd):
        if step_hook is not None:
            step_hook(k)
        v_pred, a_pred = forward_at(model, rows, audio, d, sched, k)
        rows = euler_blend(rows, v_pred, float(sched.timesteps[k]),
                           sched.sigmas[k], sched.sigmas[k + 1])
        audio = euler_blend(audio, a_pred, float(sched.audio_timesteps[k]),
                            float(sched.a.sigmas[k]), float(sched.a.sigmas[k + 1]))
    return unpack(rows)


@torch.no_grad()
def teacher_states(teacher, x0, ctx, device, ks):
    """{k: transformer_blocks[45] video rows at the k-th teacher forward (after k - 1 steps)}.
    The ReDiverse-FT targets and the ReDiverse-TF samples are both this tensor at k = tau."""
    ks = sorted(set(int(k) for k in ks))
    d = dev_ctx(ctx, device)
    sched = Schedule(C.STEPS["teacher"], ctx, device)
    grab, cur = {}, {"k": 0}
    blk = teacher.transformer_blocks[BLOCK]

    def hook(_m, _i, out):
        o = out[0] if isinstance(out, tuple) else out
        if cur["k"] + 1 in ks:
            grab[cur["k"] + 1] = o[:, -NUM_VIDEO_ROWS:, :].detach().clone()
        return None

    h = blk.register_forward_hook(hook)
    rows = pack(x0.to(device, DTYPE))
    audio = d["audio_rows"].unsqueeze(0).clone()
    try:
        for k in range(max(ks)):
            cur["k"] = k
            v_pred, a_pred = forward_at(teacher, rows, audio, d, sched, k)
            if k < max(ks) - 1:
                rows = euler_blend(rows, v_pred, float(sched.timesteps[k]),
                                   sched.sigmas[k], sched.sigmas[k + 1])
                audio = euler_blend(audio, a_pred, float(sched.audio_timesteps[k]),
                                    float(sched.a.sigmas[k]), float(sched.a.sigmas[k + 1]))
    finally:
        h.remove()
    assert max(ks) <= sched.n_fwd, f"k={max(ks)} but the teacher has only {sched.n_fwd} forwards"
    assert set(grab) == set(ks), f"got {sorted(grab)} wanted {ks}"
    return grab, sched.sigmas


class Decoder:
    """Video decode: de-normalise, fp16 autocast VAE decode, ImageNet un-normalise, clamp.
    Returns uint8 [T, H, W, 3] for the first sample of the batch."""

    def __init__(self, device):
        from diffusers import AutoencoderKLMiniMaxH3
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(
            snapshot("vae"), subfolder="vae", torch_dtype=torch.float16).to(device).eval()
        self.device = device
        c = self.vae.config
        self.mean = torch.tensor(c.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(c.latents_std, device=device).view(1, -1, 1, 1, 1)
        self.pm = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, -1, 1, 1, 1)
        self.ps = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, lat: torch.Tensor) -> np.ndarray:
        z = lat.float() * self.std + self.mean
        with torch.autocast("cuda", dtype=torch.float16):
            v = self.vae.decode(z.to(torch.float16), return_dict=False)[0]
        v = (v.float() * self.ps + self.pm).clamp(0, 1)          # [B, 3, T, H, W]
        v = (v[0].permute(1, 2, 3, 0) * 255).round().byte().cpu().numpy()
        assert np.isfinite(v).all()
        return v
