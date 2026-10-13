"""Every seed rule of the ReDiverse pipeline.

Each training pair, evaluation sample, direction-estimation latent and ReDiverse-TF
coefficient is drawn from its own explicitly seeded generator, so any single one of
them can be regenerated without replaying the rest. The rules, by stage:

ReDiverse-FT training, SDXL / SD3 / Wan 2.1 (caption ci in 0..99, noise ni in 0..15)
    seed   1_000_000 + (10000 + ci) * 100003 + ni                        [train_noise_seed]
    draw   torch.randn(shape, generator=torch.Generator("cpu")), float32, cast by the caller
    shape  SDXL (4,128,128); SD3 (16,128,128); Wan (1,16,21,60,104)

ReDiverse-FT training, MiniMax-H3 (GenEval index p in prompts/h3_train_geneval_ids.txt, i in 0..19)
    seed   1_000_000 + p * 100003 + i                                    [h3_train_latent_seed]
    draw   torch.randn((24,37,48,84), generator=CPU), float32
    audio  frozen audio-row noise, seed 424243 + p, torch.randn(414, 32)  [h3_train_audio_seed]

Image evaluation, SDXL / SD3 (GenEval index i in 0..552, sample s in 0..99), all five arms
    seed   i * 1000 + s, on a CUDA generator                             [image_eval_seed]
    draw   SDXL torch.randn((1,4,128,128), device="cuda", dtype=float16)
           SD3  torch.randn((1,16,128,128), device="cuda", dtype=bfloat16)
    The draws are frozen into an .npz noise bank by the bank builders of the sdxl/ and sd3/
    packages (one key per prompt; the SD3 bank stores the bf16 bit pattern as int16).

Video evaluation (VBench index i in 0..299, clip k in 0..9), all five arms
    Wan    1_000_000 + i * 100003 + k, torch.randn((1,16,21,60,104))    [wan_eval_seed]
    H3     1_000_000 + (10000 + i) * 100003 + k, torch.randn((24,37,48,84))  [h3_eval_seed]
    audio  H3 frozen audio-row noise, seed 434243 + i (= 424243 + 10000 + i)  [h3_eval_audio_seed]

ReDiverse-TF direction estimation (25 teacher latents per prompt, j in 100..124, CPU, float32)
    SDXL / SD3  i * 1000 + j                                             [image_direction_seed]
    Wan         1_000_000 + i * 100003 + j                               [wan_direction_seed]
    H3          1_000_000 + (10000 + i) * 100003 + j                     [h3_direction_seed]
    j starts at 100, so the estimation latents never coincide with an evaluation sample.

ReDiverse-TF coefficients (all backbones; arms rediverse_tf and rediverse)          [tf_alphas]
    alpha  torch.empty(8).uniform_(-2, 2, generator=torch.Generator("cpu").manual_seed(
               7_000_000 + i * 1009 + s))
    i is the prompt index and s the sample (image) or clip (video) index, so a given sample
    receives identical coefficients in the rediverse_tf and rediverse arms.
"""
from __future__ import annotations

import torch

SEED0 = 1_000_000                 # base of the per-(prompt, sample) rule
PROMPT_STRIDE = 100_003           # prompt stride of the per-(prompt, sample) rule
CTX_OFFSET = 10_000               # index offset of the training captions and of the H3 VBench prompts

IMAGE_EVAL_STRIDE = 1000          # image evaluation bank: seed = i * 1000 + s

DIRECTION_SAMPLE_BASE = 100       # direction-estimation latents use sample indices 100..124
N_DIRECTION_SAMPLES = 25

H3_AUDIO_SEED0 = 424_243          # frozen per-prompt H3 audio-row noise
H3_AUDIO_CHANNELS = 32

TF_ALPHA_SEED0 = 7_000_000        # ReDiverse-TF coefficients: 7_000_000 + i * 1009 + s
TF_ALPHA_STRIDE = 1009
TF_N_DIR = 8                      # K directions injected per sample
TF_BAND = 2.0                     # alpha ~ U[-TF_BAND, TF_BAND]


# ------------------------------------------------------------- generic rules --

def prompt_seed(prompt_idx: int, i: int) -> int:
    """SEED0 + prompt_idx * 100003 + i, the per-(prompt, sample) rule behind every training
    noise and every video evaluation or direction latent."""
    return SEED0 + prompt_idx * PROMPT_STRIDE + int(i)


def cpu_randn(seed: int, shape) -> torch.Tensor:
    """Unit Gaussian of `shape` from a CPU generator seeded with `seed`, float32."""
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(tuple(shape), generator=g)


def direction_sample_indices(n: int = N_DIRECTION_SAMPLES) -> list[int]:
    """Sample indices of the ReDiverse-TF estimation latents: 100, 101, ..., 100 + n - 1."""
    return list(range(DIRECTION_SAMPLE_BASE, DIRECTION_SAMPLE_BASE + n))


# ---------------------------------------------------------- ReDiverse-FT data --

def train_noise_seed(ci: int, ni: int) -> int:
    """Training noise ni of caption ci (SDXL, SD3, Wan): prompt_seed(10000 + ci, ni)."""
    return SEED0 + (CTX_OFFSET + int(ci)) * PROMPT_STRIDE + int(ni)


def train_noise(ci: int, ni: int, shape, dtype=torch.float32) -> torch.Tensor:
    """The paired initial noise of (ci, ni): CPU float32 draw, then cast to `dtype`.
    Student step 1 and the teacher rollout start from this same tensor."""
    g = torch.Generator("cpu").manual_seed(train_noise_seed(ci, ni))
    return torch.randn(tuple(shape), generator=g).to(dtype)


def train_noise_batch(pairs, shape, dtype=torch.float32) -> torch.Tensor:
    """Stack of train_noise over [(ci, ni), ...]."""
    return torch.stack([train_noise(ci, ni, shape, dtype) for ci, ni in pairs])


def h3_train_latent_seed(p: int, i: int) -> int:
    """MiniMax-H3 ReDiverse-FT target bank: latent i of GenEval prompt p, prompt_seed(p, i)."""
    return prompt_seed(p, i)


def h3_train_audio_seed(p: int) -> int:
    """Frozen audio-row noise of the H3 training context of GenEval prompt p: 424243 + p."""
    return H3_AUDIO_SEED0 + p


# ---------------------------------------------------------- image evaluation --

def image_eval_seed(i: int, s: int) -> int:
    """SDXL / SD3 evaluation sample s of GenEval prompt i: i * 1000 + s."""
    return i * IMAGE_EVAL_STRIDE + s


def image_eval_generator(i: int, s: int, device: str = "cuda") -> torch.Generator:
    """The per-sample generator of the image evaluation rule. For SDXL it also drives the
    student's between-step re-noising; with bank latents one initial-noise draw is consumed
    from it first, so the re-noise stream starts at the same offset as the seeded path."""
    return torch.Generator(device).manual_seed(image_eval_seed(i, s))


def image_eval_noise(i: int, s: int, shape, dtype, device: str = "cuda") -> torch.Tensor:
    """One evaluation draw, as diffusers' randn_tensor makes it: shape (1, C, 128, 128) on
    `device`, float16 for SDXL and bfloat16 for SD3."""
    g = image_eval_generator(i, s, device)
    return torch.randn(tuple(shape), generator=g, device=device, dtype=dtype)


def image_direction_seed(i: int, j: int) -> int:
    """SDXL / SD3 ReDiverse-TF estimation latent j (100..124) of GenEval prompt i: i * 1000 + j,
    drawn on CPU in float32 as torch.randn(1, C, 128, 128)."""
    return i * IMAGE_EVAL_STRIDE + j


# ---------------------------------------------------------- video evaluation --

def wan_eval_seed(i: int, k: int) -> int:
    """Wan 2.1 evaluation clip k of VBench prompt i: 1_000_000 + i * 100003 + k."""
    return prompt_seed(i, k)


def wan_direction_seed(i: int, j: int) -> int:
    """Wan 2.1 ReDiverse-TF estimation latent j (100..124) of VBench prompt i."""
    return prompt_seed(i, j)


def wan_latent(seed: int, frames: int = 81) -> torch.Tensor:
    """Wan 2.1 initial latent [1, 16, (frames - 1) // 4 + 1, 60, 104] (832x480), CPU float32."""
    return cpu_randn(seed, (1, 16, (frames - 1) // 4 + 1, 60, 104))


def h3_eval_seed(i: int, k: int) -> int:
    """MiniMax-H3 evaluation clip k of VBench prompt i: 1_000_000 + (10000 + i) * 100003 + k."""
    return prompt_seed(CTX_OFFSET + i, k)


def h3_direction_seed(i: int, j: int) -> int:
    """MiniMax-H3 ReDiverse-TF estimation latent j (100..124) of VBench prompt i."""
    return prompt_seed(CTX_OFFSET + i, j)


def h3_eval_audio_seed(i: int) -> int:
    """Frozen audio-row noise of the H3 evaluation context of VBench prompt i: 434243 + i."""
    return H3_AUDIO_SEED0 + CTX_OFFSET + i


def h3_audio_noise(seed: int, n_rows: int) -> torch.Tensor:
    """H3 audio-row noise [n_rows, 32] float32 (n_rows = 414 at 124 frames)."""
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn(n_rows, H3_AUDIO_CHANNELS, generator=g, dtype=torch.float32)


# ----------------------------------------------------- ReDiverse-TF coefficients --

def tf_alpha_seed(prompt_idx: int, sample_idx: int) -> int:
    """7_000_000 + prompt_idx * 1009 + sample_idx."""
    return TF_ALPHA_SEED0 + prompt_idx * TF_ALPHA_STRIDE + sample_idx


def tf_alphas(prompt_idx: int, sample_idx: int, k: int = TF_N_DIR, a: float = TF_BAND) -> torch.Tensor:
    """alpha_{i,k} ~ U[-a, a], drawn independently per sample and per direction (paper
    Sec. 4.2): [k] float32 from a CPU generator seeded with tf_alpha_seed."""
    g = torch.Generator("cpu").manual_seed(tf_alpha_seed(prompt_idx, sample_idx))
    return torch.empty(k).uniform_(-float(a), float(a), generator=g)


def tf_alphas_batch(prompt_idx: int, sample_indices, k: int = TF_N_DIR, a: float = TF_BAND) -> torch.Tensor:
    """[len(sample_indices), k] stack of tf_alphas, one row per sample of an image batch."""
    return torch.stack([tf_alphas(prompt_idx, s, k, a) for s in sample_indices])


def alpha_record(alphas: torch.Tensor) -> list[float]:
    """The coefficients of one sample rounded to 5 decimals, as recorded in scales.json."""
    return [round(float(x), 5) for x in alphas]
