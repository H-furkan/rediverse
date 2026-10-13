"""SDXL / DMD2: model ids, pinned revisions and the ReDiverse recipe constants.

Teacher  stabilityai/stable-diffusion-xl-base-1.0 (50-step EulerDiscrete, CFG 8)
Student  tianweiy/DMD2 4-step UNet (LCMScheduler, timesteps 999/749/499/249, no CFG)
Site     up_blocks.0.attentions.0, the output of the first Transformer2DModel of the up path
"""
from __future__ import annotations

import os

from rediverse.data import FT_TARGET_FORWARD, TF_DIRECTION_STEP, data_root, output_root

BACKBONE = "sdxl"

# ------------------------------------------------------------------ models --
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
DMD2_ID = "tianweiy/DMD2"
DMD2_REVISION = "be22767697a1f3ca656b73c776e15fa335c86c6c"
DMD2_UNET_FP16 = "dmd2_sdxl_4step_unet_fp16.bin"     # student of every generated arm
DMD2_UNET_FP32 = "dmd2_sdxl_4step_unet.bin"          # ReDiverse-FT training initialisation

HEIGHT = WIDTH = 1024
LATENT_SHAPE = (4, 128, 128)
STUDENT_TIMESTEPS = [999, 749, 499, 249]             # DMD2's 4-step schedule (LCMScheduler)
TEACHER_STEPS = 50                                   # EulerDiscrete, the repository's scheduler config
TEACHER_CFG = 8.0

# ------------------------------------------------------------ ReDiverse site --
ANCHOR_MODULE = "up_blocks.0.attentions.0"           # collapsed block l, output [B, 1280, 32, 32]
SITE = "up0.attn0"                                   # site name stored in the direction files
TEACHER_TARGET_STEP = FT_TARGET_FORWARD[BACKBONE]    # tau = 7: the 7th teacher forward (t = 861)
TEACHER_GUIDANCE = 8.0                               # CFG on teacher forwards 1..tau-1 of the FT target
DIRECTION_STEP = TF_DIRECTION_STEP[BACKBONE]         # 7, conditional-only teacher rollout
N_DIRECTION_SAMPLES = 25

# ReDiverse-FT LoRA: attention q/k/v/out (self and cross) and the feed-forward linears of every
# transformer block up to and including the collapsed block (440 nn.Linear modules).
LORA_TARGET_REGEX = (
    r"^(down_blocks\.\d+\.attentions\.\d+|mid_block\.attentions\.0|up_blocks\.0\.attentions\.0)"
    r"\.transformer_blocks\.\d+\.(attn[12]\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))$")
N_LORA_MODULES = 440

# ------------------------------------------------------------- evaluation --
BATCH = 10                                           # images per pipeline call; fp16 results depend on it
NUM_SAMPLES = 100                                    # samples per GenEval prompt
NUM_PROMPTS = 553
PROMPT_SEED_INDEX = 0                                # --prompt mode: sample k follows the seed rules of
                                                     # prompt index 0 (noise k, TF alphas, direction latents)


# ------------------------------------------------------------------ paths --
def data_dir(*parts) -> str:
    """<REDIVERSE_DATA>/sdxl/..."""
    return os.path.join(data_root(), BACKBONE, *parts)


def output_dir(*parts) -> str:
    """<REDIVERSE_OUT>/sdxl/..."""
    return os.path.join(output_root(), BACKBONE, *parts)


def ctx_dir() -> str:
    """Training-caption embeddings, p{ci:03d}.pt for ci in 0..99 (sdxl/encode_captions.py)."""
    return data_dir("train_ctx")


def init_ckpt_path() -> str:
    """FastGen initialisation checkpoint holding the fp32 DMD2 UNet (sdxl/build_init_ckpt.py)."""
    return data_dir("dmd2_sdxl_init.pth")


def noise_bank_path() -> str:
    """Evaluation noise bank, GenEval-553 x 100 (sdxl/make_noise_bank.py)."""
    return data_dir("noise_bank_553x100.npz")


def directions_dir() -> str:
    """ReDiverse-TF direction files p{idx:03d}.pt (sdxl/estimate_directions.py)."""
    return data_dir("directions")


def train_dir() -> str:
    """ReDiverse-FT training run: FastGen checkpoints and adapters/rediverse_ft_{iteration:07d}.pth."""
    return output_dir("train")
