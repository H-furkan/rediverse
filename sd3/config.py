"""SD3-Medium / TDM: model ids, pinned revisions and the ReDiverse recipe constants.

Teacher  stabilityai/stable-diffusion-3-medium-diffusers (28-step FlowMatchEuler, shift 3, CFG 7)
Student  the same transformer with the TDM 4-step LoRA (Luo-Yihong/TDM_sd3_lora) at scale 0.125,
         4 steps, guidance 1.0 (TDM is CFG-distilled)
Site     transformer_blocks.12, image stream (the second output of the MMDiT block)
"""
from __future__ import annotations

import os

import torch

from rediverse.data import FT_TARGET_FORWARD, TF_DIRECTION_STEP, data_root, output_root

BACKBONE = "sd3"

# ------------------------------------------------------------------ models --
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
MODEL_REVISION = "ea42f8cef0f178587cf766dc8129abd379c90671"
LORA_ID = "Luo-Yihong/TDM_sd3_lora"
LORA_REVISION = "bd01e7ccdcf2ca30f863ae18a09e36d057ac8bc8"
LORA_WEIGHT_NAME = "tdm_sd3_lora.safetensors"
TDM_SCALE = 0.125                     # TDM model card: set_adapters(["tdm"], [0.125])
TDM_ADAPTER = "tdm"                   # frozen; defines the student
FT_ADAPTER = "anchor"                 # peft name of the ReDiverse-FT adapter
DTYPE = torch.bfloat16

STEPS = dict(teacher=28, student=4)   # each model's own sampling schedule
GUIDANCE = dict(teacher=7.0, student=1.0)
N_BLOCKS = 24
LATENT_SHAPE = (16, 128, 128)         # 1024 x 1024
N_IMG_TOKENS = 4096                   # (128 / 2)^2 after the 2x2 patchify
D_MODEL = 1536
D_BLOCK = N_IMG_TOKENS * D_MODEL      # image stream of one block, flattened

# ------------------------------------------------------------ ReDiverse site --
ANCHOR_BLOCK = 12                     # collapsed block l (image stream)
TEACHER_TARGET_STEP = FT_TARGET_FORWARD[BACKBONE]     # tau = 9: lands at the student's sigmas[1]
TEACHER_GUIDANCE = 7.0                # CFG on teacher forwards 1..tau-1 of the FT target rollout
DIRECTION_STEP = TF_DIRECTION_STEP[BACKBONE]          # 9, conditional-only teacher rollout

# ReDiverse-FT LoRA: blocks 0..12, attention and feed-forward projections of both MMDiT streams.
LORA_LAST_BLOCK = ANCHOR_BLOCK
LORA_SUFFIXES = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
)
# In the collapsed block itself these four write only the text-stream output, which the
# image-stream loss never reads; they are left out (156 -> 152 modules).
TEXT_ONLY_AT_ANCHOR = ("attn.add_q_proj", "attn.to_add_out",
                       "ff_context.net.0.proj", "ff_context.net.2")
N_LORA_MODULES = 152

# ------------------------------------------------------------- evaluation --
BATCH = 10                            # images per pipeline call; bf16 results depend on it
NUM_SAMPLES = 100                     # samples per GenEval prompt
PROMPT_NUM_SAMPLES = 4                # default samples of a --prompt run
PROMPT_SEED_INDEX = 0                 # --prompt mode: sample k follows the seed rules of prompt index 0
                                      # (noise k, TF coefficients, direction latents)


# ------------------------------------------------------------------ paths --
def data_dir(*parts) -> str:
    """<REDIVERSE_DATA>/sd3/..."""
    return os.path.join(data_root(), BACKBONE, *parts)


def output_dir(*parts) -> str:
    """<REDIVERSE_OUT>/sd3/..."""
    return os.path.join(output_root(), BACKBONE, *parts)


def ctx_dir() -> str:
    """Training-caption embeddings, p{ci:06d}.pt for ci in 0..99."""
    return data_dir("ctx")


def uncond_path() -> str:
    """Empty-prompt embedding (the pipeline's negative branch) for the guided target."""
    return data_dir("uncond.pt")


def noise_bank_path() -> str:
    """Frozen evaluation latents, 553 prompts x 100 samples."""
    return data_dir("noise_bank_553x100.npz")


def directions_dir() -> str:
    """ReDiverse-TF direction files p{idx:03d}.pt for the GenEval prompts."""
    return data_dir("directions")


def geneval_ctx_dir() -> str:
    """GenEval prompt embeddings used for direction estimation, p{idx:03d}.pt."""
    return data_dir("geneval_ctx")
