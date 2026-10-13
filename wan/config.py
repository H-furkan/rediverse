"""Wan 2.1 T2V-14B / lightx2v StepDistill-CfgDistill: model ids, pinned revisions, geometry and recipe.

Teacher   Wan-AI/Wan2.1-T2V-14B-Diffusers, UniPC 50 steps (repo scheduler, flow shift 3), CFG 5
Student   lightx2v 4-step step- and CFG-distilled DiT, sigmas [1, .9375, .8333, .625, 0], guidance 1
Site      blocks.28 output (all 32,760 video tokens of an 81-frame 832x480 clip)
FT target blocks.28 at forward 13 of the CFG-distilled teacher (base teacher + Perflow-Shuai's
          CFG5->CFG1 LoRA) on a UniPC-50 schedule with flow shift 5, conditional only
TF        blocks.28 at forward 8 of the base teacher on the repo schedule (flow shift 3),
          conditional only

Every CLI default of the wan package reads its value from here, so running the scripts with no
extra flags reproduces the released target bank, adapter, direction banks and evaluation clips.
"""
from __future__ import annotations

import os

from rediverse import data as _data

BACKBONE = "wan2.1"               # Hub subfolder and data/ / outputs/ directory name

# ------------------------------------------------------------------- models --
TEACHER_REPO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
TEACHER_REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
STUDENT_REPO = "lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v"
STUDENT_REVISION = "14734f864b35505d8abf94dcca65b96642fa618f"
STUDENT_FILE = "distill_models/distill_model.safetensors"     # native Wan keys, 1095 bf16 tensors
STUDENT_N_TENSORS = 1095
CFGD_REPO = "Perflow-Shuai/Wan2.1-T2V-14B-CFG5-to-CFG1-50Step-LoRA-r128-iter500"
CFGD_REVISION = "11730fdc311ca638201e54cb9b37c035d4773688"
CFGD_FILE = "adapter_model.safetensors"

N_BLOCKS = 40
DIM = 5120
TEXT_LEN, TEXT_DIM = 512, 4096    # UMT5 context: max_sequence_length 512, padding zeroed

# ----------------------------------------------------------------- geometry --
FRAMES = 81
HEIGHT, WIDTH = 480, 832
FPS = 16
LATENT_SHAPE = (16, (FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8)        # (16, 21, 60, 104)
TOKENS = LATENT_SHAPE[1] * (LATENT_SHAPE[2] // 2) * (LATENT_SHAPE[3] // 2)  # 32,760

# ----------------------------------------------------------------- sampling --
TEACHER_STEPS = 50
TEACHER_GUIDANCE = 5.0            # evaluation teacher: CFG 5 against NEG_PROMPT (wan.model)
STUDENT_STEPS = 4
STUDENT_SHIFT = 5.0

# --------------------------------------------------------------- ReDiverse --
ANCHOR_BLOCK = 28                 # collapsed block l
CFGD_FLOW_SHIFT = 5.0             # sampling contract of the CFG-distilled teacher (its model card)
FT_TARGET_FORWARD = _data.FT_TARGET_FORWARD["wan"]     # 13: sigma 0.9407 -> 0.9344, the student's step 1 lands at 0.9375
TF_DIRECTION_FORWARD = _data.TF_DIRECTION_STEP["wan"]  # 8: base teacher, repo schedule (sigma_out 0.9404)
# Timestep fed to the student at its first step during training: the base teacher's first
# UniPC-50 timestep, int64(1000 * (1 - 1e-6)) = 999. Sampling feeds sigma * 1000 = 1000.0.
# Kept in float32: 999 is not representable in bfloat16.
TRAIN_TIMESTEP = 999.0

# ReDiverse-FT
LORA_RANK = _data.LORA_RANK       # 32
LORA_ALPHA = _data.LORA_ALPHA     # 32
LORA_LAST_BLOCK = ANCHOR_BLOCK    # LoRA on blocks 0..28
LORA_ATTENTIONS = ("attn1", "attn2")
LORA_PROJECTIONS = ("to_q", "to_k", "to_v", "to_out.0")
LORA_FFN = ("ffn.net.0.proj", "ffn.net.2")
N_LORA_MODULES = 290              # 29 blocks x (8 attention + 2 feed-forward)
NOISES_PER_CAPTION = _data.NOISES_PER_CAPTION          # 16 bank latents per caption
CHUNK = 2                         # noises of one caption per forward/backward
SAVE_EVERY = 50
LOG_EVERY = 5
TRAIN_SEED = 0

# ReDiverse-TF
N_DIRECTION_SAMPLES = 25
N_DIR = 8
TF_BAND = 2.0

# Evaluation
CLIPS_PER_PROMPT = 10
ARMS = ("teacher", "student", "rediverse_ft", "rediverse_tf", "rediverse")
FT_ARMS = ("rediverse_ft", "rediverse")
TF_ARMS = ("rediverse_tf", "rediverse")


# -------------------------------------------------------------------- paths --
def data_dir(*parts: str) -> str:
    """$REDIVERSE_DATA/wan2.1/<parts> (default data/wan2.1/...)."""
    return os.path.join(_data.data_root(), BACKBONE, *parts)


def output_dir(*parts: str) -> str:
    """$REDIVERSE_OUT/wan2.1/<parts> (default outputs/wan2.1/...)."""
    return os.path.join(_data.output_root(), BACKBONE, *parts)


def train_ctx_dir() -> str:
    """UMT5 contexts of the training captions, p{10000 + ci}.pt."""
    return data_dir("ctx", "train")


def vbench_ctx_dir() -> str:
    """UMT5 contexts of the VBench-300 prompts, v{idx:03d}.pt."""
    return data_dir("ctx", "vbench")


def neg_ctx_path() -> str:
    """UMT5 context of the teacher's negative prompt."""
    return data_dir("ctx", "neg.pt")


def target_dir() -> str:
    """ReDiverse-FT target bank (one safetensors entry per caption and latent)."""
    return data_dir("targets", f"f{FRAMES}_b{ANCHOR_BLOCK}_s{FT_TARGET_FORWARD}")


def direction_dir() -> str:
    """ReDiverse-TF direction files v{idx:03d}.pt."""
    return data_dir("directions", f"b{ANCHOR_BLOCK}_k{TF_DIRECTION_FORWARD}_f{FRAMES}")


def train_dir() -> str:
    """ReDiverse-FT run: config.json and adapters/rediverse_ft_{step:07d}.pth."""
    return output_dir("train")
