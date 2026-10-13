"""MiniMax-H3 / lightx2v Turbo: model ids, pinned revisions, geometry and the ReDiverse recipe.

Every CLI default of the h3 package reads its value from here, so running the scripts with no
extra flags reproduces the paper's ReDiverse-FT adapter, direction banks and evaluation clips.
"""
from __future__ import annotations

import os

from rediverse import data as _data

BACKBONE = "minimax-h3"           # data/ and outputs/ directory name, adapter metadata id

# ------------------------------------------------------------------- models --
TEACHER_REPO = "MiniMaxAI/MiniMax-H3"
TEACHER_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
# Subfolders read from the teacher repository (transformer weights are shared by teacher and student).
TEACHER_SUBFOLDERS = ("transformer", "scheduler", "audio_scheduler", "tokenizer", "text_encoder", "vae")
STUDENT_REPO = "lightx2v/Minimax-h3-Turbo"
STUDENT_REVISION = "05ef678438e84933c406131b59abbf86919b3aac"
STUDENT_FILE = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"   # rank-128 LoRA on the teacher
STUDENT_LORA_MODULES = 312        # 50 blocks x 6 + 2 token-refiner blocks x 6
NUM_LAYERS, HIDDEN = 50, 5376

# Scheduler steps passed to set_timesteps; MiniMax-H3 runs steps - 1 transformer forwards.
STEPS = dict(teacher=50, student=4)
TEXT_ENCODER_LAYER = 50

# ----------------------------------------------------------------- geometry --
HEIGHT, WIDTH = 768, 1344
NUM_FRAMES = 124                  # 17 * 7 + 5 pixel frames, 5.17 s
FPS = 24

# ---------------------------------------------------------------- ReDiverse --
ANCHOR_BLOCK = 45                 # collapsed block l: transformer_blocks[45], video rows
TAU = _data.FT_TARGET_FORWARD["h3"]   # 16: teacher forward of the FT target and of the TF directions

# ReDiverse-FT: target bank over 20 GenEval prompts x 20 latents
FT_PROMPT_IDS = _data.H3_TRAIN_IDS        # prompts/h3_train_geneval_ids.txt (indices into geneval_553)
FT_LATENTS_PER_PROMPT = 20
LORA_RANK = _data.LORA_RANK               # 32
LORA_ALPHA = _data.LORA_ALPHA             # 32
LORA_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
ADAPTER_NAME = "rediverse_ft"             # peft adapter name in training and generation
REG_LOSS_WEIGHT = 0.5                     # loss = 0.5 * MSE(h_student, h_teacher) / Var(h_teacher)
SAMPLES_PER_PROMPT = 2                    # latents of one caption per micro-batch
CAPTIONS_PER_STEP = 4                     # micro-batches per optimizer step
MICRO_CHUNK = 1                           # latents per forward/backward and per Var(h_teacher) normaliser
MAX_ITER = 201                            # iterations 1..200: h3.train --steps defaults to 200
SAVE_EVERY = 50
LOG_EVERY = 5
SPLIT_CUT = 23                            # training: blocks 0..22 on the first GPU, 23..45 on the second
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
    """$REDIVERSE_DATA/minimax-h3/<parts> (default data/minimax-h3/...)."""
    return os.path.join(_data.data_root(), BACKBONE, *parts)


def output_dir(*parts: str) -> str:
    """$REDIVERSE_OUT/minimax-h3/<parts> (default outputs/minimax-h3/...)."""
    return os.path.join(_data.output_root(), BACKBONE, *parts)


def train_ctx_dir() -> str:
    return data_dir("ctx", "train")


def vbench_ctx_dir() -> str:
    return data_dir("ctx", "vbench")


def target_dir(k: int = TAU) -> str:
    return data_dir("targets", f"b{ANCHOR_BLOCK}_k{k}")


def direction_dir(k: int = TAU) -> str:
    return data_dir("directions", f"b{ANCHOR_BLOCK}_k{k}")


def ft_prompt_ids() -> list[int]:
    """The 20 GenEval indices of the ReDiverse-FT target bank."""
    return _data.parse_ids(FT_PROMPT_IDS)
