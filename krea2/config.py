"""Krea 2 Raw / Turbo: model ids, pinned revisions, schedules and the ReDiverse site.

Teacher  krea/Krea-2-Raw   (midtrain checkpoint; 52 FlowMatchEuler steps, resolution-dependent shift mu,
                            Krea guidance 3.5 = conventional CFG 4.5)
Student  krea/Krea-2-Turbo (TDM-distilled + post-trained full weights; 8 steps, fixed mu = 1.15, no CFG)
Both share the text encoder (Qwen3-VL-4B), tokenizer, VAE (Qwen-Image) and transformer config; only the
Krea2Transformer2DModel weights differ.

Site     transformer_blocks.{l} output, image rows only (the last 4,096 rows of the joint [text; image] sequence,
         x 6,144). The collapsed block l is set from the probe (krea2.probe + rediverse.probe_report).

Krea's CFG convention: v = cond + g (cond - uncond), i.e. conventional w = 1 + g. Guidance values below are in
Krea's convention (as passed to Krea2Pipeline) unless named *_W.
"""
from __future__ import annotations

import os

import torch

from rediverse.data import data_root, output_root

BACKBONE = "krea2"

# ------------------------------------------------------------------ models --
TEACHER_ID = "krea/Krea-2-Raw"
TEACHER_REVISION = "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"
STUDENT_ID = "krea/Krea-2-Turbo"
STUDENT_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
DTYPE = torch.bfloat16

STEPS = dict(teacher=52, student=8)              # each model's own sampling schedule (model cards)
GUIDANCE = dict(teacher=3.5, student=0.0)        # Krea convention; teacher = conventional CFG 4.5
MAX_SEQ_LEN = 512                                 # Krea2Pipeline default fixed text length

N_BLOCKS = 28
RES = 1024                                        # Raw is trained up to ~1k; Turbo's card samples 2048 too
VAE_SCALE = 8
PATCH = 2
GRID = RES // (VAE_SCALE * PATCH)                 # 64
N_IMG_TOKENS = GRID * GRID                        # 4096
IN_CHANNELS = 64                                  # 16 latent channels x 2 x 2 patch
LATENT_SHAPE = (N_IMG_TOKENS, IN_CHANNELS)        # packed latents, as Krea2Pipeline feeds the transformer
D_MODEL = 6144
D_BLOCK = N_IMG_TOKENS * D_MODEL                  # image rows of one block, flattened (25,165,824)

# ------------------------------------------------------------ ReDiverse site --
# l = 26 of 28: probe Raw vs Turbo (N=200, 20 GenEval prompts, sigma 1): PR ratio 0.62, TV 3.26, Conc 5.54; Conc steps
# up from 2.17 at block 25 (paper App. A rule). Override with KREA2_ANCHOR_BLOCK for ablations.
ANCHOR_BLOCK = int(os.environ.get("KREA2_ANCHOR_BLOCK", "26"))
# tau: the teacher forward whose Euler update lands closest to the student's sigma after step 1.
# Turbo sigmas[1] = 0.95672; Raw (52 steps, mu 0.906 at 4096 tokens) forward 5 reads 0.96741 and lands at
# 0.95883 (|d| = 0.0021; forward 6 lands at 0.94990, |d| = 0.0068). Verified in krea2.model.teacher_target_sigmas.
TEACHER_TARGET_STEP = 5
DIRECTION_STEP = 5


# ------------------------------------------------------------------ paths --
def data_dir(*parts) -> str:
    return os.path.join(data_root(), BACKBONE, *parts)


def output_dir(*parts) -> str:
    return os.path.join(output_root(), BACKBONE, *parts)
