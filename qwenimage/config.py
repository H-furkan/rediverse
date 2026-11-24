"""Qwen-Image families: model ids, pinned revisions, sampling schedules and the ReDiverse sites.

Two families, each one teacher and its distilled students (every model at 1024^2, as the student cards sample):

  q21  Qwen/Qwen-Image-2.1 (7.1B single-stream block-causal DiT, 32 blocks x 4096, VAE /16, 64 latent channels,
       4096 image tokens). Teacher: the base model, 40 steps, no CFG (the pipeline default: "Qwen-Image 2.1 is meant
       to be sampled without guidance"), resolution-dependent exponential shift and shift_terminal 0.02.
       Students (all CFG-free, card settings):
         lightning  lightx2v/Qwen-Image-2.1-Lightning      LoRA, 4 steps, raw sigmas [1, .75, .5, .25] + dynamic shift
         viggle     Viggle/Qwen-Image-2.1-viggle-turbo v0.3 LoRA r256, 6 steps, raw [1, .9375, .875, .75, .5, .25]
         turbo      Qwen/Qwen-Image-2.1-Turbo               full weights, the checkpoint's 8 fixed sample_sigmas
                    (OFF-CARD here: its card samples at 2048^2 / 1696x2528; we run 1024^2)
         pq         PrunaAI/P-Q-Image-2.1 6-step v0.2       LoRA r64, raw [1, .925, .85, .625, .375, .125]
         v2rl_full  lightx2v/Qwen-Image-2.1-Lightning-v2-RL the released FULL transformer (repo root), 4 fixed sigmas
         v2rl_lora  the same repo's rank-64 "tfmq" SVD reconstruction LoRA (first gate run; not the release)
  q20  Qwen/Qwen-Image (20B dual-stream MMDiT, 60 blocks x 3072, VAE /8 + 2x2 patch, 4096 image tokens).
       Teacher: 50 steps, true CFG 4 with negative prompt " " and the pipeline's norm rescale.
       Student lightning20: lightx2v/Qwen-Image-Lightning 8-step V2.0 (bf16 LoRA), shift 3 (mu = log 3), no CFG.
  q2512 Qwen/Qwen-Image-2512 (same architecture as q20). Teacher: 50 steps, true CFG 4, the card's negative prompt.
       Student lightning2512: lightx2v/Qwen-Image-2512-Lightning 8-step V1.0 (bf16 LoRA), shift 3, no CFG.

Site: transformer_blocks.{l} output, image rows only ([4096 tokens x d_model]). In q21 the image rows are the last
4096 rows of the joint [text; image] sequence; in q20 the block returns (text, image) streams separately.
The collapsed block l and the teacher forward tau are chosen per student from the probe and the schedules.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from rediverse.data import data_root, output_root

DTYPE = torch.bfloat16
RES = 1024

# ------------------------------------------------------------------ families --
FAMILIES = {
    "q21": dict(
        backbone="qwen21", arch="q21",
        repo="Qwen/Qwen-Image-2.1", revision="d26bb61231c349cf6b7896fa83353113880e1ba3",
        n_blocks=32, d_model=4096, vae_scale=16, patch=1, latent_channels=64, in_channels=64,
        teacher=dict(steps=40, cfg=1.0, scheduler={}),             # base scheduler config as shipped
    ),
    "q20": dict(
        backbone="qwen20", arch="q20",
        repo="Qwen/Qwen-Image", revision="75e0b4be04f60ec59a75f475837eced720f823b6",
        n_blocks=60, d_model=3072, vae_scale=8, patch=2, latent_channels=16, in_channels=64,
        teacher=dict(steps=50, cfg=4.0, negative=" ", scheduler={}),
    ),
    "q2512": dict(       # same architecture as q20; card negative prompt (also used by the Lightning repo's script)
        backbone="qwen2512", arch="q20",
        repo="Qwen/Qwen-Image-2512", revision="25468b98e3276ca6700de15c6628e51b7de54a26",
        n_blocks=60, d_model=3072, vae_scale=8, patch=2, latent_channels=16, in_channels=64,
        teacher=dict(steps=50, cfg=4.0, scheduler={},
                     negative="低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"),
    ),
}
for _f in FAMILIES.values():
    _f["grid"] = RES // (_f["vae_scale"] * _f["patch"])          # 64 in both families
    _f["n_img"] = _f["grid"] ** 2                                 # 4096 image tokens
    _f["latent_shape"] = (_f["n_img"], _f["in_channels"])        # packed latents as the pipelines feed them
    _f["d_block"] = _f["n_img"] * _f["d_model"]

# Scheduler overrides relative to the family's shipped scheduler config (FlowMatchEulerDiscreteScheduler).
_NO_TERMINAL = dict(shift_terminal=None)
_STATIC = dict(use_dynamic_shifting=False, shift_terminal=None)
_LOG3 = dict(base_shift=math.log(3), max_shift=math.log(3), shift_terminal=None)   # lightx2v: "shift=3 in distillation"

STUDENTS = {
    "lightning": dict(family="q21", kind="lora", repo="lightx2v/Qwen-Image-2.1-Lightning",
                      revision="6b0a1176327c01522a44276c5da67331a025b6fc", file="pytorch_lora_weights.safetensors",
                      sigmas=[1.0, 0.75, 0.5, 0.25], scheduler=_NO_TERMINAL, cfg=1.0),
    "viggle": dict(family="q21", kind="lora", repo="Viggle/Qwen-Image-2.1-viggle-turbo",
                   revision="009a44a895ef85f7e643c80fdca9543795248867",
                   file="Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors",
                   sigmas=[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25], scheduler=_NO_TERMINAL, cfg=1.0),
    "turbo": dict(family="q21", kind="full", repo="Qwen/Qwen-Image-2.1-Turbo",
                  revision="fc13ba1986fcb905854dc8de2f52eee395f49b5b",
                  sigmas=[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568],
                  scheduler=_STATIC, cfg=1.0),
    "pq": dict(family="q21", kind="lora", repo="PrunaAI/P-Q-Image-2.1", revision="b479282ea5ca0d17eb8b415bc09c1394dc29f6c5",
               file="p_q_image_2.1_6_steps_v0.2.safetensors",
               sigmas=[1.0, 0.925, 0.85, 0.625, 0.375, 0.125], scheduler=_NO_TERMINAL, cfg=1.0),
    # The release is the full transformer (README: "This checkpoint is not a LoRA adapter"), stored at the repo root.
    "v2rl_full": dict(family="q21", kind="full", subfolder=None, repo="lightx2v/Qwen-Image-2.1-Lightning-v2-RL",
                      revision="682b2529ffb5ae9ba39ceb81a31ce99e602da2ee",
                      sigmas=[1.0, 0.95418, 0.89508, 0.704534], scheduler=_STATIC, cfg=1.0),
    # Its rank-64 LoRA export (metadata: an SVD reconstruction of the full weights); kept to document the first gate.
    "v2rl_lora": dict(family="q21", kind="lora", repo="lightx2v/Qwen-Image-2.1-Lightning-v2-RL",
                      revision="682b2529ffb5ae9ba39ceb81a31ce99e602da2ee",
                      file="qwen_image_21_lightning_v2_rl_4steps_lora_rank64.safetensors",
                      sigmas=[1.0, 0.95418, 0.89508, 0.704534], scheduler=_STATIC, cfg=1.0),
    # ---- second round (gate only first; notes/09 survey; settings from each card / shipped config) ----
    # linoyts DMD2 2-step v1.5 (card: "New V1.5 checkpoint ... improved prompt adherence"); fixed sigmas [1, 0.5],
    # use_dynamic_shifting=False, shift 1, shift_terminal None, true_cfg 1
    "linoyts_dmd2": dict(family="q21", kind="lora", repo="linoyts/Qwen-Image-2.1-DMD2-2steps",
                         revision="88455a22d4aae892da41302de1122a560d9fe2dc", file="pytorch_lora_weights_v1_5.safetensors",
                         sigmas=[1.0, 0.5], scheduler=_STATIC, cfg=1.0),
    # Wuli-art Qwen-Image-2.1-Turbo 4-step bf16 full transformer (4step-bf16/): shipped sampling_config sample_sigmas
    # and scheduler_config (static, shift 1, no terminal), true_cfg 1
    "wuli21_4": dict(family="q21", kind="full", subfolder="4step-bf16", repo="Wuli-art/Qwen-Image-2.1-Turbo",
                     revision="e47ebf3ae35929e9e416aa279e886b80087c44c9",
                     sigmas=[1.0, 0.9177013039588928, 0.787998616695404, 0.5533687472343445], scheduler=_STATIC, cfg=1.0),
    # congruency MFD v0.5 ("best general use", 4-6 steps; 5 chosen). Card: ComfyUI euler / sgm_uniform, no extra shift
    # node -> ComfyUI's QwenImage21 default ModelSamplingFlux shift mu = 0.69 (comfy/supported_models.py) on nodes
    # linspace(1, 0, n+1)[:-1]; in diffusers: dynamic exponential shift with mu fixed at 0.69, no terminal stretch.
    "mfd": dict(family="q21", kind="lora", repo="congruency/Qwen-Image-2.1-turbo-mfd",
                revision="22e33e71f0a5eb8e9a9f48c7eb093b66cdfd1ad1", file="Qwen-Image-2.1-turbo-mfd-v0.5.safetensors",
                sigmas=list(np.linspace(1.0, 1 / 5, 5)), scheduler=dict(base_shift=0.69, max_shift=0.69, shift_terminal=None),
                cfg=1.0),
    # nvidia Qwen-Image-Flash: full 20B transformer (DMD2, CFG 4 internalised); shipped scheduler: static shift 3,
    # 4 steps (pipeline default linspace nodes), true_cfg 1, 1024^2. Its TE / VAE / tokenizer equal Qwen-Image's (lfs sha).
    "flash20": dict(family="q20", kind="full", repo="nvidia/Qwen-Image-Flash",
                    revision="eafac15f6140e6dd9c6031217d658ac10bfb604b",
                    sigmas=list(np.linspace(1.0, 1 / 4, 4)),
                    scheduler=dict(use_dynamic_shifting=False, shift=3.0, shift_terminal=None), cfg=1.0),
    # zjwfufu SMD-Qwen-Image: r256 LoRA MERGED into the base (bf16 merge, bf16 delta) plus 2 replacement tensors
    # (norm_out.linear); sampling.json grid at 1024^2: raw [1, 5/7, 3/7, 1/7, 0] (final step 0.5), shift exp(0.6935) =
    # 2.0008, clamped to 0.999; the shifted sigmas are given explicitly (static scheduler, no second shift).
    # Their StudentEulerScheduler steps in bf16 (euler="bf16").
    "smd20": dict(family="q20", kind="merged_lora", repo="zjwfufu/SMD-Qwen-Image",
                  revision="e841750002a07af733dc39d21204fb4d05d52fae", file="smd_lora.safetensors",
                  adapter_config="lora/adapter_config.json",
                  sigmas=[0.999, 0.8333890490115364, 0.600096285704502, 0.2500752337707165], scheduler=_STATIC,
                  cfg=1.0, euler="bf16", sigma0=0.999),
    "lightning20": dict(family="q20", kind="lora", repo="lightx2v/Qwen-Image-Lightning",
                        revision="d468c534b360e0754c6bf9e38ed49e1e09cd2e2a",
                        file="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
                        sigmas=list(np.linspace(1.0, 1 / 8, 8)), scheduler=_LOG3, cfg=1.0),
    "lightning2512": dict(family="q2512", kind="lora", repo="lightx2v/Qwen-Image-2512-Lightning",
                          revision="4609161e308ce82018a42c2a0c9b107e865bb751",
                          file="Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors",
                          sigmas=list(np.linspace(1.0, 1 / 8, 8)), scheduler=_LOG3, cfg=1.0),
    # Wuli-art 2512 Turbo LoRA 4-step V3.0 (latest). Card (DiffSynth): exponential_shift_mu = log 2.5 (takes precedence
    # over the size-dependent mu), shift_terminal None, nodes linspace(1, 0, n+1)[:-1], cfg 1, card size 1328^2 (we run
    # 1024^2 like the 2512 teacher).
    "wuli2512_4": dict(family="q2512", kind="lora", repo="Wuli-art/Qwen-Image-2512-Turbo-LoRA",
                       revision="5f56f05d8c57093e1f2d27826bc9a6c8b16869f7",
                       file="Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V3.0-bf16.safetensors",
                       sigmas=list(np.linspace(1.0, 1 / 4, 4)),
                       scheduler=dict(base_shift=math.log(2.5), max_shift=math.log(2.5), shift_terminal=None), cfg=1.0),
}

# Per-student ReDiverse sites (l from the probe, tau from the schedules), filled in as the pilot proceeds.
# QWEN_ANCHOR_BLOCK / QWEN_TAU override them for ablations.
SITES: dict[str, dict] = {
    # probe N=200 x 20 prompts (results/qwen_probe.md): sharpest sustained concentration step (paper App. A)
    "lightning": dict(block=29, tau=10, ablation=21),      # Conc 2.76 -> 3.97 at 28 -> 29, PR 0.62
    "lightning20": dict(block=57, tau=4, ablation=58),     # Conc 1.20 -> 1.42 -> 1.64 at 56-58, PR 0.85 at 57
    "lightning2512": dict(block=57, tau=4, ablation=58),   # Conc 1.21 -> 1.46 -> 1.72 at 56-58, PR 0.85 at 57
    "linoyts_dmd2": dict(block=29, tau=26, ablation=21),   # Conc 3.14 -> 4.23 at 28 -> 29, PR 0.65
    "flash20": dict(block=57, tau=9),                      # Conc 0.79 -> 0.96 at 56 -> 57, PR 0.79 (TV ratio ~0.5)
    "smd20": dict(block=57, tau=14),                       # Conc 1.12 -> 1.38 at 56 -> 57, PR 0.79 (57 ~ 58 within CI)
    "wuli2512_4": dict(block=58, tau=10),                  # Conc 1.34 -> 1.56 at 57 -> 58, PR 0.88 (sharpest)
    "mfd": dict(block=30, tau=8),                          # only clear step 29 -> 30 (Conc 1.32 -> 1.43, PR 0.90)
}


def family(name: str) -> dict:
    return FAMILIES[name]


def arch(fam: str) -> str:
    """'q21' (single-stream Qwen-Image-2.1) or 'q20' (dual-stream Qwen-Image / Qwen-Image-2512)."""
    return FAMILIES[fam]["arch"]


def student(name: str) -> dict:
    return STUDENTS[name]


def student_weights(name: str) -> tuple[str, str | None]:
    """(repo, subfolder) of a full student's transformer weights (default subfolder 'transformer')."""
    s = STUDENTS[name]
    return s["repo"], s.get("subfolder", "transformer")


def family_of(student_name: str) -> str:
    return STUDENTS[student_name]["family"]


def site(student_name: str) -> dict:
    s = dict(SITES.get(student_name, {}))
    if os.environ.get("QWEN_ANCHOR_BLOCK"):
        s["block"] = int(os.environ["QWEN_ANCHOR_BLOCK"])
    if os.environ.get("QWEN_TAU"):
        s["tau"] = int(os.environ["QWEN_TAU"])
    return s


# ------------------------------------------------------------------ paths --
def data_dir(fam: str, *parts) -> str:
    return os.path.join(data_root(), FAMILIES[fam]["backbone"], *parts)


def output_dir(fam: str, *parts) -> str:
    return os.path.join(output_root(), FAMILIES[fam]["backbone"], *parts)
