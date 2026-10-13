# SPDX-License-Identifier: Apache-2.0
# Added by ReDiverse on top of NVlabs/FastGen (commit 060b857); not part of upstream FastGen.
"""ReDiverse-FT on SDXL / DMD2, the recipe of the released adapter (paper App. D).

  student   tianweiy/DMD2 4-step UNet (fp32), step 1 at t=0.999, LoRA r=32 / alpha=32 on the
            attention and feed-forward projections of every transformer block up to and including
            up_blocks.0.attentions.0 (440 Linear modules)
  target    SDXL-base teacher (fp32), forward 7 of its 50-step Euler trajectory from the same
            noise, CFG 8 on forwards 1-6, conditional activation at forward 7
  loss      1.0 * MSE over all elements of the [K, 1280, 32, 32] collapsed-block activation
  data      4 captions x 8 noises per optimizer step (4 accumulation rounds of 1 caption x 8 noises)
  optimiser AdamW lr 1e-4, weight decay 0.01, betas (0.9, 0.999), eps 1e-8, constant lr,
            gradient clipping at 10, 200 steps, seed 0, fp32 with TF32 matmuls

Paths come from sdxl/train.py: REDIVERSE_DATA (training contexts and the initialisation
checkpoint) in the environment, the run directory as FASTGEN_OUTPUT_ROOT plus log_config.
"""
import attrs
from omegaconf import DictConfig

from fastgen.callbacks.loss_print import LossPrintCallback
from fastgen.configs.callbacks import GPUStats_CALLBACK, GradClip_CALLBACK, ParamCount_CALLBACK
from fastgen.configs.config import BaseConfig, BaseModelConfig
from fastgen.configs.net import SDXLConfig
from fastgen.datasets.rediverse_ft import ReDiverseFTLoader
from fastgen.methods.rediverse_ft import ReDiverseFTModel
from fastgen.utils import LazyCall as L

from rediverse import data as D
from sdxl import config as C


@attrs.define(slots=False)
class ReDiverseFTModelConfig(BaseModelConfig):
    lora_rank: int = D.LORA_RANK                     # alpha = rank
    lora_target_regex: str = C.LORA_TARGET_REGEX
    collapsed_block: str = C.ANCHOR_MODULE
    teacher_target_step: int = C.TEACHER_TARGET_STEP
    teacher_guidance: float = C.TEACHER_GUIDANCE
    reg_loss_weight: float = D.REG_LOSS_WEIGHT
    gradient_checkpointing: bool = True


def create_config():
    config = BaseConfig()
    config.model = ReDiverseFTModelConfig()
    config.model_class = L(ReDiverseFTModel)(config=None)

    config.model.net = SDXLConfig
    config.model.input_shape = list(C.LATENT_SHAPE)
    config.model.enable_preprocessors = False
    config.model.precision = "float32"
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.749, 0.499, 0.249, 0.0]

    config.model.net_optimizer.lr = D.LR
    config.model.net_optimizer.weight_decay = D.WEIGHT_DECAY
    config.model.net_optimizer.betas = D.BETAS
    config.model.net_optimizer.eps = D.EPS
    config.model.net_scheduler.warm_up_steps = [0]  # constant lr

    config.dataloader_train = L(ReDiverseFTLoader)(
        ctx_dir=C.ctx_dir(),
        batch_size=1,
        captions_per_step=D.CAPTIONS_PER_STEP,
        noises_per_caption=D.SAMPLES_PER_STEP // D.CAPTIONS_PER_STEP,
        seed=0,
        input_shape=list(C.LATENT_SHAPE),
    )

    # one caption per micro-batch -> 4 gradient-accumulation rounds per optimizer step
    config.trainer.batch_size_global = D.CAPTIONS_PER_STEP
    config.trainer.max_iter = D.STEPS + 1           # iterations 1..200
    config.trainer.save_ckpt_iter = D.STEPS
    config.trainer.logging_iter = 5
    config.trainer.resume = False
    config.trainer.seed = 0
    config.trainer.callbacks = DictConfig({
        **GradClip_CALLBACK,                         # GradClipCallback(grad_norm=10.0, model_key="net")
        **GPUStats_CALLBACK,
        **ParamCount_CALLBACK,
        "loss_print": L(LossPrintCallback)(every_n=5),
    })
    assert config.trainer.callbacks.grad_clip.grad_norm == D.GRAD_CLIP

    config.trainer.checkpointer.pretrained_ckpt_path = C.init_ckpt_path()
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "net"}

    config.log_config.project = C.BACKBONE
    config.log_config.group = "train"
    config.log_config.name = "rediverse_ft"
    config.log_config.wandb_mode = "disabled"
    return config
