# SPDX-License-Identifier: Apache-2.0
# Added by ReDiverse on top of NVlabs/FastGen (commit 060b857); not part of upstream FastGen.
"""ReDiverse-FT for the SDXL / DMD2 student (paper Sec. 4.1 and App. D).

For a caption c and K paired noises eps_1..eps_K, the student's step-1 activation at the
collapsed block (up_blocks.0.attentions.0) is regressed onto the frozen teacher's activation at
the same block, taken at the teacher's forward tau of its own 50-step Euler trajectory started
from the same noise:

    L = || h_student(eps, c; step 1) - h_teacher(eps, c; step tau) ||^2   (mean over elements)

Only a LoRA on the attention and feed-forward projections of the blocks up to and including the
collapsed block is trained. The student's step 1 is its only forward here, so the adapter stays
enabled throughout; at inference it is enabled for step 1 only (sdxl/generate.py).
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from fastgen.methods.model import FastGenModel
import fastgen.utils.logging_utils as logger

from sdxl.teacher_rollout import TEACHER_STEPS, teacher_states


def _block_output(o):
    """Transformer2DModel inside the UNet returns a 1-tuple; take the hidden states."""
    return o.sample if hasattr(o, "sample") else (o[0] if isinstance(o, tuple) else o)


class ReDiverseFTModel(FastGenModel):
    """Student (net) with a step-1 LoRA, frozen teacher, activation-matching loss."""

    def build_model(self):
        super().build_model()
        self.build_teacher()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ LoRA
    def _attach_lora(self):
        from peft import LoraConfig
        cfg = self.config
        unet = self.net.unet
        self.net.requires_grad_(False)
        pat = re.compile(cfg.lora_target_regex)
        targets = [n for n, m in unet.named_modules() if pat.match(n) and isinstance(m, torch.nn.Linear)]
        assert targets, f"LoRA target regex matched no module: {cfg.lora_target_regex}"
        # alpha = rank (scaling 1.0); peft initialises A (kaiming-uniform) and B = 0
        unet.add_adapter(LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_rank, target_modules=targets))
        for n, p in self.net.named_parameters():
            p.requires_grad_("lora_" in n)
        n_lora = sum(p.numel() for n, p in self.net.named_parameters() if "lora_" in n)
        n_tens = sum(1 for n, _ in self.net.named_parameters() if "lora_" in n)
        n_all = sum(p.numel() for p in self.net.parameters())
        logger.info(f"Generator LoRA: rank={cfg.lora_rank} modules={len(targets)} tensors={n_tens} "
                    f"| trainable {n_lora/1e6:.1f}M / {n_all/1e6:.1f}M ({100*n_lora/n_all:.2f}%)")

    def _lora_requires_grad(self):
        """peft's enable_adapters() also flips requires_grad; only the LoRA tensors train."""
        for n, p in self.net.named_parameters():
            p.requires_grad_("lora_" in n)

    def on_train_begin(self, is_fsdp: bool = False) -> None:
        super().on_train_begin(is_fsdp=is_fsdp)
        cfg = self.config
        k = int(cfg.teacher_target_step)
        assert 2 <= k <= TEACHER_STEPS, f"teacher_target_step={k} must lie in 2..{TEACHER_STEPS}"
        self._attach_lora()
        if cfg.gradient_checkpointing:
            self.net.unet.enable_gradient_checkpointing()
            logger.info("Generator UNet gradient checkpointing enabled")
        self.net.to(device=self.device)

        # capture the student's collapsed-block activation (graph-connected) during the step-1 forward
        s_mods = dict(self.net.unet.named_modules())
        t_mods = dict(self.teacher.unet.named_modules())
        assert cfg.collapsed_block in s_mods and cfg.collapsed_block in t_mods, \
            f"collapsed block '{cfg.collapsed_block}' not found in the UNet"
        self._teacher_block = {cfg.collapsed_block: t_mods[cfg.collapsed_block]}
        self._h_student = None
        self._capture = False

        def s_hook(_m, _i, o):
            if self._capture:
                self._h_student = _block_output(o)

        s_mods[cfg.collapsed_block].register_forward_hook(s_hook)

        self.net.unet.enable_adapters()
        self._lora_requires_grad()
        self._checked = False
        logger.info(f"ReDiverse-FT: block {cfg.collapsed_block}, loss {cfg.reg_loss_weight} * MSE, teacher target "
                    f"= forward {k} of the {TEACHER_STEPS}-step Euler trajectory, CFG {cfg.teacher_guidance} on "
                    f"forwards 1..{k - 1}, conditional forward {k}")

    def _t_list_tensor(self) -> torch.Tensor:
        return torch.tensor(self.config.sample_t_cfg.t_list, device=self.device,
                            dtype=self.net.noise_scheduler.t_precision)

    # ------------------------------------------------------------- training
    def _teacher_target(self, latents: torch.Tensor, condition) -> torch.Tensor:
        """The teacher's collapsed-block activation at forward tau, from the student's own initial latent
        (see sdxl/teacher_rollout.py for why that latent goes in as-is)."""
        cfg = self.config
        k = int(cfg.teacher_target_step)
        cond = {"pe": condition[0], "pool": condition[1],
                # the teacher wrapper's add_time_ids (1024, 1024, 0, 0, 1024, 1024)
                "tid": self.teacher.add_time_ids.to(latents.device)}
        states = teacher_states(self.teacher.unet, self._teacher_block, latents, cond, k, keep={k},
                                guidance=float(cfg.teacher_guidance))
        return states[k][cfg.collapsed_block].detach()

    def single_train_step(self, data: dict, iteration: int):
        """One micro-batch: one caption with K paired noises (fastgen.datasets.rediverse_ft)."""
        cfg = self.config
        self._lora_requires_grad()
        condition = [c.to(self.precision) for c in data["condition"]]
        eps = data["noise"].to(self.device, self.precision)
        bs = eps.shape[0]
        t1 = self._t_list_tensor()[0]
        # student step-1 input: eps * sigma(t=0.999), FastGen's SDXL schedule
        x_init = self.net.noise_scheduler.latents(noise=eps, t_init=t1)

        self._h_student = None
        self._capture = True
        step1_x0 = self.net(x_init, t1.expand(bs), condition=condition, fwd_pred_type="x0")
        self._capture = False
        h_s = self._h_student
        assert h_s is not None, "student collapsed-block hook did not fire"
        if not self._checked:
            n_live = sum(1 for n, p in self.net.named_parameters() if "lora_" in n and p.requires_grad)
            assert n_live > 0, "no generator LoRA tensor requires grad at the step-1 forward"
            # gradient checkpointing must keep the hook capture graph-connected
            # (diffusers checkpoints with use_reentrant=False)
            assert h_s.requires_grad and h_s.grad_fn is not None, (
                f"the captured activation at '{cfg.collapsed_block}' is detached; set "
                "model.gradient_checkpointing=False")
            logger.info(f"step-1 forward: {n_live} LoRA tensors live; block capture is graph-connected")
            self._checked = True

        h_t = self._teacher_target(x_init, condition)
        reg_loss = F.mse_loss(h_s.float(), h_t.float())
        loss = cfg.reg_loss_weight * reg_loss

        # telemetry: mean across-noise std of the collapsed-block activation, student vs teacher
        with torch.no_grad():
            sp_s = (h_s.float().var(dim=0, unbiased=False) + 1e-6).sqrt().mean()
            sp_t = (h_t.float().var(dim=0, unbiased=False) + 1e-6).sqrt().mean()
        loss_map = {
            "total_loss": loss,
            "reg_loss": reg_loss,
            "spread_student": sp_s,
            "spread_teacher": sp_t,
            "spread_ratio": sp_s / sp_t.clamp_min(1e-12),
            "step1_std": step1_x0.detach().float().std(),
        }
        return loss_map, {}

    def _get_outputs(self, *args, **kwargs):
        return {}
