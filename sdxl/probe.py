#!/usr/bin/env python
"""Probe the variance characteristic of every SDXL UNet block (paper Sec. 3.1, Fig. 2, Tab. 1).

The SDXL-base UNet (teacher) or the DMD2 4-step UNet (student) gets N noises of one prompt as the
input of its first denoising step: one conditional UNet call at t = 999, the student's first
timestep, for both models (the teacher's Euler scheduler scales its initial latent by
init_noise_sigma and scale_model_input divides it back out, so the noise itself is the UNet input).
34 hooks, in execution order: conv_in, every resnet / Transformer2DModel / resampler of the down,
mid and up paths, conv_out. The collapsed block l = up_blocks.0.attentions.0 is "up0.attn0".

  python -m sdxl.probe --arm teacher --prompt-ids 27 --device cuda:0
  python -m sdxl.probe --arm student --prompt-ids 27 --device cuda:0
  python -m rediverse.probe_report --backbone sdxl
  -> outputs/sdxl/probe/<arm>/p027.json; the report prints TV / PR / Conc per block and the
     student/teacher ratios (Tab. 1 at the collapsed block)

Default prompts: the paper's 100 GenEval probe prompts (prompts/geneval_subset100.txt), N = 1000.
One prompt takes about 4 min per arm on an H100 (one pass, 184 GB of staged activations).
"""
from __future__ import annotations

import argparse

import torch

from rediverse import probe as P
from rediverse.data import GENEVAL_553, GENEVAL_SUBSET100, parse_ids, read_prompts
from sdxl import config as C

PROBE_PROMPTS = GENEVAL_SUBSET100
DTYPE = torch.float16
BATCH = 32
T_FIRST = 999                      # num_train_timesteps - 1 = the student's first timestep


def hook_points(unet):
    """[(name, module)] in forward-execution order, 34 hooks."""
    pts = [("conv_in", unet.conv_in)]
    for i, b in enumerate(unet.down_blocks):
        attns = getattr(b, "attentions", None)
        for j, r in enumerate(b.resnets):
            pts.append((f"down{i}.res{j}", r))
            if attns is not None:
                pts.append((f"down{i}.attn{j}", attns[j]))
        if b.downsamplers is not None:
            pts.append((f"down{i}.downsample", b.downsamplers[0]))
    pts += [("mid.res0", unet.mid_block.resnets[0]), ("mid.attn0", unet.mid_block.attentions[0]),
            ("mid.res1", unet.mid_block.resnets[1])]
    for i, b in enumerate(unet.up_blocks):
        attns = getattr(b, "attentions", None)
        for j, r in enumerate(b.resnets):
            pts.append((f"up{i}.res{j}", r))
            if attns is not None:
                pts.append((f"up{i}.attn{j}", attns[j]))
        if b.upsamplers is not None:
            pts.append((f"up{i}.upsample", b.upsamplers[0]))
    pts.append(("conv_out", unet.conv_out))
    return pts


class Backbone:
    name = C.BACKBONE
    dtype = DTYPE
    latent_shape = C.LATENT_SHAPE
    collapsed_block = C.SITE

    def __init__(self, arm: str, device):
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
        self.device = device
        unet = UNet2DConditionModel.from_pretrained(C.SDXL_ID, revision=C.SDXL_REVISION, subfolder="unet",
                                                    torch_dtype=DTYPE, variant="fp16")
        if arm == "student":
            from huggingface_hub import hf_hub_download
            sd = torch.load(hf_hub_download(C.DMD2_ID, C.DMD2_UNET_FP16, revision=C.DMD2_REVISION),
                            map_location="cpu", weights_only=True)
            missing, unexpected = unet.load_state_dict(sd, strict=True)
            assert not missing and not unexpected
        self.unet = unet.eval().requires_grad_(False).to(device)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(C.SDXL_ID, revision=C.SDXL_REVISION, unet=None,
                                                              vae=None, torch_dtype=DTYPE, variant="fp16").to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.t = torch.tensor(float(T_FIRST), device=device)

    def hooks(self):
        return hook_points(self.unet)

    @staticmethod
    def extract(name, out):
        return out[0] if isinstance(out, tuple) else out          # Transformer2DModel returns a 1-tuple

    def context(self, prompt_idx: int, text: str) -> dict:
        """Conditional embeddings only (no negative prompt, no CFG), at 1024 x 1024, no crop."""
        pe, _, pool, _ = self.pipe.encode_prompt(prompt=text, device=self.device, num_images_per_prompt=1,
                                                 do_classifier_free_guidance=False)
        tid = torch.tensor([[C.HEIGHT, C.WIDTH, 0, 0, C.HEIGHT, C.WIDTH]], dtype=DTYPE, device=self.device)
        return dict(pe=pe.to(DTYPE), pool=pool.to(DTYPE), tid=tid)

    def forward(self, x, ctx):
        b = x.shape[0]
        self.unet(x, self.t, encoder_hidden_states=ctx["pe"].repeat(b, 1, 1),
                  added_cond_kwargs={"text_embeds": ctx["pool"].repeat(b, 1), "time_ids": ctx["tid"].repeat(b, 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_cli_args(ap, batch=BATCH)
    a = ap.parse_args()
    ids = parse_ids(a.prompt_ids or PROBE_PROMPTS)
    bb = Backbone(a.arm, a.device)
    P.run_cli(bb, a, read_prompts(GENEVAL_553), ids, a.out_dir or C.output_dir("probe", a.arm),
              extra=dict(timestep=T_FIRST))


if __name__ == "__main__":
    main()
