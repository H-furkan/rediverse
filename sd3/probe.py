#!/usr/bin/env python
"""Probe the variance characteristic of every SD3-Medium MMDiT block (paper Sec. 3.1, Fig. 2, Tab. 1).

The SD3-Medium transformer (teacher) or the same transformer with the TDM LoRA at 0.125 (student)
gets N noises of one prompt as the input of its first denoising step: one conditional forward at
sigma 1 (both schedules share the first timestep). The probed tensor of each of the 24
transformer_blocks is its image stream ([4096 tokens x 1536]); the collapsed block l is block 12.

  python -m sd3.probe --arm teacher --prompt-ids 27 --device cuda:0
  python -m sd3.probe --arm student --prompt-ids 27 --device cuda:0
  python -m rediverse.probe_report --backbone sd3

Default prompts: the paper's 100 GenEval probe prompts (prompts/geneval_subset100.txt), N = 1000.
"""
from __future__ import annotations

import argparse

import torch

from rediverse import probe as P
from rediverse.data import GENEVAL_553, GENEVAL_SUBSET100, parse_ids, read_prompts
from sd3 import config as C
from sd3 import model as M

PROBE_PROMPTS = GENEVAL_SUBSET100
BATCH = 24


class Backbone:
    name = C.BACKBONE
    dtype = C.DTYPE
    latent_shape = C.LATENT_SHAPE
    collapsed_block = f"block{C.ANCHOR_BLOCK:02d}"

    def __init__(self, arm: str, device):
        from sd3.encode_prompts import load_text_pipe
        self.device = device
        self.model = M.load_teacher(device) if arm == "teacher" else M.load_student(device)
        self.pipe = load_text_pipe(device)
        self.t = torch.tensor(M.first_timestep(), dtype=torch.float32, device=device)

    def hooks(self):
        return [(f"block{i:02d}", b) for i, b in enumerate(self.model.transformer_blocks)]

    @staticmethod
    def extract(name, out):
        return M.extract(name, out)

    def context(self, prompt_idx: int, text: str) -> dict:
        """Conditional embeddings only, from StableDiffusion3Pipeline.encode_prompt."""
        pe, _, pooled, _ = self.pipe.encode_prompt(prompt=text, prompt_2=None, prompt_3=None, device=self.device,
                                                   num_images_per_prompt=1, do_classifier_free_guidance=False)
        return dict(pe=pe.to(C.DTYPE), pooled=pooled.to(C.DTYPE))

    def forward(self, x, ctx):
        b = x.shape[0]
        self.model(hidden_states=x, timestep=self.t.expand(b), encoder_hidden_states=ctx["pe"].expand(b, -1, -1),
                   pooled_projections=ctx["pooled"].expand(b, -1), return_dict=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_cli_args(ap, batch=BATCH)
    a = ap.parse_args()
    ids = parse_ids(a.prompt_ids or PROBE_PROMPTS)
    bb = Backbone(a.arm, a.device)
    P.run_cli(bb, a, read_prompts(GENEVAL_553), ids, a.out_dir or C.output_dir("probe", a.arm),
              extra=dict(timestep=float(bb.t)))


if __name__ == "__main__":
    main()
