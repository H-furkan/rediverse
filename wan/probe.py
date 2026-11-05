#!/usr/bin/env python
"""Probe the variance characteristic of every Wan 2.1 transformer block (paper Sec. 3.1, Fig. 2, Tab. 1).

Wan2.1-T2V-14B (teacher) or the lightx2v 4-step student gets N noises of one prompt as the input
of its first denoising step: one conditional forward at t = 999, no CFG, no sampler step. The
probed tensor of each of the 40 blocks is its full output (video tokens x 5120); the collapsed
block l is block 28. The paper probes 17-frame clips at 832 x 480 (5 latent frames, D = 39.9 M per
block); --frames changes that.

  python -m wan.probe --arm teacher --prompt-ids 27 --device cuda:0
  python -m wan.probe --arm student --prompt-ids 27 --device cuda:0
  python -m rediverse.probe_report --backbone wan2.1

Default prompts: the paper's 20 video probe prompts (prompts/probe_video20.txt, GenEval indices),
N = 1000. At 17 frames one block of one arm stages N x 40 M bf16 values (80 GB); with
--ram-gb 1400 the 40 blocks take three passes over the N noises.
"""
from __future__ import annotations

import argparse
import os

import torch

from rediverse import probe as P
from rediverse.data import GENEVAL_553, PROMPTS_DIR, parse_ids, read_prompts
from wan import config as C
from wan import model as M

PROBE_PROMPTS = os.path.join(PROMPTS_DIR, "probe_video20.txt")
BATCH = 4
FRAMES = 17
T_FIRST = 999.0


class Backbone:
    name = C.BACKBONE
    dtype = M.DTYPE
    collapsed_block = f"block{C.ANCHOR_BLOCK:02d}"

    def __init__(self, arm: str, device, frames: int = FRAMES):
        assert (frames - 1) % 4 == 0, f"--frames must be 4k + 1, got {frames}"
        self.device = device
        self.latent_shape = (16, (frames - 1) // 4 + 1, C.LATENT_SHAPE[2], C.LATENT_SHAPE[3])
        self.tok, self.enc = M.text_encoder(device)
        self._ctx = {}
        self.model = M.load_transformer(arm, device)
        self.t = torch.tensor([T_FIRST], dtype=torch.float32, device=device)

    def encode(self, items):
        """Text contexts of [(prompt_idx, text)], then the text encoder is freed."""
        for i, text in items:
            self._ctx[i] = M.embed(self.tok, self.enc, text, self.device)[0]
        self.tok = self.enc = None
        torch.cuda.empty_cache()

    def hooks(self):
        return [(f"block{i:02d}", b) for i, b in enumerate(self.model.blocks)]

    @staticmethod
    def extract(name, out):
        return out

    def context(self, prompt_idx: int, text: str) -> dict:
        return dict(emb=self._ctx[prompt_idx])

    def forward(self, x, ctx):
        b = x.shape[0]
        self.model(hidden_states=x, timestep=self.t.expand(b), encoder_hidden_states=ctx["emb"].expand(b, -1, -1),
                   return_dict=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_cli_args(ap, batch=BATCH)
    ap.add_argument("--frames", type=int, default=FRAMES, help="clip length in pixel frames (4k + 1)")
    a = ap.parse_args()
    prompts = read_prompts(GENEVAL_553)
    ids = parse_ids(a.prompt_ids or PROBE_PROMPTS)
    bb = Backbone(a.arm, a.device, a.frames)
    bb.encode([(i, prompts[i]) for i in ids])
    P.run_cli(bb, a, prompts, ids, a.out_dir or C.output_dir("probe", a.arm),
              extra=dict(timestep=T_FIRST, frames=a.frames))


if __name__ == "__main__":
    main()
