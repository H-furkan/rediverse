#!/usr/bin/env python
"""Probe the variance characteristic of every MiniMax-H3 transformer block (paper Sec. 3.1, Tab. 1).

The MiniMax-H3 transformer (teacher) or the teacher with the Turbo LoRA (student) gets N video
noises of one prompt as the input of its first denoising step: one conditional forward at
sigma 1 with the pipeline's own step-1 row plan; the audio rows hold one frozen noise per prompt
(seed 424243 + prompt index), so only the video noise varies. The probed tensor of each of the 50
transformer_blocks is its video rows (the tail of the packed [text | audio | video] sequence,
37,296 rows x 5376 at 124 frames); the collapsed block l is block 45.

  python -m h3.probe --arm teacher --prompt-ids 27 --device cuda:0
  python -m h3.probe --arm student --prompt-ids 27 --device cuda:0
  python -m rediverse.probe_report --backbone minimax-h3

Default prompts: the paper's 20 video probe prompts (prompts/probe_video20.txt, GenEval indices),
N = 1000. The text contexts are encoded first and the 66 GB text encoder is freed before the
transformer is loaded. One block of one arm stages N x 200 M bf16 values, so a host with less than
about 1 TB of free memory should lower --n or probe a block window with --blocks.
"""
from __future__ import annotations

import argparse
import os

import torch

from h3 import config as C
from h3 import model as H
from rediverse import probe as P
from rediverse import seeds
from rediverse.data import GENEVAL_553, PROMPTS_DIR, parse_ids, read_prompts

PROBE_PROMPTS = os.path.join(PROMPTS_DIR, "probe_video20.txt")
BATCH = 2


class Backbone:
    name = C.BACKBONE
    dtype = H.DTYPE
    latent_shape = H.latent_shape()
    collapsed_block = f"block{C.ANCHOR_BLOCK:02d}"

    def __init__(self, arm: str, device, ctxs: dict, blocks: tuple = None):
        self.device = device
        self._ctx = ctxs
        self.blocks = blocks
        self.model = H.load_arm(arm, device)

    def hooks(self):
        pts = [(f"block{i:02d}", b) for i, b in enumerate(self.model.transformer_blocks)]
        if self.blocks:
            lo, hi = self.blocks
            pts = [p for p in pts if lo <= int(p[0][5:]) <= hi]
        return pts

    @staticmethod
    def extract(name, out):
        return out[:, -H.NUM_VIDEO_ROWS:, :]

    def context(self, prompt_idx: int, text: str) -> dict:
        ctx = self._ctx[prompt_idx]
        d = H.dev_ctx(ctx, self.device)
        d["timestep"] = ctx["timestep"].to(self.device)
        d["timestep_indices"] = ctx["timestep_indices"].to(self.device)
        return d

    def forward(self, x, d):
        b = x.shape[0]
        self.model(hidden_states=H.pack(x), audio_hidden_states=d["audio_rows"].expand(b, -1, -1),
                   encoder_hidden_states=d["prompt_embeds"].expand(b, -1, -1),
                   timestep=d["timestep"], timestep_indices=d["timestep_indices"],
                   token_tags=d["token_tags"], position_ids=d["position_ids"],
                   video_indices=d["video_indices"], audio_indices=d["audio_indices"],
                   text_indices=d["text_indices"], return_dict=False)


def encode_contexts(items, device) -> dict:
    """{prompt_idx: text context} with the frozen audio noise of seed 424243 + prompt_idx."""
    enc = H.TextEncoder(device)
    out = {i: enc.context(i, text, seeds.H3_AUDIO_SEED0 + i) for i, text in items}
    enc.close()
    del enc
    H.free()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_cli_args(ap, batch=BATCH)
    ap.add_argument("--blocks", default=None, help="probe a block window only, e.g. '40-49'")
    a = ap.parse_args()
    prompts = read_prompts(GENEVAL_553)
    ids = parse_ids(a.prompt_ids or PROBE_PROMPTS)
    todo = [i for i in ids if i % a.num_shards == a.shard]
    ctxs = encode_contexts([(i, prompts[i]) for i in todo], a.device)
    blocks = tuple(int(v) for v in a.blocks.split("-")) if a.blocks else None
    bb = Backbone(a.arm, a.device, ctxs, blocks)
    P.run_cli(bb, a, prompts, ids, a.out_dir or C.output_dir("probe", a.arm),
              extra=dict(frames=C.NUM_FRAMES, audio_seed_rule=f"{seeds.H3_AUDIO_SEED0} + prompt_idx"))


if __name__ == "__main__":
    main()
