# SPDX-License-Identifier: Apache-2.0
# Added by ReDiverse on top of NVlabs/FastGen (commit 060b857); not part of upstream FastGen.
"""ReDiverse-FT training data for SDXL: one caption with K paired noises per micro-batch.

The 100 training captions of prompts/train_captions.txt are pre-encoded by sdxl/encode_captions.py
into <ctx_dir>/p{ci:03d}.pt (prompt embeddings [77, 2048] and pooled embeddings [1280], float16,
with the caption text, which must match the caption file). Each optimizer step uses
4 captions x 8 noises (rediverse.data.StepSampler), delivered as 4 micro-batches, one per
FastGen gradient-accumulation round; noise ni of caption ci is the seeded draw
rediverse.data.noise(ci, ni) (seed 1_000_000 + (10000 + ci) * 100003 + ni). The teacher target
is computed online from the same noise, so no pre-generated data is read.
"""
from __future__ import annotations

import os

import torch

import fastgen.utils.logging_utils as logger

from rediverse.data import N_CAPTIONS, StepSampler, captions, noise_batch


class ReDiverseFTLoader:
    """Infinite iterator of micro-batches {"noise": [K, 4, 128, 128] float32,
    "condition": [prompt embeddings [K, 77, 2048], pooled embeddings [K, 1280]] float16}.

    batch_size is the number of captions per micro-batch (FastGen derives the number of
    gradient-accumulation rounds from it); it must be 1."""

    def __init__(self, ctx_dir: str, batch_size: int = 1, captions_per_step: int = 4,
                 noises_per_caption: int = 8, seed: int = 0, input_shape=(4, 128, 128), log_first: int = 4,
                 log_every: int = 200):
        assert batch_size == 1, "one caption per micro-batch"
        self.batch_size = batch_size
        self.sampler = StepSampler(seed=seed, caps_per_step=captions_per_step, noises_per_draw=noises_per_caption)
        self.input_shape = tuple(input_shape)
        self.log_first, self.log_every = log_first, log_every
        self.ctx = []
        caps = captions()
        for ci in range(N_CAPTIONS):
            p = os.path.join(ctx_dir, f"p{ci:03d}.pt")
            assert os.path.exists(p), f"training context missing: {p} (run python -m sdxl.encode_captions)"
            d = torch.load(p, map_location="cpu", weights_only=False)
            assert d["ci"] == ci and tuple(d["pe"].shape) == (77, 2048) and tuple(d["pool"].shape) == (1280,)
            assert d.get("prompt") == caps[ci], (f"{p} holds {d.get('prompt')!r}, caption {ci} is {caps[ci]!r}; "
                                                 "re-encode with python -m sdxl.encode_captions")
            self.ctx.append((d["pe"], d["pool"]))
        logger.info(f"ReDiverse-FT data: {N_CAPTIONS} captions from {ctx_dir}, {captions_per_step} captions x "
                    f"{noises_per_caption} noises per step, sampler seed {seed}")

    def __iter__(self):
        n = 0
        while True:
            for ci, nis in self.sampler.step():
                n += 1
                if n <= self.log_first or n % self.log_every == 0:
                    logger.info(f"micro-batch {n}: caption {ci} noises {list(nis)}")
                k = len(nis)
                pe, pool = self.ctx[ci]
                yield {"noise": noise_batch([(ci, ni) for ni in nis], self.input_shape),
                       "condition": [pe.unsqueeze(0).expand(k, -1, -1).contiguous(),
                                     pool.unsqueeze(0).expand(k, -1).contiguous()]}
