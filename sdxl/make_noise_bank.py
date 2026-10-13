#!/usr/bin/env python
"""Build the SDXL evaluation noise bank: the initial latent of every (GenEval prompt i, sample s),
frozen to an .npz so that all five arms consume identical noise.

Each draw replicates diffusers' randn_tensor with one CUDA generator per sample:
torch.randn((1, 4, 128, 128), generator=cuda Generator seeded i*1000 + s, dtype=float16)
(rediverse.seeds.image_eval_noise). CUDA and CPU generators give different streams, so the bank
must be built on a GPU. Layout: one key per prompt, "000".."552", each (num_samples, 4, 128, 128)
float16 (7.25 GB for 553 x 100); np.load reads keys lazily.

  python -m sdxl.make_noise_bank                     # -> data/sdxl/noise_bank_553x100.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from rediverse.data import parse_ids
from rediverse.seeds import image_eval_noise
from sdxl import config as C


def draw(i: int, s: int, device: str) -> torch.Tensor:
    return image_eval_noise(i, s, (1, *C.LATENT_SHAPE), torch.float16, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=C.noise_bank_path())
    ap.add_argument("--num-prompts", type=int, default=C.NUM_PROMPTS)
    ap.add_argument("--num-samples", type=int, default=C.NUM_SAMPLES)
    ap.add_argument("--prompt-ids", default=None, help="only these prompt indices, e.g. '27' or '0-9,27'")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ids = parse_ids(a.prompt_ids) if a.prompt_ids else list(range(a.num_prompts))
    bank = {f"{i:03d}": torch.cat([draw(i, s, a.device) for s in range(a.num_samples)]).cpu().numpy()
            for i in ids}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, **bank)
    os.replace(tmp, a.out)

    # self-check: reload a few entries and compare against fresh draws
    reloaded = np.load(a.out)
    for i, s in [(ids[0], 0), (ids[0], a.num_samples - 1), (ids[len(ids) // 2], 3 % a.num_samples),
                 (ids[-1], 7 % a.num_samples)]:
        fresh = draw(i, s, a.device).cpu().numpy()
        assert np.array_equal(reloaded[f"{i:03d}"][s:s + 1], fresh), \
            f"bank differs from a fresh draw at prompt {i} sample {s}"
    print(f"wrote {a.out}: {len(ids)} prompts x {a.num_samples} samples, "
          f"{os.path.getsize(a.out) / 1e9:.2f} GB, self-check passed")


if __name__ == "__main__":
    main()
