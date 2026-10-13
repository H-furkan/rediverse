"""Build the frozen SD3 evaluation noise bank (553 GenEval prompts x 100 samples).

    python -m sd3.make_noise_bank            # about 29 GB, a few minutes on one GPU

Sample s of prompt i is torch.randn((1, 16, 128, 128), generator=torch.Generator("cuda")
.manual_seed(i * 1000 + s), device="cuda", dtype=torch.bfloat16), which is what
StableDiffusion3Pipeline.prepare_latents draws for that seed. CUDA draws depend on the GPU's
launch configuration, so the bank (built on an H100) is the reference input of every arm; a CPU
generator gives different noise. numpy has no bfloat16, so each array stores the bf16 bit pattern
as int16 and is read back with torch.from_numpy(arr).view(torch.bfloat16), never with a numeric
cast. Layout: one key per prompt, "000".."552", each (100, 16, 128, 128) int16, plus a .json
sidecar describing it.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile

import numpy as np
import torch

from rediverse.seeds import image_eval_noise
from sd3.config import DTYPE, LATENT_SHAPE, NUM_SAMPLES, noise_bank_path

N_PROMPTS = 553


def draw(i: int, s: int, device: str) -> torch.Tensor:
    """Bank entry (i, s): the image evaluation rule at SD3's shape and dtype."""
    return image_eval_noise(i, s, (1,) + LATENT_SHAPE, DTYPE, device)


def prompt_block(i: int, num_samples: int, device: str) -> np.ndarray:
    """(num_samples, 16, 128, 128) int16 bit patterns of prompt i."""
    x = torch.cat([draw(i, s, device) for s in range(num_samples)])
    return x.view(torch.int16).cpu().numpy()


def load_bank(path: str, num_samples: int = NUM_SAMPLES):
    """get(i, s0, s1) -> bf16 CPU tensor [s1 - s0, 16, 128, 128] from a bank written here."""
    bank = np.load(path)
    probe = bank[bank.files[0]]
    assert probe.dtype == np.int16, f"{path}: arrays are {probe.dtype}, expected int16 bf16 bit patterns"
    assert probe.shape[1:] == LATENT_SHAPE, f"{path}: latent {probe.shape[1:]} != {LATENT_SHAPE}"
    assert probe.shape[0] >= num_samples, f"{path}: {probe.shape[0]} samples/prompt < {num_samples}"
    cache = {}

    def get(i: int, s0: int, s1: int) -> torch.Tensor:
        key = f"{i:03d}"
        if key not in cache:
            cache.clear()
            cache[key] = bank[key]
        arr = cache[key][s0:s1]
        assert arr.dtype == np.int16 and arr.shape[1:] == LATENT_SHAPE
        return torch.from_numpy(np.ascontiguousarray(arr)).view(torch.bfloat16)

    return get


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="default <REDIVERSE_DATA>/sd3/noise_bank_553x100.npz")
    ap.add_argument("--num-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = a.out or noise_bank_path()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    # The same container np.savez writes (stored zip of .npy members), filled one prompt at a time.
    tmp = out + ".tmp.npz"
    with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i in range(a.num_prompts):
            with zf.open(f"{i:03d}.npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, prompt_block(i, a.num_samples, a.device), allow_pickle=False)
            if (i + 1) % 100 == 0:
                print(f"  drew {i + 1}/{a.num_prompts} prompts", flush=True)
    os.replace(tmp, out)

    # Reload and compare probe entries with fresh draws, as bf16 after the view.
    reloaded = np.load(out)
    assert len(reloaded.files) == a.num_prompts, f"{len(reloaded.files)} keys, expected {a.num_prompts}"
    probes = [(0, 0), (0, a.num_samples - 1), (a.num_prompts // 2, 3), (a.num_prompts - 1, 7)]
    for i, s in [(min(i, a.num_prompts - 1), min(s, a.num_samples - 1)) for i, s in probes]:
        arr = reloaded[f"{i:03d}"]
        assert arr.dtype == np.int16 and arr.shape == (a.num_samples,) + LATENT_SHAPE, (arr.dtype, arr.shape)
        got = torch.from_numpy(arr[s:s + 1]).view(DTYPE)
        assert torch.equal(got, draw(i, s, a.device).cpu()), f"prompt {i} sample {s} differs from a fresh draw"
        assert float(got.float().std()) > 0.9, f"prompt {i} sample {s} is not unit noise"
    meta = dict(backbone="sd3", latent_shape=list(LATENT_SHAPE), stored_dtype="int16",
                true_dtype="bfloat16", read_back="torch.from_numpy(arr).view(torch.bfloat16)",
                seed_rule="prompt_idx * 1000 + sample_idx", num_prompts=a.num_prompts,
                num_samples=a.num_samples,
                draw="torch.randn((1,16,128,128), generator=torch.Generator('cuda')"
                     ".manual_seed(i*1000+s), device='cuda', dtype=torch.bfloat16)")
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out}: {a.num_prompts} prompts x {a.num_samples} samples x {LATENT_SHAPE} "
          f"bf16 as int16, {os.path.getsize(out) / 1e9:.2f} GB, self-check passed", flush=True)


if __name__ == "__main__":
    main()
