"""Cache the SD3 text embeddings ReDiverse-FT training reads.

    python -m sd3.encode_prompts            # 100 caption files + the empty-prompt embedding

Writes <REDIVERSE_DATA>/sd3/ctx/p{ci:06d}.pt for the 100 training captions (prompts/
train_captions.txt) and <REDIVERSE_DATA>/sd3/uncond.pt for the empty prompt. Each file holds
prompt_embeds [1, 333, 4096] and pooled_projections [1, 2048] in bf16 from
StableDiffusion3Pipeline.encode_prompt (conditional only). The captions are encoded in batches of
8; the last batch is padded to 8 by repeating its final caption, so every caption is embedded
under the same batch shape (bf16 kernels depend on it). The empty-prompt embedding is checked
against the pipeline's own negative branch for negative_prompt="", which is the unconditional
input of the CFG-guided teacher rollout.
"""
from __future__ import annotations

import argparse
import gc
import os

import torch

from rediverse import data as D
from sd3.config import DTYPE, MODEL_ID, MODEL_REVISION, ctx_dir, uncond_path


def load_text_pipe(device: str):
    """StableDiffusion3Pipeline with only the three text encoders (no transformer, no VAE)."""
    from diffusers import StableDiffusion3Pipeline
    return StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, transformer=None, vae=None, torch_dtype=DTYPE).to(device)


def encode_uncond(pipe, out_path: str, device: str):
    """The empty prompt, encoded as the captions are; equal to the pipeline's negative branch."""
    with torch.no_grad():
        pe, _, pooled, _ = pipe.encode_prompt(
            prompt="", prompt_2=None, prompt_3=None, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=False)
        _, pe_n, _, pooled_n = pipe.encode_prompt(
            prompt="a photo of a dog", prompt_2=None, prompt_3=None, negative_prompt="",
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
    assert torch.equal(pe, pe_n) and torch.equal(pooled, pooled_n), \
        "encode_prompt('') differs from the pipeline's negative branch for negative_prompt=''"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(dict(prompt_embeds=pe.to(DTYPE).cpu(), pooled_projections=pooled.to(DTYPE).cpu(),
                    prompt=""), out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    print(f"[sd3] uncond {tuple(pe.shape)} {tuple(pooled.shape)} -> {out_path}", flush=True)


def encode_captions(pipe, texts: list[str], out_dir: str, device: str, batch: int = 8):
    """p{ci:06d}.pt for every caption, in batches of `batch` (the last one padded)."""
    os.makedirs(out_dir, exist_ok=True)
    for s in range(0, len(texts), batch):
        chunk = list(enumerate(texts))[s:s + batch]
        prompts = [t for _, t in chunk]
        prompts += [prompts[-1]] * (batch - len(prompts))
        with torch.no_grad():
            pe, _, pooled, _ = pipe.encode_prompt(
                prompt=prompts, prompt_2=None, prompt_3=None,
                device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        for j, (i, t) in enumerate(chunk):
            f = os.path.join(out_dir, f"p{i:06d}.pt")
            torch.save(dict(prompt_embeds=pe[j:j + 1].to(DTYPE).cpu(),
                            pooled_projections=pooled[j:j + 1].to(DTYPE).cpu(),
                            prompt=t, index=i), f + ".tmp")
            os.replace(f + ".tmp", f)
    print(f"[sd3] {len(texts)} captions {tuple(pe.shape[1:])} {tuple(pooled.shape[1:])} -> {out_dir}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--captions", default=None, help="default prompts/train_captions.txt")
    ap.add_argument("--out", default=None, help="default <REDIVERSE_DATA>/sd3/ctx")
    ap.add_argument("--uncond-out", default=None, help="default <REDIVERSE_DATA>/sd3/uncond.pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    texts = D.captions(a.captions)
    pipe = load_text_pipe(a.device)
    encode_captions(pipe, texts, a.out or ctx_dir(), a.device)
    encode_uncond(pipe, a.uncond_out or uncond_path(), a.device)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
