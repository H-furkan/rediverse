#!/usr/bin/env python
"""Text contexts of MiniMax-H3 prompts (Qwen3-VL, about 66 GB, runs alone on its GPU).

A context holds the prompt embeddings, the packed-sequence layout, the step-1 row plan and the
prompt's frozen audio-row noise. Two prompt sets:

  train   the 20 GenEval prompts of the ReDiverse-FT target bank (prompts/h3_train_geneval_ids.txt
          over prompts/geneval_553.txt) -> <data>/ctx/train/p{idx:03d}.pt, audio seed 424243 + idx
  vbench  the 300 VBench prompts (prompts/vbench_300.txt)
          -> <data>/ctx/vbench/v{idx:03d}.pt, audio seed 434243 + idx

  python -m h3.encode_prompts --set train --device cuda:0
  python -m h3.encode_prompts --set vbench --device cuda:0
"""
from __future__ import annotations

import argparse
import os

import torch

from h3 import config as C
from h3.model import TextEncoder, free
from rediverse import data, seeds


def train_entries(prompt_ids=None, out_dir=None):
    """(path, prompt_idx, text, audio_seed) of the ReDiverse-FT training prompts."""
    texts = data.read_prompts(data.GENEVAL_553)
    ids = prompt_ids if prompt_ids is not None else C.ft_prompt_ids()
    out_dir = out_dir or C.train_ctx_dir()
    return [(os.path.join(out_dir, f"p{p:03d}.pt"), p, texts[p], seeds.h3_train_audio_seed(p)) for p in ids]


def vbench_entries(prompt_ids=None, out_dir=None):
    """(path, prompt_idx, text, audio_seed) of the VBench prompts."""
    texts = data.read_prompts(data.VBENCH_300)
    ids = prompt_ids if prompt_ids is not None else range(len(texts))
    out_dir = out_dir or C.vbench_ctx_dir()
    return [(os.path.join(out_dir, f"v{i:03d}.pt"), i, texts[i], seeds.h3_eval_audio_seed(i)) for i in ids]


def prompt_context(text: str, prompt_idx: int, device) -> dict:
    """Text context of one free-form prompt, with the audio noise of evaluation prompt
    `prompt_idx`; the text encoder is released before returning."""
    enc = TextEncoder(device)
    ctx = enc.context(prompt_idx, text, seeds.h3_eval_audio_seed(prompt_idx))
    enc.close()
    del enc
    free()
    return ctx


def encode(entries, device):
    """Write the contexts of `entries` that do not exist yet."""
    pending = [e for e in entries if not os.path.exists(e[0])]
    print(f"[h3/encode] {len(pending)}/{len(entries)} contexts to encode", flush=True)
    if not pending:
        return
    enc = TextEncoder(device)
    for n, (path, idx, text, audio_seed) in enumerate(pending):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ctx = enc.context(idx, text, audio_seed)
        torch.save(ctx, path + ".tmp")
        os.replace(path + ".tmp", path)
        if n % 25 == 0 or n == len(pending) - 1:
            print(f"  [{n + 1}/{len(pending)}] {os.path.basename(path)} "
                  f"({ctx['text_indices'].numel()} text tokens) {text[:50]!r}", flush=True)
    enc.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["train", "vbench"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt-ids", default="", help="subset of prompt indices, e.g. '0,27' or '0-9' (default: all)")
    ap.add_argument("--out-dir", default="", help="output directory (default <data>/ctx/<set>)")
    a = ap.parse_args()
    ids = data.parse_ids(a.prompt_ids) if a.prompt_ids else None
    entries = (train_entries if a.set == "train" else vbench_entries)(ids, a.out_dir or None)
    encode(entries, a.device)
    print("[h3/encode] done", flush=True)


if __name__ == "__main__":
    main()
