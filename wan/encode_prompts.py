#!/usr/bin/env python
"""UMT5 text contexts for Wan 2.1 (run in the `video` environment).

  train   the 100 training captions (prompts/train_captions.txt) -> ctx/train/p{10000+ci}.pt
  vbench  the VBench-300 prompts (prompts/vbench_300.txt)         -> ctx/vbench/v{idx:03d}.pt
  neg     the evaluation teacher's negative prompt                 -> ctx/neg.pt

Every file holds encoder_hidden_states [1, 512, 4096] bf16 (WanPipeline convention:
prompt_clean, max length 512, positions past the prompt zeroed), seq_len and the text.
Existing files are kept unless --force.

  python -m wan.encode_prompts --set train,vbench,neg --device cuda:0
"""
from __future__ import annotations

import argparse
import os

import torch

from rediverse import data
from wan import config as C
from wan import model as M


def _save(path: str, emb, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert emb.shape == (1, C.TEXT_LEN, C.TEXT_DIM) and emb.dtype == M.DTYPE, (emb.shape, emb.dtype)
    torch.save(dict(encoder_hidden_states=emb.cpu(), **meta), path + ".tmp")
    os.replace(path + ".tmp", path)


def jobs(sets, captions_path: str = None, vbench_path: str = None):
    """[(output path, text, metadata)] of the requested sets."""
    out = []
    if "train" in sets:
        for ci, text in enumerate(data.captions(captions_path)):
            idx = data.ctx_index(ci)
            out.append((os.path.join(C.train_ctx_dir(), f"p{idx}.pt"), text,
                        dict(prompt_idx=idx, caption_idx=ci, prompt=text)))
    if "vbench" in sets:
        for i, text in enumerate(data.read_prompts(vbench_path or data.VBENCH_300)):
            out.append((os.path.join(C.vbench_ctx_dir(), f"v{i:03d}.pt"), text, dict(prompt_idx=i, prompt=text)))
    if "neg" in sets:
        out.append((C.neg_ctx_path(), M.NEG_PROMPT, dict(prompt=M.NEG_PROMPT)))
    return out


def encode(pending, device):
    """Encode [(path, text, meta)] with one text-encoder load; frees it afterwards."""
    if not pending:
        return
    tok, enc = M.text_encoder(device)
    for n, (path, text, meta) in enumerate(pending):
        emb, seq_len = M.embed(tok, enc, text, device)
        _save(path, emb, seq_len=seq_len, **meta)
        if n % 25 == 0 or n == len(pending) - 1:
            print(f"[encode] {n + 1}/{len(pending)} {os.path.basename(path)} seq_len={seq_len} {text[:60]!r}", flush=True)
    del enc
    torch.cuda.empty_cache()


def ensure(sets, device, vbench_path: str = None):
    """Encode whatever of `sets` is not on disk yet (used by generate / estimate_directions)."""
    encode([j for j in jobs(sets, vbench_path=vbench_path) if not os.path.exists(j[0])], device)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="train,vbench,neg", help="comma list of train, vbench, neg")
    ap.add_argument("--captions", default=None, help=f"training caption file (default {data.TRAIN_CAPTIONS})")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true", help="re-encode existing files")
    a = ap.parse_args()
    sets = [s.strip() for s in a.set.split(",") if s.strip()]
    assert set(sets) <= {"train", "vbench", "neg"}, sets
    pending = [j for j in jobs(sets, a.captions) if a.force or not os.path.exists(j[0])]
    print(f"[encode] {len(pending)} contexts to encode ({', '.join(sets)})", flush=True)
    encode(pending, a.device)


if __name__ == "__main__":
    main()
