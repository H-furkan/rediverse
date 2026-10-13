"""The ReDiverse-FT target bank: precomputed teacher activations, one file per (caption, latent).

The target of a training pair is a pure function of the caption and the initial latent: the
CFG-distilled teacher's blocks.28 output at forward 13 of its own UniPC-50 rollout (12 full
forwards of a 14B model on 32,760 tokens, then a forward truncated after the block). It is
computed once by wan.build_targets and only read by wan.train, so the trainer never loads a
teacher.

On disk: <bank>/p{10000+ci}_i{ni:03d}.safetensors with
    h       bfloat16 [32760, 5120]          the target activation
    latent  float32  [16, 21, 60, 104]      the initial latent it was rolled from (seed rule)
(328 MiB per entry, 512 GiB for 100 captions x 16 latents) and a manifest.json that records the
geometry, the site, the teacher forward and its sigmas.
"""
from __future__ import annotations

import json
import os
import re

import torch

from rediverse.seeds import PROMPT_STRIDE, SEED0
from wan import config as C
from wan import model as M


def entry_path(d: str, prompt_idx: int, i: int) -> str:
    return os.path.join(d, f"p{prompt_idx:03d}_i{i:03d}.safetensors")


def write_manifest(d: str, extra: dict = None) -> dict:
    sig = M.teacher_target_sigmas(C.FT_TARGET_FORWARD, C.CFGD_FLOW_SHIFT)
    man = dict(family="wan2.1-t2v-14b", teacher=C.TEACHER_REPO, teacher_revision=C.TEACHER_REVISION,
               teacher_adapter=C.CFGD_REPO, teacher_adapter_revision=C.CFGD_REVISION,
               frames=C.FRAMES, latent_shape=list(C.LATENT_SHAPE), tokens=C.TOKENS, dim=C.DIM,
               dtype="bfloat16", anchor_site=f"blocks.{C.ANCHOR_BLOCK}",
               teacher_step=C.FT_TARGET_FORWARD, teacher_steps=C.TEACHER_STEPS,
               scheduler=f"UniPCMultistepScheduler, flow_shift {C.CFGD_FLOW_SHIFT}",
               teacher_flow_shift=C.CFGD_FLOW_SHIFT,
               guidance="1 (CFG-distilled teacher, one conditional forward per step)",
               seed_rule=f"{SEED0} + prompt*{PROMPT_STRIDE} + i",
               bytes_per_entry=C.TOKENS * C.DIM * 2 + 16 * C.LATENT_SHAPE[1] * 60 * 104 * 4,
               **{k: v for k, v in sig.items() if k != "student_sigmas"},
               student_sigmas=sig["student_sigmas"], **(extra or {}))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def load_manifest(d: str) -> dict:
    """The bank's manifest, refused unless its geometry, site and teacher forward are the recipe's."""
    path = os.path.join(d, "manifest.json")
    assert os.path.exists(path), f"no manifest.json in {d}: build the bank first (python -m wan.build_targets)"
    with open(path) as f:
        man = json.load(f)
    assert man["frames"] == C.FRAMES, (man["frames"], C.FRAMES)
    assert tuple(man["latent_shape"]) == C.LATENT_SHAPE, (man["latent_shape"], C.LATENT_SHAPE)
    assert man["tokens"] == C.TOKENS, (man["tokens"], C.TOKENS)
    assert man["anchor_site"] == f"blocks.{C.ANCHOR_BLOCK}", man["anchor_site"]
    assert man["teacher_step"] == C.FT_TARGET_FORWARD, (
        f"the bank at {d} holds teacher forward {man['teacher_step']}, the recipe trains on "
        f"forward {C.FT_TARGET_FORWARD}")
    return man


def save_entry(path: str, h: torch.Tensor, latent: torch.Tensor):
    from safetensors.torch import save_file

    assert h.dtype == M.DTYPE and h.shape == (C.TOKENS, C.DIM), (h.dtype, tuple(h.shape))
    assert latent.dtype == torch.float32 and tuple(latent.shape) == C.LATENT_SHAPE, (latent.dtype, tuple(latent.shape))
    tmp = path + ".tmp"
    save_file({"h": h.contiguous().cpu(), "latent": latent.contiguous().cpu()}, tmp)
    os.replace(tmp, path)


def read_entry(path: str) -> tuple:
    from safetensors.torch import load_file

    d = load_file(path)
    h, lat = d["h"], d["latent"]
    assert h.shape == (C.TOKENS, C.DIM) and h.dtype == M.DTYPE, (tuple(h.shape), h.dtype)
    assert tuple(lat.shape) == C.LATENT_SHAPE, tuple(lat.shape)
    return h, lat


class RamCache:
    """The bank as {caption index: [latent indices]}, entries kept in host RAM after the first
    read (up to `max_gib`; past it, later misses are read from disk again)."""

    def __init__(self, d: str, max_gib: float = 400.0):
        self.dir = d
        self.manifest = load_manifest(d)
        self.max_bytes = int(max_gib * 2 ** 30)
        self._c: dict = {}
        self._bytes = 0
        self.index = self._scan()
        assert self.index, f"bank {d} has a manifest but no entries"
        self.hits = self.misses = 0

    def _scan(self) -> dict:
        out: dict = {}
        for f in sorted(os.listdir(self.dir)):
            m = re.match(r"p(\d+)_i(\d+)\.safetensors$", f)
            if m:
                out.setdefault(int(m.group(1)), []).append(int(m.group(2)))
        return {p: sorted(v) for p, v in sorted(out.items())}

    def n_pairs(self) -> int:
        return sum(len(v) for v in self.index.values())

    def get(self, prompt_idx: int, i: int) -> tuple:
        key = (prompt_idx, i)
        v = self._c.get(key)
        if v is not None:
            self.hits += 1
            return v
        h, lat = read_entry(entry_path(self.dir, prompt_idx, i))
        self.misses += 1
        nb = h.numel() * 2 + lat.numel() * 4
        if self._bytes + nb <= self.max_bytes:
            self._c[key] = (h, lat)
            self._bytes += nb
        return h, lat

    def batch(self, prompt_idx: int, idxs) -> tuple:
        """(latents [K, 16, 21, 60, 104] fp32, targets [K, 32760, 5120] bf16) of one caption."""
        got = [self.get(prompt_idx, int(i)) for i in idxs]
        return torch.stack([g[1] for g in got]), torch.stack([g[0] for g in got])

    def stats(self) -> str:
        return f"{self.n_pairs()} pairs, {self._bytes / 2 ** 30:.1f} GiB resident"
