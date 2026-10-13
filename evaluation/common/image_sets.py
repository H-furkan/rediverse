"""Image sample sets: one directory per arm and backbone.

    <images_dir>/<prompt_idx:03d>_s<seed>.png      e.g. 027_s0.png

`prompt_idx` is the 0-based line number in the prompt file (prompts/geneval_553.txt) and `seed`
the per-prompt sample index 0..R-1. `load_groups()` returns one PromptGroup per prompt, the unit
of every per-prompt diversity metric; quality metrics flatten the groups into (path, prompt)
pairs. Pairwise metrics are only comparable at a fixed R, so a set must hold exactly seeds
0..R-1 for every requested prompt; anything else raises.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from PIL import Image

from rediverse.data import GENEVAL_553, read_prompts

FNAME_RE = re.compile(r"^(\d+)_s(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)
SEEDS = 100                         # samples per prompt in the paper's image protocol


@dataclass
class PromptGroup:
    prompt_idx: int
    prompt: str
    paths: list[str]                # sorted by seed


def scan(images_dir: str) -> dict[int, dict[int, str]]:
    """{prompt_idx: {seed: path}} for every file named <idx>_s<seed>.<ext>."""
    by_idx: dict[int, dict[int, str]] = {}
    for name in os.listdir(images_dir):
        m = FNAME_RE.match(name)
        if m:
            by_idx.setdefault(int(m.group(1)), {})[int(m.group(2))] = os.path.join(images_dir, name)
    return by_idx


def load_groups(images_dir: str, prompts_file: str = GENEVAL_553, prompt_ids=None,
                seeds: int = SEEDS) -> list[PromptGroup]:
    """Prompt groups of `images_dir` in ascending prompt order.

    prompt_ids  the prompts to use (default: every line of `prompts_file`); files of other
                prompts are ignored.
    seeds       R: each used prompt must have exactly the files s0..s{R-1}.
    """
    prompts = read_prompts(prompts_file)
    by_idx = scan(images_dir)
    if not by_idx:
        raise ValueError(f"no images named '<idx>_s<seed>.png' in {images_dir}")
    bad = sorted(i for i in by_idx if i >= len(prompts))
    if bad:
        raise ValueError(f"prompt indices {bad[:5]} out of range for {prompts_file} ({len(prompts)} prompts)")
    ids = sorted(set(prompt_ids)) if prompt_ids is not None else list(range(len(prompts)))
    want = set(range(seeds))
    problems = []
    for i in ids:
        have = set(by_idx.get(i, {}))
        if have != want:
            miss, extra = sorted(want - have), sorted(have - want)
            problems.append(f"p{i:03d}: missing seeds {miss[:5]}{'...' if len(miss) > 5 else ''}"
                            f" extra seeds {extra[:5]}{'...' if len(extra) > 5 else ''}")
    if problems:
        raise ValueError(f"{images_dir} is not a {len(ids)} prompts x {seeds} seeds grid "
                         f"({len(problems)} prompts differ):\n  " + "\n  ".join(problems[:10]))
    return [PromptGroup(i, prompts[i], [by_idx[i][s] for s in range(seeds)]) for i in ids]


def flatten(groups: list[PromptGroup]) -> list[tuple[str, str]]:
    """(image_path, prompt) pairs, prompt-major and seed-minor, for per-image quality metrics."""
    return [(p, g.prompt) for g in groups for p in g.paths]


def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pairwise_indices(n: int) -> list[tuple[int, int]]:
    """All pairs i < j of n items, row-major."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def up_to_date(path: str, images: list[str]) -> bool:
    """True if the cache file `path` exists and is at least as new as every image it was computed from."""
    return os.path.exists(path) and os.path.getmtime(path) >= max(os.path.getmtime(p) for p in images)


def shard(items: list, i: int, n: int) -> list:
    """Items k with k % n == i (the --shard i --num-shards n convention of every stage)."""
    return [g for k, g in enumerate(items) if k % n == i]
