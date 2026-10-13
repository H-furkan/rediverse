"""VBench-300 video sets: prompts, prompt subsets and the per-prompt clip groups read by every video metric.

A set is a directory <set>/videos/<prompt>-<k>.mp4 (VBench naming, k = 0..9) holding 10 clips for each of the
300 prompts of prompts/vbench_300.txt. The prompt index used in every feature cache (p<idx>.np[yz]) is the
0-based line number of that file. Pairwise diversity grows with the number of clips per prompt, so the clip
count is constant and a missing clip is an error.

Prompt subsets: TPIPS aspects use prompts/vbench_subset100.txt (0, 3, ..., 297); camera trajectories use
prompts/vbench_subset30.txt (0, 10, ..., 290).
"""
from __future__ import annotations

import os

from rediverse.data import VBENCH_300, VBENCH_SUBSET30, VBENCH_SUBSET100, parse_ids, read_prompts
from rediverse.video_io import clip_path

N_PROMPTS = 300
N_PER_PROMPT = 10
SUBSETS = {"subset100": VBENCH_SUBSET100, "subset30": VBENCH_SUBSET30}


def prompts() -> list[str]:
    p = read_prompts(VBENCH_300)
    assert len(p) == N_PROMPTS, f"{VBENCH_300}: expected {N_PROMPTS} prompts, found {len(p)}"
    return p


def prompt_ids(spec: str | None = "all") -> list[int]:
    """Prompt indices from 'all', a subset name ('subset100', 'subset30'), a comma list / ranges, or a file."""
    if spec is None or spec == "all":
        return list(range(N_PROMPTS))
    ids = parse_ids(SUBSETS.get(spec, spec))
    assert all(0 <= i < N_PROMPTS for i in ids), f"prompt ids out of range 0..{N_PROMPTS - 1}: {spec}"
    return ids


def groups(videos_dir: str, ids: list[int] | None = None, n: int = N_PER_PROMPT):
    """[(prompt_idx, prompt, [n clip paths])] for the prompts in `ids` (default all 300), in index order.
    Raises FileNotFoundError when any of those clips is missing."""
    ps = prompts()
    out, missing = [], []
    for k in (range(N_PROMPTS) if ids is None else ids):
        paths = [clip_path(videos_dir, ps[k], i) for i in range(n)]
        missing += [x for x in paths if not os.path.exists(x)]
        out.append((k, ps[k], paths))
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {n * len(out)} clips missing under {videos_dir} "
                                f"(first: {os.path.basename(missing[0])}); every prompt needs clips 0..{n - 1}")
    return out


def shard(items: list, i: int, n: int) -> list:
    """Items k with k % n == i (the --shard i --num-shards n convention)."""
    return [g for k, g in enumerate(items) if k % n == i]
