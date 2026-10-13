"""ReDiverse-FT training data, recipe constants and run bookkeeping (paper App. D).

SDXL, SD3 and Wan 2.1 share one recipe: the 100 DiffusionDB captions of
prompts/train_captions.txt, 16 paired noises per caption, 4 captions x 8 noises per
optimizer step for 200 steps (every pair used exactly 4 times), plain MSE, AdamW 1e-4,
gradient clipping at 10, LoRA rank 32 / alpha 32. The MiniMax-H3 adapter is trained on its
own target bank (h3/) with the same AdamW settings and clipping threshold.

  captions()             the 100 training captions
  ctx_index(ci)          where caption ci lives in a backbone's text-context cache: 10000 + ci,
                         a range disjoint from the GenEval-553 / VBench-300 evaluation indices
  noise_seed/noise/noise_batch   the paired training noise (rediverse.seeds)
  StepSampler            the per-step (caption, noises) composition
  shard_for_rank         how a step's 32 pairs split across data-parallel ranks
  make_optimizer         AdamW 1e-4 / wd 0.01 / betas (0.9, 0.999) / eps 1e-8
  clip_and_step          nan_to_num + clip_grad_norm_(10) + step
  dist_*                 torch.distributed helpers: LoRA grads are all-reduced (averaged), so a
                         data-parallel step equals the single-process step over the same pairs
  run_meta/write_run_meta  the config.json every trainer writes, with no absolute paths
  read_prompts/parse_ids/prompt_slug/prompt_dir/data_root/output_root
                         prompt files and output locations

Importing this module needs only the standard library, so the prompt and path helpers also
run without torch; torch and rediverse.seeds are imported where they are used.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(REPO_ROOT, "prompts")
TRAIN_CAPTIONS = os.path.join(PROMPTS_DIR, "train_captions.txt")
GENEVAL_553 = os.path.join(PROMPTS_DIR, "geneval_553.txt")
GENEVAL_SUBSET100 = os.path.join(PROMPTS_DIR, "geneval_subset100.txt")
VBENCH_300 = os.path.join(PROMPTS_DIR, "vbench_300.txt")
VBENCH_SUBSET100 = os.path.join(PROMPTS_DIR, "vbench_subset100.txt")
VBENCH_SUBSET30 = os.path.join(PROMPTS_DIR, "vbench_subset30.txt")
H3_TRAIN_IDS = os.path.join(PROMPTS_DIR, "h3_train_geneval_ids.txt")

# ------------------------------------------------------------ recipe constants --
RECIPE_NAME = "rediverse-ft"
N_CAPTIONS = 100                  # DiffusionDB lines 0-99
NOISES_PER_CAPTION = 16
CAPTIONS_PER_STEP = 4
SAMPLES_PER_STEP = 32             # 4 captions x 8 noises
STEPS = 200
LR = 1e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8
GRAD_CLIP = 10.0
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
REG_LOSS_WEIGHT = 1.0             # plain MSE, no other loss terms

# Teacher forward whose collapsed-block activation is the ReDiverse-FT target (tau).
# Wan 2.1: forward 13 of the CFG-distilled teacher's UniPC-50 (flow shift 5) schedule.
FT_TARGET_FORWARD = {"sdxl": 7, "sd3": 9, "wan": 13, "h3": 16}
# Teacher step at which the ReDiverse-TF directions are estimated (also stored as `k`
# in every direction file). Wan 2.1: step 8 of the base teacher's schedule.
TF_DIRECTION_STEP = {"sdxl": 7, "sd3": 9, "wan": 8, "h3": 16}


def recipe() -> dict:
    """The shared recipe constants as a plain dict (recorded in every run's config.json)."""
    return dict(name=RECIPE_NAME, n_captions=N_CAPTIONS, noises_per_caption=NOISES_PER_CAPTION,
                captions_per_step=CAPTIONS_PER_STEP, samples_per_step=SAMPLES_PER_STEP, steps=STEPS,
                optimizer="AdamW", lr=LR, weight_decay=WEIGHT_DECAY, betas=list(BETAS), eps=EPS,
                grad_clip=GRAD_CLIP, lr_schedule="constant, no warmup", lora_rank=LORA_RANK,
                lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, reg_loss_weight=REG_LOSS_WEIGHT,
                ft_target_forward=dict(FT_TARGET_FORWARD), tf_direction_step=dict(TF_DIRECTION_STEP))


# Names re-exported from rediverse.seeds, resolved on first access so that this module imports
# without torch.
_FROM_SEEDS = {"SEED0": "SEED0", "CTX_OFFSET": "CTX_OFFSET", "noise": "train_noise",
               "noise_batch": "train_noise_batch", "noise_seed": "train_noise_seed"}


def __getattr__(name: str):
    if name not in _FROM_SEEDS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from rediverse import seeds
    value = getattr(seeds, _FROM_SEEDS[name])
    globals()[name] = value
    return value


# ------------------------------------------------------------------ prompts --

def read_prompts(path: str) -> list[str]:
    """Non-empty, stripped lines of a prompt file; the line index is the prompt index."""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_ids(spec: str) -> list[int]:
    """Sorted unique prompt indices from a comma list with optional 'a-b' ranges, or from a
    file holding such a list (e.g. prompts/geneval_subset100.txt)."""
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            spec = f.read()
    out = set()
    for tok in spec.replace("\n", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok and not tok.startswith("-"):
            lo, hi = (int(v) for v in tok.split("-"))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def data_root() -> str:
    """REDIVERSE_DATA, default ./data."""
    return os.environ.get("REDIVERSE_DATA", "data")


def output_root() -> str:
    """REDIVERSE_OUT, default ./outputs."""
    return os.environ.get("REDIVERSE_OUT", "outputs")


def prompt_slug(text: str) -> str:
    """Directory name of a --prompt run: the lowercased text with every run of characters other
    than a-z and 0-9 replaced by '_' and leading/trailing '_' removed. A slug longer than 80
    characters is cut to 71 and followed by '_' and the first 8 hex digits of the SHA-1 of the
    full text."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if len(s) > 80:
        s = s[:71] + "_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return s or "prompt"


def prompt_dir(backbone: str, prompt: str, arm: str = None) -> str:
    """Output directory of a --prompt run: <REDIVERSE_OUT>/<backbone>/prompt/<slug>/<arm>.
    With arm=None, the prompt's own directory (holds directions.pt of the TF arms)."""
    d = os.path.join(output_root(), backbone, "prompt", prompt_slug(prompt))
    return os.path.join(d, arm) if arm else d


# ----------------------------------------------------------------- captions --

def captions(path: str = None) -> list[str]:
    """The 100 training captions (prompts/train_captions.txt, written by
    prompts/download_train_captions.py), line i = caption ci."""
    path = path or TRAIN_CAPTIONS
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run: python prompts/download_train_captions.py")
    with open(path, encoding="utf-8") as f:
        out = [ln.rstrip("\n") for ln in f]
    assert len(out) == N_CAPTIONS and all(out), f"expected exactly {N_CAPTIONS} non-empty captions in {path}"
    return out


def n_captions() -> int:
    return N_CAPTIONS


def ctx_index(ci: int) -> int:
    from rediverse.seeds import CTX_OFFSET
    return CTX_OFFSET + int(ci)


# ------------------------------------------------------------------ sampler --

class StepSampler:
    """Yields, per optimizer step, a list of (ci, [ni, ...]) micro-batches.

    caps_per_step captions, noises_per_draw noises each. A reshuffled pass over the
    captions; within a caption a reshuffled cyclic pass over its noises. With 100 x 16
    pairs and 4 x 8 per step, 200 steps is exactly 4 passes over every pair.
    """

    def __init__(self, seed: int = 0, n_caps: int = None, n_noises: int = None,
                 caps_per_step: int = None, noises_per_draw: int = None):
        self.n_caps = n_caps or n_captions()
        self.n_noises = n_noises or NOISES_PER_CAPTION
        self.caps_per_step = caps_per_step or CAPTIONS_PER_STEP
        self.noises_per_draw = noises_per_draw or (SAMPLES_PER_STEP // self.caps_per_step)
        assert self.caps_per_step * self.noises_per_draw == SAMPLES_PER_STEP, \
            "caps_per_step x noises_per_draw must equal SAMPLES_PER_STEP"
        self.rng = random.Random(seed)
        self.pos = {}
        for ci in range(self.n_caps):
            idx = list(range(self.n_noises)); self.rng.shuffle(idx)
            self.pos[ci] = [idx, 0]
        self.order, self.oi = [], 0

    def _next_caption(self) -> int:
        if self.oi >= len(self.order):
            self.order = list(range(self.n_caps)); self.rng.shuffle(self.order); self.oi = 0
        ci = self.order[self.oi]; self.oi += 1
        return ci

    def _draw(self, ci: int) -> list[int]:
        idx, i = self.pos[ci]
        take = []
        while len(take) < self.noises_per_draw:
            if i >= len(idx):
                self.rng.shuffle(idx); i = 0
            take.append(idx[i]); i += 1
        self.pos[ci] = [idx, i]
        return take

    def step(self) -> list[tuple[int, list[int]]]:
        return [(ci, self._draw(ci)) for ci in [self._next_caption() for _ in range(self.caps_per_step)]]

    def steps(self, n: int):
        for _ in range(n):
            yield self.step()


def flatten(step_batches) -> list[tuple[int, int]]:
    return [(ci, ni) for ci, nis in step_batches for ni in nis]


def shard_for_rank(step_batches, rank: int, world: int) -> list[tuple[int, int]]:
    """Pairs [rank::world] of the step's flattened list. Every rank calls the sampler
    identically (same seed), so the split is consistent without any communication;
    grads are averaged, so the step equals the single-process one."""
    pairs = flatten(step_batches)
    assert len(pairs) % world == 0, f"{len(pairs)} pairs not divisible by world {world}"
    return pairs[rank::world]


def group_by_caption(pairs) -> list[tuple[int, list[int]]]:
    out = {}
    for ci, ni in pairs:
        out.setdefault(ci, []).append(ni)
    return list(out.items())


# ---------------------------------------------------------------- optimizer --

def make_optimizer(params):
    import torch
    return torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS, eps=EPS)


def clip_and_step(params, opt) -> tuple[float, int]:
    """nan_to_num, clip to GRAD_CLIP, step. Returns (grad norm, n_nonfinite)."""
    import torch
    n_bad = 0
    for p in params:
        if p.grad is not None:
            bad = p.grad.numel() - int(p.grad.isfinite().sum())
            if bad:
                n_bad += bad
                torch.nan_to_num(p.grad, nan=0.0, posinf=None, neginf=None, out=p.grad)
    gnorm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP).item()
    opt.step()
    return gnorm, n_bad


# --------------------------------------------------------------------- dist --

def dist_init() -> tuple[int, int]:
    """(rank, world). World 1 when not launched under torchrun."""
    if "RANK" not in os.environ:
        return 0, 1
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size()


def dist_allreduce_grads(params, world: int):
    if world == 1:
        return
    import torch.distributed as dist
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)


def dist_allreduce_scalar(x: float, world: int) -> float:
    if world == 1:
        return x
    import torch
    import torch.distributed as dist
    t = torch.tensor([x], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / world


def dist_barrier(world: int):
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


# ----------------------------------------------------------------- metadata --

def portable(obj, root: str = None):
    """Copy of `obj` (dicts, lists, tuples, strings) in which absolute paths are made relative
    to `root` (default: the working directory) or, outside it, reduced to their file name."""
    root = os.path.abspath(root or os.getcwd())
    if isinstance(obj, dict):
        return {k: portable(v, root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(portable(v, root) for v in obj)
    if isinstance(obj, str) and os.path.isabs(obj):
        p = os.path.abspath(obj)
        return os.path.relpath(p, root) if p == root or p.startswith(root + os.sep) else os.path.basename(p)
    return obj


def run_meta(family: str, extra: dict = None) -> dict:
    """What every run's config.json records, so runs are comparable by inspection."""
    from rediverse.seeds import CTX_OFFSET, SEED0
    m = dict(recipe=RECIPE_NAME, family=family, spec=recipe(),
             captions=f"prompts/train_captions.txt (DiffusionDB lines 0-{N_CAPTIONS - 1})",
             seed_rule="SEED0 + (10000+ci)*100003 + ni", seed0=SEED0, ctx_offset=CTX_OFFSET)
    m.update(extra or {})
    return portable(m)


def write_run_meta(path: str, meta: dict):
    """Write `meta` as JSON (indent 2, non-JSON values as str) with absolute paths removed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(portable(meta), f, indent=2, default=str)
