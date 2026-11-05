"""Activation probing: the variance characteristic of every block (paper Sec. 3.1, App. A).

For one prompt, N initial noises are fed to a model as the input of its first denoising step
(conditional branch only, one forward, no sampler step). At every hooked block each noise gives
one activation; flattened and centred over the noises they form X_b [N, D]. The Gram matrix
G_b = X_b X_b^T / N has the same nonzero eigenvalues lambda_j as the covariance
Sigma_b = X_b^T X_b / N, so with D in the millions everything is computed from the N x N Gram:

    TV   = Tr(Sigma_b) = sum_j lambda_j                  total variance
    PR   = (sum_j lambda_j)^2 / sum_j lambda_j^2         participation ratio, the effective
                                                         number of variance-carrying directions
    Conc = TV / PR                                       variance per effective direction

Teacher and student receive the same noises (seed 1_000_000 + prompt_idx * 100003 + i, CPU
float32, cast to the model's dtype), so their values compare block by block. Distillation keeps
TV but lowers PR from an intermediate block on: the collapsed block l of ReDiverse.

Activations are staged on the host in the model's compute dtype; the Gram is accumulated in
float64 on the GPU over feature chunks, each chunk centred over the noises before the product
(no cancellation in the collapsed blocks). PR comes from the exact eigenvalues of the centred
Gram. When the activations of all hooks do not fit the host-memory budget, the hooks are split
into contiguous groups and the N forwards are repeated per group; each forward stops after the
last block of its group. The values do not depend on the grouping.

A backbone module (sdxl.probe, sd3.probe, wan.probe, h3.probe) provides a Backbone object with
  hooks()          [(name, module)] in forward-execution order
  extract(n, out)  the probed tensor of a hooked module's output, [B, ...]
  context(i, t)    the conditioning of prompt i with text t
  forward(x, ctx)  one first-step forward of a batch of noises x
  latent_shape     shape of one initial noise
and run_prompt() below does the rest. rediverse.probe_report aggregates the per-prompt records.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from rediverse.seeds import SEED0, PROMPT_STRIDE, cpu_randn, prompt_seed

N_DEFAULT = 1000                  # noises per prompt (paper App. A)
TOP_EIGS = 32                     # leading eigenvalues kept in each record
GRAM_GPU_BYTES = 6 << 30          # float64 feature chunk on the GPU
RAM_BUDGET_GB = float(os.environ.get("REDIVERSE_PROBE_RAM_GB", "256"))
SEED_RULE = f"{SEED0} + prompt_idx * {PROMPT_STRIDE} + i"


class _Stop(Exception):
    """Raised by the hook of a group's deepest block: the rest of the forward is not needed."""


def probe_noise(prompt_idx: int, i: int, shape) -> torch.Tensor:
    """Initial noise i of prompt prompt_idx, shared by the teacher and the student."""
    return cpu_randn(prompt_seed(prompt_idx, i), shape)


# ------------------------------------------------------------------ statistics --
def centred_gram(X: torch.Tensor, device, gpu_bytes: int = GRAM_GPU_BYTES) -> torch.Tensor:
    """Xc Xc^T in float64 on `device`, Xc = X centred over its rows; X [N, D] on the host."""
    n, d = X.shape
    chunk = max(1024, int(gpu_bytes / (n * 8 * 2)))
    G = torch.zeros(n, n, dtype=torch.float64, device=device)
    for s in range(0, d, chunk):
        xb = X[:, s:s + chunk].to(device, non_blocking=True).double()
        xb -= xb.mean(0, keepdim=True)
        G += xb @ xb.T
        del xb
    return G


def spectrum(Gc: torch.Tensor) -> np.ndarray:
    """Eigenvalues of the centred Gram, clipped at 0, ascending (the order the sums use)."""
    try:
        lam = torch.linalg.eigvalsh(Gc)
    except RuntimeError as e:                 # cuSOLVER without a free handle on a full card
        if "cusolver" not in str(e).lower():
            raise
        lam = torch.linalg.eigvalsh(Gc.cpu())
    return lam.clamp_min(0).cpu().numpy()


def block_stats(Gc: torch.Tensor, n: int, d: int) -> dict:
    """TV, PR, Conc of one block from its centred (unnormalised) Gram Xc Xc^T."""
    lam = spectrum(Gc) / n                    # eigenvalues of Sigma = Xc^T Xc / N
    tv = float(lam.sum())
    pr = float(tv ** 2 / max(float((lam ** 2).sum()), 1e-300))
    desc = lam[::-1]
    return dict(n=n, d=d, tv=tv, pr=pr, conc=tv / pr, pr_cap=n - 1,
                top_eigs=[float(v) for v in desc[:TOP_EIGS]],
                pc1_share=float(desc[0] / tv) if tv > 0 else 0.0)


# ---------------------------------------------------------------- staging --
def plan_groups(dims, n: int, esize: int, budget_bytes: float):
    """Contiguous groups of (name, D) whose staged activations, n * D * esize bytes, fit the budget.
    A hook larger than the budget on its own forms a group by itself."""
    groups, cur, used = [], [], 0
    for name, d in dims:
        need = n * d * esize
        if cur and used + need > budget_bytes:
            groups.append(cur)
            cur, used = [], 0
        cur.append((name, d))
        used += need
    if cur:
        groups.append(cur)
    return groups


def _dims(bb, ctx, device):
    """[(name, D)] of every hook in execution order, from one forward of a single noise."""
    order, mods = [], dict(bb.hooks())

    def mk(name):
        def hook(_m, _i, out):
            order.append((name, int(bb.extract(name, out)[0].numel())))
        return hook

    hs = [m.register_forward_hook(mk(nm)) for nm, m in mods.items()]
    try:
        with torch.no_grad():
            bb.forward(torch.zeros(1, *bb.latent_shape, device=device, dtype=bb.dtype), ctx)
    finally:
        for h in hs:
            h.remove()
    seen, out = set(), []
    for nm, d in order:                        # a module called twice keeps its first call
        if nm not in seen:
            seen.add(nm)
            out.append((nm, d))
    missing = [nm for nm in mods if nm not in seen]
    assert not missing, f"hooks that never fired: {missing}"
    return out


@torch.no_grad()
def run_prompt(bb, prompt_idx: int, text: str, n: int, batch: int, device,
               ram_budget_gb: float = RAM_BUDGET_GB, ctx: dict = None, log=print) -> dict:
    """The per-block record of one prompt: {name: block_stats} for every hook of `bb`."""
    ctx = ctx if ctx is not None else bb.context(prompt_idx, text)
    dims = _dims(bb, ctx, device)
    esize = torch.empty((), dtype=bb.dtype).element_size()
    groups = plan_groups(dims, n, esize, ram_budget_gb * (1 << 30))
    mods = dict(bb.hooks())
    stats = {}
    for gi, group in enumerate(groups):
        t0 = time.time()
        X = {nm: torch.empty(n, d, dtype=bb.dtype) for nm, d in group}
        last = group[-1][0]
        cursor = {"i": 0}

        def mk(name):
            def hook(_m, _i, out):
                h = bb.extract(name, out)
                b = h.shape[0]
                X[name][cursor["i"]:cursor["i"] + b].copy_(h.reshape(b, -1).to(bb.dtype), non_blocking=False)
                if name == last:
                    raise _Stop
            return hook

        hs = [mods[nm].register_forward_hook(mk(nm)) for nm, _ in group]
        try:
            for i0 in range(0, n, batch):
                idx = range(i0, min(i0 + batch, n))
                x = torch.stack([probe_noise(prompt_idx, i, bb.latent_shape) for i in idx]).to(device, bb.dtype)
                cursor["i"] = i0
                try:
                    bb.forward(x, ctx)
                except _Stop:
                    pass
        finally:
            for h in hs:
                h.remove()
        for nm, d in group:
            Gc = centred_gram(X[nm], device)
            stats[nm] = block_stats(Gc, n, d)
            del Gc
            X[nm] = None
        del X
        torch.cuda.empty_cache()
        log(f"  group {gi + 1}/{len(groups)}: {len(group)} blocks ({group[0][0]} .. {last}) in {time.time() - t0:.0f} s")
    return {nm: stats[nm] for nm, _ in dims}


def record(backbone: str, arm: str, prompt_idx: int, text: str, n: int, dtype, blocks: dict, **extra) -> dict:
    return dict(backbone=backbone, arm=arm, prompt_idx=prompt_idx, prompt=text, n=n,
                seed_rule=SEED_RULE, dtype=str(dtype), **extra,
                blocks=[dict(name=k, **v) for k, v in blocks.items()])


def save_record(path: str, rec: dict):
    """Atomic JSON write (temp file + rename): resume trusts existing files."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(path + ".tmp", path)


def run_cli(bb, a, prompts: list[str], ids: list[int], out_dir: str, extra: dict = None):
    """Shared driver of the backbone probe CLIs: one JSON per prompt, resumable and shardable."""
    os.makedirs(out_dir, exist_ok=True)
    work = [i for i in ids if i % a.num_shards == a.shard
            and not os.path.exists(os.path.join(out_dir, f"p{i:03d}.json"))]
    print(f"[{bb.name}/{a.arm} probe] {len(work)} prompts x N={a.n} -> {out_dir}", flush=True)
    for i in work:
        t0 = time.time()
        blocks = run_prompt(bb, i, prompts[i], a.n, a.batch_size, a.device, a.ram_gb,
                            log=lambda s: print(s, flush=True))
        save_record(os.path.join(out_dir, f"p{i:03d}.json"),
                    record(bb.name, a.arm, i, prompts[i], a.n, bb.dtype, blocks, **(extra or {})))
        lb = bb.collapsed_block
        msg = f" | {lb}: TV {blocks[lb]['tv']:.4g} PR {blocks[lb]['pr']:.1f}" if lb in blocks else ""
        print(f"p{i:03d} done in {(time.time() - t0) / 60:.1f} min{msg}", flush=True)


def add_cli_args(ap, batch: int, n: int = N_DEFAULT):
    ap.add_argument("--arm", required=True, choices=("teacher", "student"))
    ap.add_argument("--prompt-ids", default=None, help="prompt indices or a file of them (default: the paper's set)")
    ap.add_argument("--n", type=int, default=n, help="noises per prompt")
    ap.add_argument("--batch-size", type=int, default=batch, help="noises per forward")
    ap.add_argument("--ram-gb", type=float, default=RAM_BUDGET_GB,
                    help="host-memory budget for staged activations; hooks are probed in groups that fit it")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
