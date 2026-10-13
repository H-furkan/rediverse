"""ReDiverse-TF: per-prompt teacher principal directions and their injection (paper Sec. 4.2).

For one prompt the teacher is run from N independent latents, the activation of the
collapsed block at the backbone's direction step k (rediverse.data.TF_DIRECTION_STEP; the
ReDiverse-FT tau on SDXL, SD3 and MiniMax-H3, step 8 of the base teacher on Wan 2.1) is
flattened to a D-vector per latent, and the top-K principal directions are taken with the
Gram trick (D is huge, N = 25):

    X   = sample-centred activations                [N, D]
    G   = X X^T,  eigenpairs (lambda_k, v_k)         [N, N]
    u_k = X^T v_k / ||X^T v_k||                      unit direction in activation space
    s_k = std(sqrt(lambda_k) * v_k)                  the teacher's spread along u_k

At sampling time the student's block activation at step 1 becomes
h + sum_k alpha_{i,k} s_k u_k with alpha_{i,k} ~ U[-2, 2] (rediverse.seeds.tf_alphas).

A direction file (one per prompt; p{idx:03d}.pt for GenEval, v{idx:03d}.pt for VBench) is a
torch.save'd dict with U [K, D] bfloat16, s [K] float32, eigvals [N] float32, pc1_share,
the estimation metadata (prompt_idx, prompt, k = direction step, n, seeds, shape) and the
backbone's site keys (site / block / stream / frames). matches() checks which prompt a file
belongs to without reading its tensors.
"""
from __future__ import annotations

import os

import torch

from rediverse.seeds import TF_BAND, TF_N_DIR, alpha_record, tf_alphas, tf_alphas_batch  # noqa: F401

N_DIR = 8            # top-K pairs stored per prompt


def directions_from_samples(X: torch.Tensor, n_dir: int = N_DIR):
    """X [N, D] bf16 on CPU -> (U bf16 [n_dir, D], s [n_dir], eigvals [N] descending).

    Paper eq. (basis): u_k = X^T v_k / ||.||, s_k = std(sqrt(lambda_k) v_k), on
    sample-centred X with the raw (un-normalised) Gram X X^T; s_k uses the biased (/N) std."""
    N = X.shape[0]
    mean = X.float().mean(0, keepdim=True)              # [1, D] fp32
    Xc = torch.cat([(X[i:i + 1].float() - mean) for i in range(N)], 0)   # [N, D] fp32
    G = (Xc @ Xc.T).double()                            # raw Gram
    evals, evecs = torch.linalg.eigh(G)                 # ascending
    order = torch.arange(N - 1, -1, -1)
    evals, evecs = evals[order], evecs[:, order]        # descending
    U, S = [], []
    for k in range(n_dir):
        lam, v = float(evals[k]), evecs[:, k].float()
        u = v @ Xc
        U.append((u / u.norm()).to(torch.bfloat16))
        # biased std (/N); torch defaults to /(N-1)
        S.append(float((lam ** 0.5 * v).std(unbiased=False)))   # s_k = std(sqrt(lam) v_k)
    return torch.stack(U), torch.tensor(S), evals.float()


def tf_offset(alpha: torch.Tensor, s: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """sum_k alpha_k s_k u_k, computed in U's dtype: alpha [K] -> [D], or [B, K] -> [B, D].
    The caller reshapes it to the hooked activation and adds it in the activation's dtype."""
    return (alpha * s).to(U.dtype) @ U.reshape(U.shape[0], -1)


def direction_file(bank_dir: str, prompt_idx: int, prefix: str = "p") -> str:
    """<bank_dir>/<prefix><idx:03d>.pt; prefix "p" for GenEval banks, "v" for VBench banks."""
    return os.path.join(bank_dir, f"{prefix}{prompt_idx:03d}.pt")


def save_directions(path: str, U: torch.Tensor, s: torch.Tensor, eigvals: torch.Tensor, **meta) -> dict:
    """Write one direction file atomically (temp file + rename). `meta` carries prompt_idx,
    prompt, k, n, seeds, shape and the site keys; pc1_share = eigvals[0] / sum(eigvals)."""
    rec = dict(meta)
    rec.update(U=U, s=s, eigvals=eigvals, pc1_share=float(eigvals[0] / eigvals.sum()))
    torch.save(rec, path + ".tmp")
    os.replace(path + ".tmp", path)
    return rec


def load_direction_file(path: str) -> dict:
    """The full record of one direction file, on CPU."""
    return torch.load(path, map_location="cpu")


def matches(path: str, prompt: str, prompt_idx: int = None) -> bool:
    """True when the direction file `path` exists and was estimated for `prompt` and, when
    given, for prompt index `prompt_idx`, which seeds the estimation latents. The file is
    memory-mapped, so only its metadata is read, even for a multi-GB video file."""
    if not os.path.isfile(path):
        return False
    d = torch.load(path, map_location="cpu", mmap=True)
    return d.get("prompt") == prompt and (prompt_idx is None or d.get("prompt_idx") == prompt_idx)


def load_directions(path: str, n_dir: int = N_DIR, device="cpu", prompt: str = None):
    """(U [K, D] in the stored bfloat16, s [K] float32) on `device`, K = min(n_dir, stored).
    With `prompt`, the file must hold that prompt's directions (ValueError otherwise)."""
    d = torch.load(path, map_location="cpu")
    if prompt is not None and d.get("prompt") != prompt:
        raise ValueError(f"{path} holds the directions of {d.get('prompt')!r}, not of {prompt!r}; "
                         "pass --directions-dir to use a separate direction bank for a different prompt file")
    k = min(n_dir, d["U"].shape[0])
    return d["U"][:k].to(device), d["s"][:k].to(device, torch.float32)
