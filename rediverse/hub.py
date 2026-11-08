"""Released ReDiverse-FT adapters: where to find them and how to read them.

The SDXL, SD3 and Wan 2.1 adapters are single safetensors files on the Hugging Face Hub,
<REDIVERSE_HF_REPO>/<backbone>/rediverse_ft.safetensors; the MiniMax-H3 adapter has the same
format and is read from a local path. Each file holds the generator LoRA in float32 under
keys "<module path inside the denoiser>.lora_A.weight" / ".lora_B.weight", with string
metadata (rank, alpha, denoiser, backbone, student, teacher, anchor_block,
active_student_steps, ...).
"""
from __future__ import annotations

import os

REDIVERSE_HF_REPO = "ReDiverse/ReDiverse"
ADAPTER_FILE = "rediverse_ft.safetensors"
# package name -> Hub subfolder
BACKBONES = {"sdxl": "sdxl", "sd3": "sd3", "wan": "wan2.1", "h3": "minimax-h3"}


def hub_backbone(name: str) -> str:
    """Hub subfolder for a package name ("wan" -> "wan2.1") or an existing subfolder id."""
    if name in BACKBONES:
        return BACKBONES[name]
    if name in BACKBONES.values():
        return name
    raise ValueError(f"unknown backbone {name!r}; expected one of {sorted(BACKBONES)} or {sorted(BACKBONES.values())}")


def adapter_filename(backbone: str) -> str:
    """'<backbone>/rediverse_ft.safetensors' inside the Hub repository."""
    return f"{hub_backbone(backbone)}/{ADAPTER_FILE}"


def resolve_adapter(path_or_default, backbone: str, repo_id: str = None, revision: str = None) -> str:
    """Local path of the ReDiverse-FT adapter of `backbone`.

    `path_or_default` may be a .safetensors file, a directory laid out like the Hub repository
    (<dir>/<backbone>/rediverse_ft.safetensors) or holding rediverse_ft.safetensors directly,
    or None / "" / "hf" to download from the Hub (repo `repo_id`, default REDIVERSE_HF_REPO or
    the REDIVERSE_HF_REPO environment variable) into the user's Hugging Face cache."""
    sub = hub_backbone(backbone)
    if path_or_default and path_or_default != "hf":
        p = os.path.expanduser(str(path_or_default))
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            for cand in (os.path.join(p, sub, ADAPTER_FILE), os.path.join(p, ADAPTER_FILE)):
                if os.path.isfile(cand):
                    return cand
        raise FileNotFoundError(f"no ReDiverse-FT adapter for {sub} at {p}")
    from huggingface_hub import hf_hub_download
    repo = repo_id or os.environ.get("REDIVERSE_HF_REPO", REDIVERSE_HF_REPO)
    return hf_hub_download(repo, filename=f"{sub}/{ADAPTER_FILE}", revision=revision)


def load_lora_safetensors(path: str):
    """(state_dict, metadata): every tensor of the file on CPU, dtypes as stored, and the
    string metadata dict (empty if the file has none)."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        sd = {k: f.get_tensor(k) for k in f.keys()}
    return sd, meta


def save_lora_safetensors(path: str, state_dict: dict, metadata: dict):
    """Write a released adapter file; metadata values are stored as strings and tensors keep
    their dtype (contiguous copies where needed)."""
    from safetensors.torch import save_file
    tensors = {k: v.detach().contiguous() for k, v in state_dict.items()}
    save_file(tensors, path, metadata={str(k): str(v) for k, v in metadata.items()})
