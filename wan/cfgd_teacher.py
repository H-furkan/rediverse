"""The ReDiverse-FT target teacher: Wan2.1-T2V-14B base + Perflow-Shuai's CFG5->CFG1 LoRA.

The LoRA (rank 128, alpha 128, scale 1.0, Apache-2.0) distils the base teacher's classifier-free
guidance at scale 5 into a single conditional forward. Sampled at its own contract (UniPC,
50 steps, flow shift 5.0, guidance 1), its trajectory reaches the student's post-step-1 sigma
(0.9375) at forward 13 (sigma_out 0.9344), so the ReDiverse-FT target is its blocks.28 output at
forward 13 (config.FT_TARGET_FORWARD).

Key renaming (PEFT names of the release -> diffusers WanTransformer3DModel):
  blocks.N.self_attn.{q,k,v,o}  -> blocks.N.attn1.to_{q,k,v,out.0}
  blocks.N.cross_attn.{q,k,v,o} -> blocks.N.attn2.to_{q,k,v,out.0}
  blocks.N.ffn.{0,2}            -> blocks.N.ffn.net.{0.proj,2}
400 modules (40 blocks x 10 projections), 800 tensors.
"""
from __future__ import annotations

import re

import torch

from wan import config as C

ADAPTER_NAME = "cfgd"
RANK = ALPHA = 128
N_MODULES = 400


def adapter_file() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C.CFGD_REPO, C.CFGD_FILE, revision=C.CFGD_REVISION)


def rename(k: str) -> str:
    k = re.sub(r"^base_model\.model\.", "", k)
    k = (k.replace(".self_attn.q.", ".attn1.to_q.").replace(".self_attn.k.", ".attn1.to_k.")
          .replace(".self_attn.v.", ".attn1.to_v.").replace(".self_attn.o.", ".attn1.to_out.0."))
    k = (k.replace(".cross_attn.q.", ".attn2.to_q.").replace(".cross_attn.k.", ".attn2.to_k.")
          .replace(".cross_attn.v.", ".attn2.to_v.").replace(".cross_attn.o.", ".attn2.to_out.0."))
    k = k.replace(".ffn.0.", ".ffn.net.0.proj.").replace(".ffn.2.", ".ffn.net.2.")
    return (k.replace(".lora_A.weight", f".lora_A.{ADAPTER_NAME}.weight")
             .replace(".lora_B.weight", f".lora_B.{ADAPTER_NAME}.weight"))


def attach(model, dtype=torch.bfloat16):
    """Attach the CFG-distillation LoRA (scale 1.0, weights cast to `dtype`) to a loaded Wan
    transformer, in place, and enable it."""
    from peft import LoraConfig
    from safetensors.torch import load_file

    sd = {rename(k): v for k, v in load_file(adapter_file()).items()}
    targets = sorted({re.sub(rf"\.lora_[AB]\.{ADAPTER_NAME}\.weight$", "", k) for k in sd})
    assert len(targets) == N_MODULES, len(targets)
    model.add_adapter(LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=targets), adapter_name=ADAPTER_NAME)
    missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=False)
    assert not unexpected, list(unexpected)[:3]
    loaded = [k for k in sd if k not in missing]
    assert len(loaded) == len(sd), f"{len(loaded)}/{len(sd)} adapter tensors matched"
    model.enable_adapters()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[wan] CFG-distilled teacher: LoRA on {len(targets)} modules, r={RANK} alpha={ALPHA}; "
          f"sampled at flow shift {C.CFGD_FLOW_SHIFT}, guidance 1", flush=True)
    return model
