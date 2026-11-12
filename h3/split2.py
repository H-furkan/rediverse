"""Two-GPU block split of MiniMaxH3Transformer3DModel used by ReDiverse-FT training.

At 124 frames the packed sequence has 37,715 tokens, and the forward and backward of one block need
about 19 GiB of activations even with gradient checkpointing; with the 57 GB model resident this
exceeds an 80 GB GPU. Blocks [0, cut) stay on dev0 with the embedders, blocks [cut, keep) move to
dev1 and blocks >= keep are dropped (the loss reads block 45, and the trainer's hook on that block
stops the forward). forward_to_block is the model's own block loop with one transfer to dev1 at
the cut; autograd differentiates across devices, so the gradients of the LoRA parameters on dev0
are unchanged. The result equals the single-device forward up to bf16 non-associativity.
"""
# forward_to_block adapted from huggingface/diffusers MiniMaxH3Transformer3DModel.forward
# (Apache-2.0, Copyright 2025 The MiniMax Team and The HuggingFace Team).
from __future__ import annotations

import torch
import torch.nn as nn

MODALITY_NUM = 3


def split_model(model, dev0: str, dev1: str, cut: int, keep: int):
    """Move blocks [cut, keep) to dev1 and drop blocks >= keep; returns the number dropped."""
    blocks = list(model.transformer_blocks)
    assert 0 < cut < keep <= len(blocks)
    for b in blocks[cut:keep]:
        b.to(dev1)
    model.transformer_blocks = nn.ModuleList(blocks[:keep])
    dropped = len(blocks) - keep
    # everything else (embedders, rope, time embedder, heads) stays on dev0
    model._split = dict(dev0=dev0, dev1=dev1, cut=cut, keep=keep)
    torch.cuda.empty_cache()
    return dropped


def forward_to_block(model, *, hidden_states, audio_hidden_states, encoder_hidden_states,
                     timestep, timestep_indices, token_tags, position_ids,
                     video_indices, audio_indices, text_indices):
    """The model's own forward up to and including the last kept block, with the device hop.
    Returns the packed hidden state after the last block; the trainer's hook on block 45 normally
    captures that block's output and stops the forward before this return."""
    from diffusers.models.modeling_utils import get_parameter_dtype
    sp = model._split
    dev0, dev1, cut = sp["dev0"], sp["dev1"], sp["cut"]

    rotary_emb = model.rope(position_ids)
    sequence_length = position_ids.shape[0]
    video_embeds = model.proj_in(hidden_states.to(get_parameter_dtype(model.proj_in)))
    audio_embeds = model.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(model.audio_proj_in)))
    text_embeds = model.context_embedder(encoder_hidden_states.to(get_parameter_dtype(model.context_embedder)))
    text_embeds = model.token_refiner(text_embeds)
    h = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
    h = h.index_copy(1, text_indices, text_embeds)
    h = h.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
    h = h.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
    temb = model.time_proj(timestep)
    temb = model.time_embedder(temb.to(get_parameter_dtype(model.time_embedder)))
    adaln = timestep_indices * MODALITY_NUM + token_tags

    ckpt = torch.is_grad_enabled() and model.gradient_checkpointing
    rot1 = None
    for i, block in enumerate(model.transformer_blocks):
        if i == cut:
            h = h.to(dev1)
            temb1 = temb.to(dev1); adaln1 = adaln.to(dev1)
            rot1 = tuple(r.to(dev1) for r in rotary_emb)
        args = (h, temb, adaln, rotary_emb) if i < cut else (h, temb1, adaln1, rot1)
        if ckpt:
            h = model._gradient_checkpointing_func(block, *args)
        else:
            h = block(*args)
    return h
