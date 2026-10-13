"""PickScore (Kirstain et al., NeurIPS 2023), optional: computed on request, reported in no table.

CLIP ViT-H/14 fine-tuned on Pick-a-Pic preferences; score = logit_scale.exp() * cos(text, image),
fp16 model, batch 32.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from evaluation.common.image_sets import open_rgb
from evaluation.image import config as C


class PickScore:
    def __init__(self, device: str = "cuda", batch_size: int = 32):
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(C.PICKSCORE_PROCESSOR, revision=C.PICKSCORE_PROCESSOR_REV)
        self.model = AutoModel.from_pretrained(C.PICKSCORE_ID, revision=C.PICKSCORE_REV, torch_dtype=torch.float16)
        self.model.eval().to(device)
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def compute(self, pairs: list[tuple[str, str]]) -> dict:
        scores: list[float] = []
        logit_scale = self.model.logit_scale.exp().float()
        for k in tqdm(range(0, len(pairs), self.batch_size), desc="pickscore"):
            chunk = pairs[k:k + self.batch_size]
            img_in = self.processor(images=[open_rgb(p) for p, _ in chunk], return_tensors="pt").to(self.device)
            txt_in = self.processor(text=[t for _, t in chunk], padding=True, truncation=True, max_length=77,
                                    return_tensors="pt").to(self.device)
            ie = self.model.get_image_features(**img_in).float()
            te = self.model.get_text_features(**txt_in).float()
            ie = torch.nn.functional.normalize(ie, dim=-1)
            te = torch.nn.functional.normalize(te, dim=-1)
            scores.extend((logit_scale * (ie * te).sum(-1)).tolist())
        return {"mean": sum(scores) / len(scores), "per_image": scores}
