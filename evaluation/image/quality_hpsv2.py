"""HPSv2.1, Human Preference Score v2.1 (Wu et al.; higher = better).

OpenCLIP ViT-H-14 built without pretrained weights (open_clip logs that the model is initialised
randomly) and then loaded with the HPS_v2.1 state dict from xswu/HPSv2. Score per image =
cos(image embedding, prompt embedding), fp16 model, batch 32.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from evaluation.common.image_sets import open_rgb
from evaluation.image import config as C


class HpsV2Score:
    def __init__(self, device: str = "cuda", version: str = "v2.1", batch_size: int = 32):
        import open_clip
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(C.HPS_REPO, C.HPS_FILES[version], revision=C.HPS_REV)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="fp16", device=device)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state["state_dict"] if "state_dict" in state else state
        self.model.load_state_dict(sd)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def compute(self, pairs: list[tuple[str, str]]) -> dict:
        scores: list[float] = []
        for k in tqdm(range(0, len(pairs), self.batch_size), desc="hpsv2"):
            chunk = pairs[k:k + self.batch_size]
            imgs = torch.stack([self.preprocess(open_rgb(p)) for p, _ in chunk])
            imgs = imgs.to(self.device, torch.float16)
            toks = self.tokenizer([t for _, t in chunk]).to(self.device)
            ie = self.model.encode_image(imgs, normalize=True).float()
            te = self.model.encode_text(toks, normalize=True).float()
            scores.extend((ie * te).sum(-1).tolist())
        return {"mean": sum(scores) / len(scores), "per_image": scores}
