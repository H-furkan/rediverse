"""Pairwise LPIPS diversity (Zhang et al., CVPR 2018), VGG16 backbone (higher = more diverse).

Per prompt: the mean LPIPS distance over all R*(R-1)/2 image pairs; images are resized to
256x256 (antialiased) and scaled to [-1, 1]. Pairs are scored in chunks of `pair_batch`; the
chunk sums are divided by the total pair count, so the value does not depend on the chunking.
"""
from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from evaluation.common.image_sets import PromptGroup, open_rgb, pairwise_indices


class LpipsDiversity:
    def __init__(self, device: str = "cuda", net: str = "vgg", size: int = 256, pair_batch: int = 64):
        import lpips
        self.model = lpips.LPIPS(net=net, verbose=False).eval().to(device)
        self.device = device
        self.size = size
        self.pair_batch = pair_batch

    def _to_tensor(self, path: str) -> torch.Tensor:
        img = open_rgb(path)
        t = TF.to_tensor(TF.resize(img, [self.size, self.size], antialias=True))
        return (t * 2 - 1).unsqueeze(0)

    @torch.no_grad()
    def group_score(self, paths: list[str]) -> float:
        imgs = torch.cat([self._to_tensor(p) for p in paths]).to(self.device)
        idx = pairwise_indices(len(paths))
        ia = [i for i, _ in idx]
        ib = [j for _, j in idx]
        total, n = 0.0, len(idx)
        for k in range(0, n, self.pair_batch):
            a = imgs[ia[k:k + self.pair_batch]]
            b = imgs[ib[k:k + self.pair_batch]]
            total += self.model(a, b).sum().item()   # (chunk, 1, 1, 1)
        return total / n

    def compute(self, groups: list[PromptGroup]) -> dict:
        per_prompt = [self.group_score(g.paths) for g in tqdm(groups, desc="lpips")]
        return {"mean": sum(per_prompt) / len(per_prompt), "per_prompt": per_prompt}
