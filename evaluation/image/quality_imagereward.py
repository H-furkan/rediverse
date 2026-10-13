"""ImageReward (Xu et al., NeurIPS 2023): BLIP-based human-preference reward (higher = better).

Package image-reward 1.5, weights ImageReward-v1.0 (downloaded by the package into
~/.cache/ImageReward); one image per call, fp32; prompts are truncated to 35 BLIP tokens by the
package. Needs transformers < 4.57 (the `image` environment pins 4.49).
"""
from __future__ import annotations

from tqdm import tqdm

from evaluation.image import config as C


class ImageRewardScore:
    def __init__(self, device: str = "cuda"):
        import ImageReward as RM
        self.model = RM.load(C.IMAGEREWARD_NAME, device=device)

    def compute(self, pairs: list[tuple[str, str]]) -> dict:
        scores: list[float] = []
        for path, prompt in tqdm(pairs, desc="imagereward"):
            scores.append(float(self.model.score(prompt, [path])))
        return {"mean": sum(scores) / len(scores), "per_image": scores}
