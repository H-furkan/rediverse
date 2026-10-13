"""Stage A of the image evaluation: per-prompt feature caches, computed once per sample set on
CPU in fp32, so every feature-space metric (evaluation.image.feature_metrics) is a numpy pass.

Per prompt (R seeds) the cache holds
  feats/dino/p<idx>.npz      tokens (R, 257, 768) fp16  DINOv2-base last_hidden_state (CLS at 0, 256 patches)
  feats/clip/p<idx>.npz      img (R, 512) fp32, txt (512,) fp32  CLIP ViT-B/32 image and prompt features
  feats/dreamsim/p<idx>.npy  (R, E) fp32                         DreamSim ensemble embeddings

  python -m evaluation.image.extract_feats --images-dir <set>/images --out <set>/feats --shard 0 --num-shards 4 --threads 32

Run several shards in parallel (one process each). A prompt whose cache files exist and are newer
than all of its images is skipped, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from evaluation.common.atomic import save
from evaluation.common.image_sets import SEEDS, load_groups, open_rgb, shard, up_to_date
from evaluation.image import config as C
from rediverse.data import GENEVAL_553, parse_ids

EXT = {"dino": ".npz", "clip": ".npz", "dreamsim": ".npy"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--prompts", default=GENEVAL_553)
    ap.add_argument("--prompt-ids", default=None, help="comma list / a-b ranges / file (default: every prompt)")
    ap.add_argument("--seeds", type=int, default=SEEDS, help="samples per prompt the set must hold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker takes the prompts whose list index modulo --num-shards equals --shard")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--which", default="dino,clip,dreamsim")
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        ap.error(f"--shard must be in 0..{a.num_shards - 1}")
    torch.set_num_threads(a.threads)
    which = a.which.split(",")
    ids = parse_ids(a.prompt_ids) if a.prompt_ids else None
    groups = shard(load_groups(a.images_dir, a.prompts, ids, a.seeds), a.shard, a.num_shards)
    for d in which:
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    device, dtype = "cpu", torch.float32
    models = {}
    if "dino" in which:
        from transformers import AutoImageProcessor, AutoModel
        models["dino"] = (AutoImageProcessor.from_pretrained(C.DINO_ID, revision=C.DINO_REV),
                          AutoModel.from_pretrained(C.DINO_ID, revision=C.DINO_REV, torch_dtype=dtype).eval().to(device))
    if "clip" in which:
        from transformers import CLIPModel, CLIPProcessor
        models["clip"] = (CLIPProcessor.from_pretrained(C.CLIP_ID, revision=C.CLIP_REV),
                          CLIPModel.from_pretrained(C.CLIP_ID, revision=C.CLIP_REV, torch_dtype=dtype).eval().to(device))
    if "dreamsim" in which:
        from dreamsim import dreamsim
        cache = C.dreamsim_cache(); os.makedirs(cache, exist_ok=True)
        m, pre = dreamsim(pretrained=True, device=device, cache_dir=cache, dreamsim_type=C.DREAMSIM_TYPE)
        models["dreamsim"] = (pre, m.eval())

    t0 = time.time(); n_img = 0
    for gi, g in enumerate(groups):
        tag = f"p{g.prompt_idx:03d}"
        pending = [d for d in which if not up_to_date(os.path.join(a.out, d, tag + EXT[d]), g.paths)]
        if not pending:
            continue
        imgs = [open_rgb(p) for p in g.paths]
        with torch.no_grad():
            if "dino" in pending:
                proc, m = models["dino"]; outs = []
                for k in range(0, len(imgs), a.batch):
                    x = proc(images=imgs[k:k + a.batch], return_tensors="pt").to(device)
                    outs.append(m(**{kk: vv.to(dtype) if vv.is_floating_point() else vv for kk, vv in x.items()}).last_hidden_state.float().cpu())
                tok = torch.cat(outs).to(torch.float16).numpy()
                save(os.path.join(a.out, "dino", tag + ".npz"), lambda f: np.savez(f, tokens=tok, seeds=np.arange(len(imgs))))
            if "clip" in pending:
                proc, m = models["clip"]; ie = []
                for k in range(0, len(imgs), a.batch):
                    x = proc(images=imgs[k:k + a.batch], return_tensors="pt").to(device)
                    ie.append(m.get_image_features(pixel_values=x["pixel_values"].to(dtype)).float().cpu())
                tx = proc(text=[g.prompt], return_tensors="pt", padding=True, truncation=True).to(device)
                te = m.get_text_features(**tx).float().cpu()[0]
                save(os.path.join(a.out, "clip", tag + ".npz"),
                     lambda f: np.savez(f, img=torch.cat(ie).numpy(), txt=te.numpy(), prompt=g.prompt))
            if "dreamsim" in pending:
                pre, m = models["dreamsim"]; es = []
                for k in range(0, len(imgs), a.batch):
                    x = torch.cat([pre(im) for im in imgs[k:k + a.batch]]).to(device)
                    es.append(m.embed(x).float().cpu())
                save(os.path.join(a.out, "dreamsim", tag + ".npy"), lambda f: np.save(f, torch.cat(es).numpy()))
        n_img += len(imgs)
        if gi % 10 == 0:
            el = time.time() - t0
            print(f"[feats {a.shard}/{a.num_shards}] {gi+1}/{len(groups)} prompts  {n_img/el:.1f} img/s  "
                  f"eta {(len(groups)-gi-1)*el/(gi+1)/60:.0f} min", flush=True)
    print(f"[feats {a.shard}/{a.num_shards}] done {len(groups)} prompts in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
