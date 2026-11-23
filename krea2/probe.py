#!/usr/bin/env python
"""Probe the variance characteristic of every Krea 2 transformer block (paper Sec. 3.1, Fig. 2, Tab. 1).

Krea-2-Raw (teacher) or Krea-2-Turbo (student) gets N noises of one prompt as the input of its first denoising
step: one conditional forward at sigma 1 (both schedules start at sigma 1, whatever their shift). The probed tensor
of each of the 28 transformer_blocks is its image rows ([4096 tokens x 6144]).

  python -m krea2.probe --arm teacher --prompt-ids 0-19 --n 200 --device cuda:0
  python -m krea2.probe --arm student --prompt-ids 0-19 --n 200 --device cuda:0
  python -m rediverse.probe_report --teacher <out>/teacher --student <out>/student --collapsed block26

Default prompts: prompts/probe_video20.txt (the reference 20-prompt probe set: GenEval ids over all six categories).
Noises are the shared CPU rule of rediverse.probe (seed 1_000_000 + prompt_idx * 100003 + i) drawn directly in the
packed latent layout [4096, 64] (i.i.d. Gaussian either way).
"""
from __future__ import annotations

import argparse

import torch

from krea2 import config as C
from krea2 import model as M
from rediverse import probe as P
from rediverse.data import GENEVAL_553, PROMPTS_DIR, parse_ids, read_prompts

PROBE20 = PROMPTS_DIR + "/probe_video20.txt"   # the reference 20-prompt probe set (wan/h3 probes), GenEval ids

BATCH = 8


class Backbone:
    name = C.BACKBONE
    dtype = C.DTYPE
    latent_shape = C.LATENT_SHAPE

    def __init__(self, arm: str, device, prompts: dict[int, str], collapsed: int = C.ANCHOR_BLOCK, blocks=None,
                 adapter: str = None):
        self.device = device
        self.blocks = list(range(C.N_BLOCKS)) if blocks is None else sorted(blocks)
        self.collapsed_block = f"block{collapsed:02d}" if collapsed >= 0 else None
        ids = sorted(prompts)
        ctxs = M.encode_many([prompts[i] for i in ids], device)   # text encoder freed before the transformer loads
        self._ctx = dict(zip(ids, ctxs))
        self.model = M.load_transformer(arm, device)
        if adapter:                      # paper App. A 'after the fine-tuning': student step 1 with the FT adapter on
            assert arm == "student", "--adapter applies to the student"
            self.lora = M.load_lora(self.model, adapter)
            self.model.enable_adapters()

    def hooks(self):
        return [(f"block{i:02d}", self.model.transformer_blocks[i]) for i in self.blocks]

    @staticmethod
    def extract(name, out):
        return M.extract(name, out)

    def context(self, prompt_idx: int, text: str):
        return self._ctx[prompt_idx].to(self.device)

    def forward(self, x, ctx):
        M.forward(self.model, x, ctx, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_cli_args(ap, batch=BATCH, n=200)
    ap.add_argument("--blocks", default=None, help="probe only these blocks, e.g. 20-27 (default: all 28)")
    ap.add_argument("--adapter", default=None, help="student only: ReDiverse-FT adapter active in the probed forward")
    a = ap.parse_args()
    texts = read_prompts(GENEVAL_553)
    ids = parse_ids(a.prompt_ids) if a.prompt_ids else parse_ids(PROBE20)   # all six GenEval categories
    import os
    out_dir = a.out_dir or C.output_dir(f"probe_n{a.n}", a.arm)
    mine = [i for i in ids if i % a.num_shards == a.shard and not os.path.exists(os.path.join(out_dir, f"p{i:03d}.json"))]
    if not mine:
        print(f"[krea2/{a.arm} probe] nothing to do in {out_dir}")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    bb = Backbone(a.arm, a.device, {i: texts[i] for i in mine},
                  blocks=parse_ids(a.blocks) if a.blocks else None, adapter=a.adapter)
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    code = {f: hashlib.sha256(open(os.path.join(here, f), "rb").read()).hexdigest()[:16] for f in ("model.py", "probe.py")}
    P.run_cli(bb, a, texts, ids, out_dir, extra=dict(
        sigma=1.0, revision=(C.TEACHER_REVISION if a.arm == "teacher" else C.STUDENT_REVISION), adapter=a.adapter,
        norm_dtype=str(bb.model.transformer_blocks[0].norm1.weight.dtype), code_sha256=code))


if __name__ == "__main__":
    main()
