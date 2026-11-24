#!/usr/bin/env python
"""Probe the variance characteristic of every Qwen-Image transformer block (paper Sec. 3.1, App. A).

The teacher or a student gets N noises of one prompt as the input of its first denoising step: one conditional
forward at sigma 1 (every schedule here starts at sigma 1; model timestep 1.0). The probed tensor of each block is
its image rows ([4096 tokens x d_model]).

  python -m qwenimage.probe --family q21 --arm teacher --n 200
  python -m qwenimage.probe --student lightning --arm student --n 200
  python -m rediverse.probe_report --teacher <out>/probe_n200/teacher --student <out>/probe_n200/lightning --collapsed block20

Prompts: prompts/probe_video20.txt (the reference 20-prompt probe set, GenEval ids). Noises: the shared CPU rule of
rediverse.probe in the packed latent layout [4096, 64].
"""
from __future__ import annotations

import argparse
import hashlib
import os

import torch

from qwenimage import config as C
from qwenimage import model as M
from rediverse import probe as P
from rediverse.data import GENEVAL_553, PROMPTS_DIR, parse_ids, read_prompts

PROBE20 = PROMPTS_DIR + "/probe_video20.txt"
BATCH = 8


class Backbone:
    dtype = C.DTYPE

    def __init__(self, who: str, device, prompts: dict[int, str], collapsed: int = -1, blocks=None, adapter=None):
        self.who, self.device = who, device
        self.fam = M.who_family(who)
        f = C.family(self.fam)
        self.name = f["backbone"]
        self.latent_shape = f["latent_shape"]
        self.blocks = list(range(f["n_blocks"])) if blocks is None else sorted(blocks)
        self.collapsed_block = f"block{collapsed:02d}" if collapsed >= 0 else None
        ids = sorted(prompts)
        self._ctx = dict(zip(ids, M.encode_many(self.fam, [prompts[i] for i in ids], device)))
        self.model = M.load_transformer(who, device)
        if adapter:                                  # App. A "after the fine-tuning": FT adapter on in step 1
            M.load_lora(self.fam, self.model, adapter)
            M.set_active(self.model, M.base_adapters(self.model) + [M.FT_ADAPTER])
        self.t1 = None

    def hooks(self):
        return [(f"block{i:02d}", self.model.transformer_blocks[i]) for i in self.blocks]

    def extract(self, name, out):
        return M.extract(self.fam, name, out)

    def context(self, prompt_idx: int, text: str):
        return self._ctx[prompt_idx].to(self.device)

    def forward(self, x, ctx):
        """The model's first sampling forward: pure noise at its schedule's first sigma. Every model starts at sigma 1
        except smd20, whose released grid starts at 0.999 (config 'sigma0'); its model timestep bf16(999)/1000 rounds
        to 1.0 exactly as in its own pipeline, so the probed forward is the same one its sampler runs."""
        sg0 = M.schedule(self.who)["sigmas"][0]
        want = 1.0 if self.who.startswith("teacher") else C.student(self.who).get("sigma0", 1.0)
        assert abs(sg0 - want) < 1e-6, (self.who, sg0, want)
        t = M.model_timestep(self.who, 0, x.shape[0], self.device)
        assert float(t[0]) == 1.0, float(t[0])
        M.forward(self.fam, self.model, x, ctx, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_cli_args(ap, batch=BATCH, n=200)
    ap.add_argument("--family", default=None, help="teacher arm: q21 | q20")
    ap.add_argument("--student", default=None)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--adapter", default=None, help="student only: ReDiverse-FT adapter active in the probed forward")
    ap.add_argument("--collapsed", type=int, default=-1)
    a = ap.parse_args()
    if a.arm == "teacher":
        fam = a.family or C.family_of(a.student)
        who, tag = f"teacher:{fam}", "teacher"
    else:
        fam, who = C.family_of(a.student), a.student
        tag = a.student + ("_ft" if a.adapter else "")
    texts = read_prompts(GENEVAL_553)
    ids = parse_ids(a.prompt_ids) if a.prompt_ids else parse_ids(PROBE20)
    out_dir = a.out_dir or C.output_dir(fam, f"probe_n{a.n}", tag)
    mine = [i for i in ids if i % a.num_shards == a.shard and not os.path.exists(os.path.join(out_dir, f"p{i:03d}.json"))]
    if not mine:
        print(f"[qwen/{who} probe] nothing to do in {out_dir}")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    bb = Backbone(who, a.device, {i: texts[i] for i in mine}, collapsed=a.collapsed,
                  blocks=parse_ids(a.blocks) if a.blocks else None, adapter=a.adapter)
    here = os.path.dirname(os.path.abspath(__file__))
    code = {f: hashlib.sha256(open(os.path.join(here, f), "rb").read()).hexdigest()[:16] for f in ("model.py", "probe.py")}
    P.run_cli(bb, a, texts, ids, out_dir, extra=dict(sigma=1.0, who=who, adapter=a.adapter, code_sha256=code,
                                                     student_lora=bb.model._qwen_info.get("student_lora")))


if __name__ == "__main__":
    main()
