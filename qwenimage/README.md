# ReDiverse on Qwen-Image

Ports of ReDiverse (probe -> collapsed block l -> ReDiverse-FT LoRA + ReDiverse-TF directions) to the Qwen-Image
families, with the most-downloaded distilled students of each (Hugging Face downloads, 2026-11-23):

| family | teacher | student (key) | kind | steps | schedule (card) |
|---|---|---|---|---|---|
| q21 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), 40 steps, no CFG | [lightx2v/Qwen-Image-2.1-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning) (`lightning`) | LoRA | 4 | raw [1, .75, .5, .25], dynamic shift |
| | | [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) v0.3 r256 (`viggle`) | LoRA | 6 | raw [1, .9375, .875, .75, .5, .25], dynamic shift |
| | | [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (`turbo`) | full | 8 | checkpoint `sample_sigmas`, static (card: 2048^2) |
| | | [PrunaAI/P-Q-Image-2.1](https://huggingface.co/PrunaAI/P-Q-Image-2.1) 6-step v0.2 (`pq`) | LoRA | 6 | raw [1, .925, .85, .625, .375, .125], dynamic shift |
| | | [lightx2v/Qwen-Image-2.1-Lightning-v2-RL](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning-v2-RL) full transformer (`v2rl_full`; `v2rl_lora` = its r64 SVD reconstruction) | full | 4 | fixed [1, .954, .895, .705], static |
| q20 | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), 50 steps, true CFG 4 | [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) 8-step V2.0 (`lightning20`) | LoRA | 8 | shift 3 |
| q2512 | [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512), 50 steps, true CFG 4 (card negative prompt) | [lightx2v/Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) 8-step V1.0 (`lightning2512`) | LoRA | 8 | shift 3 |

All at 1024^2 (on-card for the LoRA students; off-card for Qwen-Image-2.1-Turbo, whose card samples 2048^2, and for the
Qwen-Image / 2512 teachers, whose cards use 1328^2 plus a prompt suffix). Model ids, revisions, schedules and the
second-round students (linoyts DMD2, Wuli, congruency MFD, Qwen-Image-Flash, SMD, Wuli 2512) live in `qwenimage/config.py`;
the per-student ReDiverse sites (collapsed block l, teacher forward tau) in `config.SITES`.

Environment: `diffusers` >= 0.43 (`QwenImage21Pipeline` needs >= 0.41), `transformers` >= 5.17, `peft`; the image
metrics use the shared `evaluation/` environments. Weights come from the Hugging Face cache (`HF_HUB_CACHE`); data go to
`<REDIVERSE_DATA>/<backbone>/`, outputs to `<REDIVERSE_OUT>/<backbone>/` (backbone `qwen21`, `qwen20` or `qwen2512`).

## Stages

```bash
python -m qwenimage.check --family q21 --pipeline            # schedules, LoRA loading, loop == pipeline, timing
python -m qwenimage.generate --family q21 --arm teacher --prompt-ids 0,11,17,27,28,34 --num-samples 8 --tag _preview
python -m qwenimage.generate --student lightning --arm student --prompt-ids 0,11,17,27,28,34 --num-samples 8 --tag _preview
python -m qwenimage.probe --family q21 --arm teacher --n 200 ; python -m qwenimage.probe --student lightning --arm student --n 200
python -m rediverse.probe_report --teacher <out>/qwen21/probe_n200/teacher --student <out>/qwen21/probe_n200/lightning
python -m qwenimage.build_targets --student lightning --tau <tau> --blocks <l>
torchrun --nproc_per_node 4 -m qwenimage.train --student lightning --block <l> --tau <tau> --run lightning_b<l>
python -m qwenimage.estimate_directions --student lightning --tau <tau> --blocks <l>
python -m qwenimage.generate --student lightning --arm rediverse --adapter <ft> --block <l> --tau <tau> --band 2
```

`scripts/qwen.sh <student> <stage> ...` runs the stages end to end on one node (`preview`, `probe`, `bank`, `train`,
`directions`, `sweep`, `generate`). Tables: `qwenimage/table.py` (diversity and quality), `qwenimage/image_stats.py`
(colour / exposure statistics). `qwenimage/hub/` holds the Hugging Face release files of Qwen-Image-Lightning ReDiverse
(model card, `config.json`, `NOTICE`, standalone `generate_qwen_image_lightning.py`); `convert_release.py` writes its
LoRA file.

Conventions shared with the other ports: eval latent of GenEval prompt i, sample s = randn([4096, 64], seed
i * 1000 + s) in every arm; TF alphas from seed 7_000_000 + i * 1009 + s; the FT adapter is active in the student's
first forward only; TF offsets the block-l image rows of the first forward.
