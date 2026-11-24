---
license: apache-2.0
base_model:
  - Qwen/Qwen-Image
  - lightx2v/Qwen-Image-Lightning
base_model_relation: adapter
pipeline_tag: text-to-image
library_name: diffusers
tags:
  - rediverse
  - diversity
  - lora
  - qwen-image
  - qwen-image-lightning
  - few-step
  - text-to-image
---

<p align="center"><img src="https://huggingface.co/spaces/ReDiverse/README/resolve/main/assets/banner.jpg" width="100%" alt="ReDiverse: Restoring Diversity in Distilled Image and Video Generators by Probing Activations"></p>

<p align="center">
  Furkan Horoz<sup>1</sup>, Ozgur Kara<sup>1</sup>, Kaan Oktay<sup>2</sup>, Adil Kaan Akan<sup>2</sup>, Atakan Kara<sup>2</sup>,
  Anand Bhattad<sup>3,†</sup>, James Matthew Rehg<sup>1,†</sup><br>
  <sup>1</sup>University of Illinois at Urbana-Champaign &nbsp;&nbsp; <sup>2</sup>fal &nbsp;&nbsp; <sup>3</sup>Johns Hopkins University<br>
  <sub><sup>†</sup>Equal advising.</sub>
</p>

<p align="center">
  <a href="https://rediverse-distillation.github.io/rediverse-distillation/"><img src="https://img.shields.io/badge/Project-Page-2ea44f.svg" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2611.18550"><img src="https://img.shields.io/badge/arXiv-2611.18550-b31b1b.svg?logo=arxiv" alt="arXiv 2611.18550"></a>
  <a href="https://github.com/H-furkan/rediverse"><img src="https://img.shields.io/badge/GitHub-Code-181717.svg?logo=github" alt="Code on GitHub"></a>
  <a href="https://huggingface.co/ReDiverse"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20ReDiverse-all%20models-ffd21e.svg" alt="All ReDiverse models"></a>
</p>

# Qwen-Image-Lightning ReDiverse

ReDiverse for [Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) (the 8-step V2.0 LoRA on
[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)). Qwen-Image-Lightning generates 1024² images in eight steps, but
different seeds of the same prompt give nearly the same picture: the same subject, pose, colour and framing. Probing its
activations against Qwen-Image shows the collapse at `transformer_blocks.57` of 60 in the first sampling step: the student
keeps the activation variance across seeds but squeezes it into fewer directions. ReDiverse puts the missing directions
back at that block:

- **ReDiverse-FT**: a rank-32 LoRA on blocks 0–57 (`qwen-image-lightning-rediverse.safetensors`, 783 MiB), trained to match
  Qwen-Image's block-57 activations. It is active in the first of the eight sampling steps only, on top of the Lightning LoRA.
- **ReDiverse-TF**: for each prompt, the top-8 principal directions of Qwen-Image at block 57 over 25 noises; every sample
  adds a random combination of them (α ~ U[−4, 4]) in the first step. No weights change.

The remaining seven steps run Qwen-Image-Lightning unchanged, so its look is kept. The teacher is the same Qwen-Image
model with the Lightning LoRA turned off, so the steering needs no second model.

<p align="center"><img src="assets/showcase.jpg" width="100%" alt="Seed-matched samples of Qwen-Image-Lightning (top of each pair) and Qwen-Image-Lightning ReDiverse (bottom)"></p>

| | Teacher | 8-step student | This release |
|---|---|---|---|
| | [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), 50 steps, true CFG 4 | [Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) 8-step V2.0, no guidance | ReDiverse-FT LoRA + ReDiverse-TF steering at `transformer_blocks.57`, step 1 |

## Results

Preview evaluation: 6 GenEval prompts (bench, potted plant, car, dog, tie, cell phone) × 8 seeds, 1024², the same initial
noises in every row. Diversity is the mean pairwise distance between the samples of a prompt (×100, Vendi raw), higher is
more diverse; "gap closed" is the share of the Lightning → Qwen-Image DINO gap recovered. Quality is HPSv2.1 (×100) and
ImageReward, reported as the change from Qwen-Image-Lightning.

| | DINO | gap closed | Vendi | DreamSim | CLIPScore | Δ HPSv2 | Δ ImageReward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Image-Lightning | 29.0 | 0% | 1.66 | 14.1 | 28.9 | 0.00 | 0.000 |
| ReDiverse-TF only, α ~ U[−2, 2] | 46.3 | 165% | 2.25 | 24.1 | 29.4 | −0.10 | −0.071 |
| ReDiverse-FT only | 48.2 | 183% | 2.41 | 22.5 | 29.0 | +0.41 | +0.025 |
| ReDiverse, α ~ U[−2, 2] (paper setting) | 48.4 | 185% | 2.32 | 23.6 | 29.3 | +0.11 | −0.027 |
| **ReDiverse, α ~ U[−4, 4] (default here)** | **50.1** | **202%** | **2.41** | **25.7** | 29.5 | −0.14 | −0.110 |
| *Qwen-Image (teacher, 50 steps, CFG 4)* | *39.5* | *100%* | *2.03* | *16.9* | *29.7* | *−0.34* | *+0.017* |

At 1024² with plain GenEval prompts, Qwen-Image itself is not very diverse, so every ReDiverse setting goes past the
teacher's diversity; the added variety stays on the prompt (car colours and viewpoints, a tie worn or laid flat, a phone
held, on a table or at an angle). The paper setting changes quality the least; the default setting gives the most variety
at an ImageReward cost of about 0.1. This is a preview on 48 images per row, not the paper's full protocol.

## Examples

Seed-matched: the top row of each pair is Qwen-Image-Lightning, the bottom row Qwen-Image-Lightning ReDiverse (default).

| | |
|---|---|
| ![a photo of a car](assets/examples/car.jpg) | ![a photo of a tie](assets/examples/tie.jpg) |
| ![a photo of a cell phone](assets/examples/cell_phone.jpg) | ![a photo of a dog](assets/examples/dog.jpg) |
| ![a photo of a bench](assets/examples/bench.jpg) | ![a photo of a potted plant](assets/examples/potted_plant.jpg) |

## Usage

You need `diffusers` with `QwenImagePipeline` (0.43 tested), `transformers` and `peft`. The script downloads
[Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), the
[Qwen-Image-Lightning 8-step V2.0 LoRA](https://huggingface.co/lightx2v/Qwen-Image-Lightning) and this adapter.

```bash
hf download ReDiverse/Qwen-Image-Lightning-ReDiverse --local-dir Qwen-Image-Lightning-ReDiverse
# adapter + steering (default; estimates the prompt's teacher directions with the Qwen-Image base once)
python Qwen-Image-Lightning-ReDiverse/generate_qwen_image_lightning.py --prompt "a photo of a dog" --seeds 0-7 --out out --compare
# adapter only
python Qwen-Image-Lightning-ReDiverse/generate_qwen_image_lightning.py --prompt "a photo of a dog" --seeds 0-7 --out out --mode ft --compare
# the paper's steering strength
python Qwen-Image-Lightning-ReDiverse/generate_qwen_image_lightning.py --prompt "a photo of a dog" --seeds 0-7 --out out --band 2
```

This writes `out/rediverse/<seed>.png` and, with `--compare`, seed-matched `out/lightning/<seed>.png` rendered by
Qwen-Image-Lightning alone from the same noise. Directions are cached per prompt (`--cache`), so later seeds of the same
prompt start at once.

The adapter alone in plain diffusers (keep it on for the first step only, together with the Lightning LoRA):

```python
import math
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, base_shift=math.log(3), max_shift=math.log(3), shift_terminal=None)   # Lightning's schedule
pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
                       adapter_name="lightning")
pipe.load_lora_weights("ReDiverse/Qwen-Image-Lightning-ReDiverse", weight_name="qwen-image-lightning-rediverse.safetensors",
                       adapter_name="rediverse")

def first_step_only(p, i, t, kw):
    if i == 0:
        p.set_adapters(["lightning"], adapter_weights=[1.0])
    return kw

pipe.set_adapters(["lightning", "rediverse"], adapter_weights=[1.0, 1.0])
image = pipe("a photo of a dog", width=1024, height=1024, num_inference_steps=8, true_cfg_scale=1.0,
             callback_on_step_end=first_step_only).images[0]
```

## Training details

| | |
|---|---|
| Collapsed block | `transformer_blocks.57` of 60 (probe, 20 prompts × 200 noises: Lightning/Qwen-Image ratio at step 1 of participation ratio 0.85, total variance 1.13, concentration 1.42; the concentration steps up at blocks 56–58) |
| Target | Qwen-Image's block-57 image-stream activations at its 4th step (σ 0.9685 → 0.9575, matched to Lightning's first step, σ 1 → 0.9545), after 3 steps with true CFG 4 (negative prompt " ") from the same noise and caption |
| Adapter | LoRA r = 32, α = 32 on the image- and text-stream attention projections (`attn.to_q/k/v/to_out.0`, `attn.add_q/k/v_proj`, `attn.to_add_out`) and both MLPs (`img_mlp`, `txt_mlp`) of blocks 0–57 (696 modules, 205.3M parameters, fp32), trained on top of the Lightning LoRA |
| Data and optimisation | 100 DiffusionDB captions × 16 noises, 32 pairs per step, 200 steps, AdamW 1e-4, plain MSE, as in the paper |
| Steering | 25 conditional Qwen-Image rollouts per prompt, top-8 directions at block 57, step 4; α ~ U[−4, 4] per sample and direction |

Code for probing, training, steering and evaluation: [`qwenimage/`](https://github.com/H-furkan/rediverse/tree/main/qwenimage) in the ReDiverse repository.

## License

Qwen-Image-Lightning ReDiverse is released under the [Apache License 2.0](LICENSE). It is an adapter trained on and applied
to [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Apache License 2.0) together with
[lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) (Apache License 2.0); use of those
models remains subject to their own licenses. [NOTICE](NOTICE) lists the attributions. It is not an official Qwen or
LightX2V product and is not endorsed by either team.

## Citation

```bibtex
@article{horoz2026rediverse,
  title   = {Restoring Diversity in Distilled Image and Video Generators by Probing Activations},
  author  = {Horoz, Furkan and Kara, Ozgur and Oktay, Kaan and Akan, Adil Kaan and Kara, Atakan
             and Bhattad, Anand and Rehg, James Matthew},
  journal = {arXiv preprint arXiv:2611.18550},
  year    = {2026}
}
```
