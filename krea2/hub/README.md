---
license: other
license_name: krea-2-community-license
license_link: LICENSE.pdf
base_model: krea/Krea-2-Turbo
base_model_relation: adapter
pipeline_tag: text-to-image
library_name: diffusers
tags:
- lora
- diversity
- distillation
- rediverse
- krea-2
extra_gated_prompt: "Krea 2 ReDiverse is a Derivative of Krea 2 and is governed by the Krea 2 Community License Agreement (LICENSE.pdf) and Krea's Acceptable Use Policy (https://www.krea.ai/krea-2-use-policy). By requesting access you agree to be bound by them, including the content-filtering requirement for any deployment."
extra_gated_button_content: "I agree to the Krea 2 Community License"
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

# Krea 2 ReDiverse

ReDiverse for [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo). Krea 2 Turbo generates 1024² images in eight
steps, but different seeds of the same prompt give nearly the same picture: the same composition, subject and framing.
Probing its activations against [Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw) shows the collapse at
`transformer_blocks.26` of 28 in the first sampling step: Turbo keeps (and inflates) the activation variance across
seeds but squeezes it into far fewer directions. ReDiverse puts the missing directions back at that block:

- **ReDiverse-FT**: a rank-32 LoRA on blocks 0–26 (`krea-2-rediverse.safetensors`, 414 MB), trained to match Krea 2 Raw's
  block-26 activations. It is active in the first of the eight sampling steps only.
- **ReDiverse-TF**: for each prompt, the top-8 principal directions of Krea 2 Raw at block 26 over 25 noises; every sample
  adds a random combination of them (α ~ U[−4, 4]) in the first step. No weights change.

The remaining seven steps run Krea 2 Turbo unchanged, so its look is kept.

<p align="center"><img src="assets/showcase.jpg" width="100%" alt="Seed-matched samples of Krea 2 Turbo (top of each pair) and Krea 2 ReDiverse (bottom)"></p>

| | Teacher | 8-step student | This release |
|---|---|---|---|
| | [Krea 2 Raw](https://huggingface.co/krea/Krea-2-Raw), 52 steps, CFG 4.5 (Krea guidance 3.5) | [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo), 8 steps, no guidance | ReDiverse-FT LoRA + ReDiverse-TF steering at `transformer_blocks.26`, step 1 |

## Results

Six GenEval prompts (bench, potted plant, car, dog, tie, cell phone) × 8 seeds, 1024², the same initial noises in every row. Diversity is the mean pairwise distance
between the samples of a prompt (×100, Vendi raw), higher is more diverse; "gap closed" is the share of the Turbo → Raw DINO
gap recovered. Quality is HPSv2.1 and ImageReward, reported as the change from Krea 2 Turbo.

| | DINO | gap closed | Vendi | DreamSim | CLIPScore | Δ HPSv2 | Δ ImageReward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Krea 2 Turbo | 49.4 | 0% | 2.38 | 22.6 | 29.7 | 0.00 | 0.000 |
| ReDiverse-TF only, α ~ U[−2, 2] | 50.2 | 4% | 2.54 | 24.5 | 29.5 | −0.37 | +0.042 |
| ReDiverse-FT only | 69.1 | 95% | 3.74 | 39.6 | 29.2 | −0.39 | −0.174 |
| ReDiverse, α ~ U[−2, 2] (paper setting) | 68.4 | 92% | 3.62 | 37.0 | 29.1 | −0.23 | −0.035 |
| **ReDiverse, α ~ U[−4, 4] (default here)** | **71.5** | **107%** | **4.10** | **43.0** | 29.1 | −1.00 | **−0.031** |
| *Krea 2 Raw (teacher, 52 steps)* | *70.0* | *100%* | *4.10* | *43.2* | *28.9* | *−2.23* | *−0.243* |

On this sweep, the default steering strength brings Krea 2 Turbo to Krea 2 Raw's diversity (DINO 71.5 vs 70.0, Vendi 4.10
vs 4.10, DreamSim 43.0 vs 43.2) while keeping Turbo's ImageReward; the paper's setting (α ~ U[−2, 2]) trades a little
diversity for the smallest quality change.

## Examples

Each figure: Krea 2 Turbo (top row) and Krea 2 ReDiverse (bottom row, default setting) from the same six initial noises,
seeds 0–5 as rendered.

| | |
|---|---|
| ![a wizard casting a spell](assets/examples/wizard.jpg) | ![a bowl of ramen](assets/examples/ramen.jpg) |
| ![a robot in a kitchen](assets/examples/robot.jpg) | ![a sailboat at sunset](assets/examples/sailboat.jpg) |
| ![a teddy bear on a bed](assets/examples/teddy.jpg) | ![a pair of sneakers](assets/examples/sneakers.jpg) |
| ![a castle on a hill](assets/examples/castle.jpg) | ![a child flying a kite](assets/examples/kite.jpg) |
| ![a woman playing the violin](assets/examples/violin.jpg) | ![a train crossing a bridge](assets/examples/train.jpg) |
| ![a red sports car on a road](assets/examples/car.jpg) | ![a portrait of an astronaut](assets/examples/astronaut.jpg) |

## Usage

You need access to [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) and, for the steering,
[krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) (both gated: accept Krea's license and log in with `hf auth login`),
and `diffusers` with `Krea2Pipeline` (0.39 or newer) and `transformers` 5.x.

```bash
hf download ReDiverse/Krea-2-ReDiverse --local-dir Krea-2-ReDiverse
# adapter + steering (default; estimates the prompt's teacher directions with Krea 2 Raw once, about a minute on an H100)
python Krea-2-ReDiverse/generate_krea2.py --prompt "a photo of a dog" --seeds 0-7 --out out --compare
# adapter only (no Krea 2 Raw needed)
python Krea-2-ReDiverse/generate_krea2.py --prompt "a photo of a dog" --seeds 0-7 --out out --mode ft --compare
# the paper's steering strength
python Krea-2-ReDiverse/generate_krea2.py --prompt "a photo of a dog" --seeds 0-7 --out out --band 2
```

This writes `out/rediverse/<seed>.png` and, with `--compare`, seed-matched `out/turbo/<seed>.png` rendered by Krea 2 Turbo
alone from the same noise. Directions are cached per prompt (`--cache`), so later seeds of the same prompt start at once.

The adapter alone in plain diffusers (keep it on for the first step only):

```python
import torch
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Turbo", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("ReDiverse/Krea-2-ReDiverse", weight_name="krea-2-rediverse.safetensors", adapter_name="rediverse")

def first_step_only(p, i, t, kw):
    if i == 0:
        p.transformer.disable_adapters()
    return kw

pipe.transformer.enable_adapters()
image = pipe("a photo of a dog", num_inference_steps=8, guidance_scale=0.0,
             callback_on_step_end=first_step_only).images[0]
```

`krea-2-rediverse-comfy.safetensors` holds the same weights under the original Krea 2 module names
(`diffusion_model.blocks.{i}.attn.wq ...`), as used by the [Krea 2 reference code](https://github.com/krea-ai/krea-2) and
ComfyUI; diffusers converts it on load as well. Keep it active in the first step only.

## Training details

| | |
|---|---|
| Collapsed block | `transformer_blocks.26` of 28 (probe: Turbo/Raw ratio at step 1 of participation ratio 0.49, total variance 3.4, concentration 7.1) |
| Target | Krea 2 Raw's block-26 image activations at its 5th step (σ 0.9674 → 0.9588, matched to Turbo's first step, σ 1 → 0.9567), after 4 steps with Krea guidance 3.5 from the same noise and caption |
| Adapter | LoRA r = 32, α = 32 on `attn.to_q/k/v/to_gate/to_out.0` and `ff.gate/up/down` of blocks 0–26 (216 modules, 103.5M parameters, fp32) |
| Data and optimisation | 100 DiffusionDB captions × 16 noises, 32 pairs per step, 200 steps, AdamW 1e-4, plain MSE, as in the paper |
| Steering | 25 conditional Krea 2 Raw rollouts per prompt, top-8 directions at block 26, step 5; α ~ U[−4, 4] per sample and direction |

Code for probing, training, steering and evaluation: [`krea2/`](https://github.com/H-furkan/rediverse/tree/main/krea2) in the ReDiverse repository.

## Limitations

- Krea 2 Raw is Krea's mid-training checkpoint, while Turbo is also post-trained (SFT, preference and RL tuning) besides
  distillation. The probe therefore measures post-training and distillation together, and the adapter can move Turbo's
  first step toward Raw's style; HPSv2 drops by up to 1 point at the default steering strength (Raw itself: −2.2).
- Trained and tested at 1024×1024; the steering directions are tied to that resolution.
- The results above are on GenEval-style prompts; long, very specific prompts leave less room for variation.
- Krea 2 Raw sometimes frames its images with a border or letterbox band; a few ReDiverse samples inherit it.

## License

Krea 2 ReDiverse is a Derivative of Krea 2, released under and governed by the [Krea 2 Community License Agreement](LICENSE.pdf)
and Krea's [Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy). Commercial use is allowed only within the license's
revenue threshold. Anyone deploying it must implement content filters as required by Section 4.2 of the license.
[NOTICE](NOTICE) holds the required notice. Krea 2 ReDiverse is not an official Krea product and is not endorsed by Krea.

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
