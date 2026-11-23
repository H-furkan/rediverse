# Environments

The paper's numbers were produced in six Python environments with incompatible pins (for example,
SDXL generation needs diffusers 0.33 / transformers 4.49, MiniMax-H3 needs diffusers 0.40 /
transformers 5.x, VBench needs Python 3.10 / transformers 4.33). Each requirements file pins every
package to the version that produced the results; numerics in bf16 depend on these versions and on
the documented batch sizes.

| Environment | Python | torch | File |
|---|---|---|---|
| `train`  | 3.12 | 2.13.0 (CUDA 13.0, PyPI) | `requirements-train.txt` |
| `image`  | 3.12 | 2.13.0+cu126 | `requirements-image.txt` |
| `video`  | 3.12 | 2.13.0+cu126 | `requirements-video.txt` |
| `tpips`  | 3.12 | 2.13.0+cu126 | `requirements-tpips.txt` |
| `vbench` | 3.10 | 2.7.1+cu126  | `requirements-vbench.txt` |
| `vipe`   | 3.10 | 2.7.1+cu126  | `vipe.md` + `constraints-vipe.txt` |
| `krea2`  | 3.12 | 2.9.1+cu128  | `requirements-krea2.txt` (direct dependencies; installed with resolution) |

```bash
python3.12 -m venv venvs/train && venvs/train/bin/pip install --no-deps -r envs/requirements-train.txt
python3.12 -m venv venvs/image && venvs/image/bin/pip install --no-deps -r envs/requirements-image.txt
python3.12 -m venv venvs/video && venvs/video/bin/pip install --no-deps -r envs/requirements-video.txt
python3.12 -m venv venvs/tpips && venvs/tpips/bin/pip install --no-deps -r envs/requirements-tpips.txt
python3.10 -m venv venvs/vbench && venvs/vbench/bin/pip install --no-deps -r envs/requirements-vbench.txt
# vipe: see envs/vipe.md
python3.12 -m venv venvs/krea2 && venvs/krea2/bin/pip install -r envs/requirements-krea2.txt
```

Each file lists the complete package set of its reference environment, so it is installed with
`--no-deps`: two of the reference environments violate a dependency's declared pins (`image-reward`
1.5 declares `timm==0.6.13`, `opencv-python-headless` 5.0 declares `numpy>=2`), which a dependency
resolver would reject. The CUDA 12.6 files fetch torch from
`https://download.pytorch.org/whl/cu126` (an `--extra-index-url` line at the top of each file);
`requirements-train.txt` uses the PyPI wheels, which are the CUDA 13.0 builds.

## Pipeline stage -> environment

| Stage | Environment |
|---|---|
| SDXL: training-caption encoding, ReDiverse-FT training, adapter extraction and conversion | `train` |
| SDXL: evaluation noise bank, generation (all five arms), ReDiverse-TF directions | `image` |
| SD3: caption encoding, evaluation noise bank, ReDiverse-FT training, generation (all five arms), ReDiverse-TF directions | `train` |
| Wan 2.1: training-caption and VBench text contexts | `video` |
| Wan 2.1: ReDiverse-FT target bank and training | `train` |
| Wan 2.1: generation (all five arms), ReDiverse-TF directions | `video` |
| MiniMax-H3: text contexts, target bank, ReDiverse-FT training, ReDiverse-TF directions, generation, decoding | `video` |
| Activation probing of all four backbones (`sdxl.probe`, `sd3.probe`, `wan.probe`, `h3.probe`) and `rediverse.probe_report` | `video` |
| Image metrics: DINO, Vendi, DreamSim, CLIPScore features; LPIPS, HPSv2.1, ImageReward; image tables (PickScore is optional and in no paper table) | `image` |
| TPIPS (images and video), video DINO and V-JEPA 2 features, video metric aggregation and tables | `tpips` |
| ViCLIP scores, VBench subject / background consistency | `vbench` |
| Camera trajectories (ViPE) | `vipe` |
| Krea 2: every stage (`krea2.*`: probe, directions, target bank, training, generation, sweep, conversion) | `krea2` |
| `prompts/download_train_captions.py` | any (standard library only) |

## System dependencies

- An NVIDIA GPU and a driver that supports CUDA 13.0 (the `train` environment) and CUDA 12.6 (all
  others). Training and generation were run on 80 GB H100 GPUs.
- `ffmpeg` and `ffprobe` built with `libx264`, on `PATH`. Every generated video is encoded by an
  ffmpeg rawvideo pipe (H.264, yuv420p, CRF 17) and every video metric decodes frames with ffmpeg;
  the reference build is ffmpeg 6.1.1.
- `git` (the `image` environment installs OpenAI CLIP from its repository at a pinned commit).
- For ViPE only: a CUDA 12.6 toolkit with `nvcc` (see `vipe.md`).

## Runtime variables

- `HF_HOME` is honoured everywhere for model downloads; nothing in the code sets it.
- `REDIVERSE_DATA` (default `./data`) and `REDIVERSE_OUT` (default `./outputs`) relocate data and
  outputs.
- `vbench` environment: `VBENCH_CACHE_DIR` (VBench model weights) and
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, as in the reference runs (`evaluate_set.sh` sets it).
- SDXL training (FastGen) needs `HF_HOME` set: FastGen loads the teacher with
  `cache_dir=$HF_HOME`. `sdxl/setup_fastgen.sh` needs `git` and network access; it clones
  NVlabs/FastGen at commit `060b857` into `third_party/FastGen`.
- The end-to-end drivers in `scripts/` take each environment's interpreter as `PY_TRAIN`,
  `PY_IMAGE` or `PY_VIDEO` (default `python`); see [`scripts/README.md`](../scripts/README.md).
- `evaluation/video/evaluate_set.sh` runs the video metrics with `PY_TPIPS`, `PY_VBENCH` and
  `PY_VIPE` (default `venvs/<env>/bin/python` under the repository root);
  `evaluation/image/evaluate_set.sh` needs `PY_IMAGE` and `PY_TPIPS` set.
- `evaluation/video/camtraj_vipe.py` reads `VIPE_CUDA_HOME` (a CUDA 12.6 toolkit whose `bin/` and
  `lib64/` are put in front of `PATH` / `LD_LIBRARY_PATH` of the ViPE process) and `VIPE_BIN` (the
  ViPE executable; default the `vipe` script next to its interpreter).
- `DREAMSIM_CACHE` holds the DreamSim checkpoint (default `<torch hub dir>/dreamsim`). The LPIPS
  VGG16 trunk goes to the torch hub directory, and ImageReward downloads into `~/.cache/ImageReward`
  regardless of `HF_HOME`.

## Licenses of installed packages

`requirements-tpips.txt` installs `tpips==0.1.2` from PyPI. Its package metadata declares the Adobe
Research License (`LicenseRef-Adobe-Research-License`), which permits noncommercial research use
only.

## Backbone notes

Backbone-specific environment details, by package.

### SD3

- SD3-Medium is a gated model: accept its license on the Hub and run `hf auth login` before the
  first run.
- Every SD3 stage runs in `train`. Reproducing the released images pixel for pixel needs that
  environment's diffusers 0.37.1 and transformers 4.56.2 and the default batch of 10 latents per
  pipeline call.

### Wan 2.1

- The text contexts are encoded in `video` (transformers 5.x for the UMT5 tokenizer).
- The ReDiverse-FT target bank and training run in `train` (diffusers 0.37.1, peft 0.19.1).
- Generation and the ReDiverse-TF directions run in `video`.
- Training runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the released adapter was
  trained with `torchrun` on 8 ranks.

### MiniMax-H3

- Every stage runs in `video`.
- The text encoder, the transformer and the video VAE each need most of an 80 GB GPU, so they run
  as separate phases (text contexts, latents, decoding).
- Training needs two 80 GB GPUs.
- Each ReDiverse-TF direction worker needs about 30 GB of host RAM.
- After the first download of the models, `HF_HUB_OFFLINE=1` works.

### Krea 2

- Every `krea2` stage runs in `krea2` (diffusers 0.39 for `Krea2Pipeline`; transformers 5.x for Krea 2's tokenizer and
  Qwen3-VL text-encoder configs). Image metrics run in `image` and `tpips` as for the other backbones.
- krea/Krea-2-Turbo and krea/Krea-2-Raw are gated: accept Krea's license on the Hub and run `hf auth login`.
- One transformer is 25.6 GB in bf16; generation, directions and the target bank need one GPU per worker, training about
  45 GB per rank (two ranks: ~2.2 h on RTX PRO 6000) and 80 GB of host RAM per rank for the target bank.
- After the first download of the models, `HF_HUB_OFFLINE=1` works.
