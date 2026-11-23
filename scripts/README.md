# End-to-end drivers

Thin shell drivers that chain the stages of one backbone over several GPUs. Run them from anywhere;
they change to the repository root. Each stage runs the commands documented in the backbone README.

| Driver | Stages | Variables |
|---|---|---|
| `bash scripts/sdxl.sh <stage> ...` | `data` (noise bank), `train` (FastGen checkout, captions, init checkpoint, ReDiverse-FT + conversion), `directions`, `generate`, `probe` | `GPUS`, `PY_IMAGE` (`image` env), `PY_TRAIN` (`train` env), `PY_VIDEO` (`video` env, `probe`), `ARMS`, `ADAPTER`, `PROBE_RAM_GB`; `HF_HOME` (required by `train`) |
| `bash scripts/sd3.sh <stage> ...` | `data` (captions, embeddings, noise bank), `train` (ReDiverse-FT + conversion), `directions`, `generate`, `probe` | `GPUS`, `PY_TRAIN` (`train` env), `PY_VIDEO` (`video` env, `probe`), `ARMS`, `ADAPTER`, `PROBE_RAM_GB` |
| `bash scripts/wan.sh <stage> ...` | `encode`, `bank` (ReDiverse-FT targets), `train` (torchrun + conversion), `directions`, `generate`, `probe` | `GPUS`, `PY_TRAIN` (`train` env), `PY_VIDEO` (`video` env), `ARMS`, `ADAPTER`, `PROBE_RAM_GB` |
| `bash scripts/h3.sh <stage> ...` | `encode`, `bank` (ReDiverse-FT targets), `train` (two GPUs + conversion), `directions`, `generate` (latents, then mp4s), `probe` | `GPUS`, `PY_VIDEO` (`video` env), `ARMS`, `ADAPTER`, `PROBE_RAM_GB` |
| `bash scripts/krea2.sh <stage> ...` | `probe`, `directions`, `bank` (ReDiverse-FT targets), `train` (torchrun + conversion), `generate` | `GPUS`, `PY` (`krea2` env), `BAND` (ReDiverse-TF α range, default 4; the paper uses 2), `ADAPTER` |

- `GPUS` lists the devices (default `0 1 2 3 4 5 6 7`); sharded stages start one worker per device.
- `PY_<ENV>` is the Python interpreter of an environment of [`envs/`](../envs/README.md), e.g.
  `venvs/train/bin/python` for `PY_TRAIN`; each defaults to `python`.
- `HF_HOME` must be exported for the SDXL `train` stage: FastGen reads the SDXL teacher from it.
  The other stages use the default Hugging Face cache when it is unset.
- `ARMS` selects the arms to generate (default: all five).
- `ADAPTER` is passed to `--adapter` of the generator. It defaults to `hf`
  (the Hugging Face Hub); after a `train` stage, point it at the converted adapter:
  `outputs/sdxl/release`, `outputs/sd3/release`, `outputs/wan2.1/release` or
  `outputs/minimax-h3/train/rediverse_ft`.
- `probe` runs the teacher and then the student on the paper's probe prompts, one shard per device,
  and writes `outputs/<backbone>/probe/report.{tsv,png}`. `PROBE_RAM_GB` (default 96) is each
  worker's staging budget for activations, so the host needs about `PROBE_RAM_GB` x the number of
  devices of free memory. A block larger than the budget is staged whole: one MiniMax-H3 block at the
  default N = 1000 is about 400 GB (1000 x 37,296 x 5376 bf16 values), so each `h3.sh probe` worker
  needs at least that much free host RAM whatever `PROBE_RAM_GB` is; on a smaller host list fewer
  `GPUS` (e.g. `GPUS=0` runs the workers one at a time).
- `REDIVERSE_DATA` / `REDIVERSE_OUT` relocate data and outputs, as for the Python entry points.

Example:

```bash
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} GPUS="0 1 2 3" \
  PY_IMAGE=venvs/image/bin/python PY_TRAIN=venvs/train/bin/python ADAPTER=outputs/sdxl/release \
  bash scripts/sdxl.sh data train directions generate
```

Evaluation has its own per-set drivers, `evaluation/image/evaluate_set.sh` and
`evaluation/video/evaluate_set.sh` (see [`evaluation/README.md`](../evaluation/README.md)).
