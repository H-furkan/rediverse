# End-to-end drivers

Thin shell drivers that chain the stages of one backbone over several GPUs. Run them from anywhere;
they change to the repository root. Each stage runs the commands documented in the backbone README.

| Driver | Stages | Variables |
|---|---|---|
| `bash scripts/sdxl.sh <stage> ...` | `data` (noise bank), `train` (FastGen checkout, captions, init checkpoint, ReDiverse-FT + conversion), `directions`, `generate`, `probe` | `GPUS`, `PY_IMAGE` (`image` env), `PY_TRAIN` (`train` env), `PY_VIDEO` (`video` env, `probe`), `ARMS`, `ADAPTER`, `PROBE_RAM_GB`; `HF_HOME` (required by `train`) |
| `bash scripts/sd3.sh <stage> ...` | `data` (captions, embeddings, noise bank), `train` (ReDiverse-FT + conversion), `directions`, `generate`, `probe` | `GPUS`, `PY_TRAIN` (`train` env), `PY_VIDEO` (`video` env, `probe`), `ARMS`, `ADAPTER`, `PROBE_RAM_GB` |
| `bash scripts/wan.sh <stage> ...` | `encode`, `bank` (ReDiverse-FT targets), `train` (torchrun + conversion), `directions`, `generate`, `probe` | `GPUS`, `PY_TRAIN` (`train` env), `PY_VIDEO` (`video` env), `ARMS`, `ADAPTER`, `PROBE_RAM_GB` |
| `bash scripts/h3.sh <stage> ...` | `encode`, `bank` (ReDiverse-FT targets), `train` (two GPUs + conversion), `directions`, `generate` (latents, then mp4s), `probe` | `GPUS`, `PY_VIDEO` (`video` env), `ARMS`, `ADAPTER`, `PROBE_RAM_GB` |

- `GPUS` lists the devices (default `0 1 2 3 4 5 6 7`); sharded stages start one worker per device.
- `PY_<ENV>` is the Python interpreter of an environment of [`envs/`](../envs/README.md), e.g.
  `venvs/train/bin/python` for `PY_TRAIN`; each defaults to `python`.
- `HF_HOME` must be exported for the SDXL `train` stage: FastGen reads the SDXL teacher from it.
  The other stages use the default Hugging Face cache when it is unset.
- `ARMS` selects the arms to generate (default: all five).
- `ADAPTER` is passed to `--adapter` of the generator. For SDXL, SD3 and Wan 2.1 it defaults to `hf`
  (the Hugging Face Hub); after a `train` stage, point it at the converted adapter:
  `outputs/sdxl/release`, `outputs/sd3/release` or `outputs/wan2.1/release`. For MiniMax-H3 it
  defaults to `outputs/minimax-h3/train/rediverse_ft`, where the `train` stage converts the adapter,
  since that adapter is not on the Hub; `generate` stops with a message if an FT arm is selected
  and the adapter does not exist.
- `probe` runs the teacher and then the student on the paper's probe prompts, one shard per device,
  and writes `outputs/<backbone>/probe/report.{tsv,png}`. `PROBE_RAM_GB` (default 96) is each
  worker's staging budget for activations, so the host needs about `PROBE_RAM_GB` x the number of
  devices of free memory.
- `REDIVERSE_DATA` / `REDIVERSE_OUT` relocate data and outputs, as for the Python entry points.

Example:

```bash
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} GPUS="0 1 2 3" \
  PY_IMAGE=venvs/image/bin/python PY_TRAIN=venvs/train/bin/python \
  bash scripts/sdxl.sh data train directions generate
```

Evaluation has its own per-set drivers, `evaluation/image/evaluate_set.sh` and
`evaluation/video/evaluate_set.sh` (see [`evaluation/README.md`](../evaluation/README.md)).
