#!/usr/bin/env bash
# SD3-Medium / TDM end to end: bash scripts/sd3.sh <stage> [<stage> ...]
#
#   data         training captions, their embeddings + the empty prompt, the evaluation noise bank
#   train        ReDiverse-FT training on the first GPU, then conversion to safetensors
#   directions   ReDiverse-TF directions for GenEval-553 (one worker per GPU)
#   generate     the five arms on GenEval-553 x 100 (one worker per GPU per arm)
#   probe        activation probing of teacher and student on the paper's probe prompts (one worker
#                per GPU), then the per-block report (video env; PY_VIDEO, PROBE_RAM_GB per worker, default 96)
#
# GPUS lists the devices to use (default 0 1 2 3 4 5 6 7); PY_TRAIN the interpreter of the train
# environment (default python); ARMS the arms to generate (default all five). ADAPTER is passed to
# sd3.generate --adapter (default: the Hugging Face Hub; after the train stage use
# outputs/sd3/release).
set -euo pipefail
cd "$(dirname "$0")/.."

PY_TRAIN=${PY_TRAIN:-python}
read -r -a G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a A <<< "${ARMS:-teacher student rediverse_ft rediverse_tf rediverse}"
ADAPTER=${ADAPTER:-hf}
OUT=${REDIVERSE_OUT:-outputs}
PY_PROBE=${PY_VIDEO:-python}
PROBE_RAM_GB=${PROBE_RAM_GB:-96}

for stage in "$@"; do
    case "$stage" in
        data)
            [ -f prompts/train_captions.txt ] || "$PY_TRAIN" prompts/download_train_captions.py
            "$PY_TRAIN" -m sd3.encode_prompts --device "cuda:${G[0]}"
            "$PY_TRAIN" -m sd3.make_noise_bank --device "cuda:${G[0]}"
            ;;
        train)
            "$PY_TRAIN" -m sd3.train --device "cuda:${G[0]}"
            "$PY_TRAIN" -m sd3.convert_weights --src "$OUT/sd3/train/adapters/rediverse_ft_0000200.pth" \
                --out-dir "$OUT/sd3/release"
            ;;
        directions)
            n=${#G[@]}; k=0
            for g in "${G[@]}"; do
                "$PY_TRAIN" -m sd3.estimate_directions --shard "$k" --num-shards "$n" --device "cuda:$g" &
                k=$((k + 1))
            done
            wait
            ;;
        generate)
            for arm in "${A[@]}"; do
                for g in "${G[@]}"; do
                    "$PY_TRAIN" -m sd3.generate --arm "$arm" --adapter "$ADAPTER" --device "cuda:$g" &
                done
                wait
            done
            ;;
        probe)
            for arm in teacher student; do
                n=${#G[@]}; k=0
                for g in "${G[@]}"; do
                    CUDA_VISIBLE_DEVICES=$g "$PY_PROBE" -m sd3.probe --arm "$arm" --ram-gb "$PROBE_RAM_GB" \
                        --shard "$k" --num-shards "$n" --device cuda:0 &
                    k=$((k + 1))
                done
                wait
            done
            "$PY_PROBE" -m rediverse.probe_report --backbone sd3 --tsv "$OUT/sd3/probe/report.tsv" \
                --plot "$OUT/sd3/probe/report.png"
            ;;
        *)
            echo "unknown stage $stage (data | train | directions | generate | probe)" >&2
            exit 1
            ;;
    esac
done
