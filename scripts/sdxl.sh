#!/usr/bin/env bash
# SDXL / DMD2 end to end: bash scripts/sdxl.sh <stage> [<stage> ...]
#
#   data         evaluation noise bank (image env)
#   train        FastGen checkout, training captions and their embeddings, the initialisation
#                checkpoint, ReDiverse-FT training on the first GPU, conversion to safetensors
#                (train env)
#   directions   ReDiverse-TF directions for GenEval-553, one worker per GPU (image env)
#   generate     the five arms on GenEval-553 x 100, one shard per GPU per arm (image env)
#
# GPUS lists the devices to use (default 0 1 2 3 4 5 6 7); PY_IMAGE / PY_TRAIN the interpreters of
# the image and train environments (default python); ARMS the arms to generate (default all five);
# ADAPTER is passed to sdxl.generate --adapter (default: the Hugging Face Hub; after the train stage
# use outputs/sdxl/release). The train stage needs HF_HOME (FastGen reads it).
set -euo pipefail
cd "$(dirname "$0")/.."

PY_IMAGE=${PY_IMAGE:-python}
PY_TRAIN=${PY_TRAIN:-python}
read -r -a G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a A <<< "${ARMS:-teacher student rediverse_ft rediverse_tf rediverse}"
ADAPTER=${ADAPTER:-hf}
OUT=${REDIVERSE_OUT:-outputs}

for stage in "$@"; do
    case "$stage" in
        data)
            "$PY_IMAGE" -m sdxl.make_noise_bank --device "cuda:${G[0]}"
            ;;
        train)
            : "${HF_HOME:?export HF_HOME first (FastGen reads the SDXL-base UNet from cache_dir=\$HF_HOME)}"
            [ -d third_party/FastGen/.git ] || bash sdxl/setup_fastgen.sh
            [ -f prompts/train_captions.txt ] || "$PY_TRAIN" prompts/download_train_captions.py
            "$PY_TRAIN" -m sdxl.encode_captions --device "cuda:${G[0]}"
            "$PY_TRAIN" -m sdxl.build_init_ckpt
            CUDA_VISIBLE_DEVICES=${G[0]} "$PY_TRAIN" -m sdxl.train
            "$PY_TRAIN" -m sdxl.convert_weights --src "$OUT/sdxl/train/adapters/rediverse_ft_0000200.pth" \
                --out-dir "$OUT/sdxl/release"
            ;;
        directions)
            n=${#G[@]}; k=0
            for g in "${G[@]}"; do
                "$PY_IMAGE" -m sdxl.estimate_directions --shard "$k" --num-shards "$n" --device "cuda:$g" &
                k=$((k + 1))
            done
            wait
            ;;
        generate)
            n=${#G[@]}
            for arm in "${A[@]}"; do
                k=0
                for g in "${G[@]}"; do
                    "$PY_IMAGE" -m sdxl.generate --arm "$arm" --adapter "$ADAPTER" \
                        --shard "$k" --num-shards "$n" --device "cuda:$g" &
                    k=$((k + 1))
                done
                wait
            done
            ;;
        *)
            echo "unknown stage $stage (data | train | directions | generate)" >&2
            exit 1
            ;;
    esac
done
