#!/usr/bin/env bash
# Wan 2.1 end to end: bash scripts/wan.sh <stage> [<stage> ...]
#
#   encode       text contexts of the training captions, the VBench-300 prompts and the negative prompt
#   bank         ReDiverse-FT target bank (one worker per GPU)
#   train        ReDiverse-FT training (torchrun, one rank per GPU), then conversion to safetensors
#   directions   ReDiverse-TF directions for VBench-300 (one worker per GPU)
#   generate     the five arms on VBench-300 x 10 (one worker per GPU per arm)
#
# GPUS lists the devices to use (default 0 1 2 3 4 5 6 7; the released adapter was trained with
# 8 ranks). PY_TRAIN / PY_VIDEO are the interpreters of the train and video
# environments (default python). ARMS the arms to generate (default all five). ADAPTER is
# passed to wan.generate --adapter (default: the Hugging Face Hub; after the train stage use
# outputs/wan2.1/release).
set -euo pipefail
cd "$(dirname "$0")/.."

PY_TRAIN=${PY_TRAIN:-python}
PY_VIDEO=${PY_VIDEO:-python}
read -r -a G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a A <<< "${ARMS:-teacher student rediverse_ft rediverse_tf rediverse}"
ADAPTER=${ADAPTER:-hf}
OUT=${REDIVERSE_OUT:-outputs}
N=${#G[@]}

sharded() {   # sharded <python> <module> <args...>: one worker per GPU; @K@ in an argument becomes its index
    local py=$1; shift
    local k=0
    for g in "${G[@]}"; do
        local args=("${@//@K@/$k}")
        CUDA_VISIBLE_DEVICES=$g "$py" -m "${args[@]}" --device cuda:0 &
        k=$((k + 1))
    done
    wait
}

for stage in "$@"; do
    case "$stage" in
        encode)
            [ -f prompts/train_captions.txt ] || "$PY_VIDEO" prompts/download_train_captions.py
            CUDA_VISIBLE_DEVICES=${G[0]} "$PY_VIDEO" -m wan.encode_prompts --set train,vbench,neg --device cuda:0
            ;;
        bank)
            sharded "$PY_TRAIN" wan.build_targets --shard @K@ --num-shards "$N"
            ;;
        train)
            CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${G[*]}") PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                "$PY_TRAIN" -m torch.distributed.run --standalone --nproc_per_node "$N" -m wan.train
            "$PY_TRAIN" -m wan.convert_weights --src "$OUT/wan2.1/train/adapters/rediverse_ft_0000200.pth" \
                --out-dir "$OUT/wan2.1/release"
            ;;
        directions)
            sharded "$PY_VIDEO" wan.estimate_directions --shard @K@ --num-shards "$N"
            ;;
        generate)
            for arm in "${A[@]}"; do
                sharded "$PY_VIDEO" wan.generate --arm "$arm" --adapter "$ADAPTER" --shard @K@ --num-shards "$N"
            done
            ;;
        *)
            echo "unknown stage $stage (encode | bank | train | directions | generate)" >&2
            exit 1
            ;;
    esac
done
