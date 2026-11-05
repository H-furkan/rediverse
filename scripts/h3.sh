#!/usr/bin/env bash
# MiniMax-H3 end to end: bash scripts/h3.sh <stage> [<stage> ...]
#
#   encode       text contexts of the 20 training prompts and the 300 VBench prompts
#   bank         ReDiverse-FT target bank (one worker per GPU)
#   train        ReDiverse-FT training on the first two GPUs, then conversion to
#                outputs/minimax-h3/train/rediverse_ft/rediverse_ft.safetensors
#   directions   ReDiverse-TF directions (one worker per GPU)
#   generate     latents of the selected arms (one worker per GPU), then mp4s
#   probe        activation probing of teacher and student on the paper's probe prompts (one worker
#                per GPU), then the per-block report (video env; PY_VIDEO, PROBE_RAM_GB per worker, default 96)
#
# GPUS lists the devices to use (default 0 1 2 3 4 5 6 7); PY_VIDEO the interpreter of the video
# environment (default python); ARMS the arms to generate (default all five). ADAPTER is the
# ReDiverse-FT adapter of the rediverse_ft and rediverse arms (default the train stage's output,
# outputs/minimax-h3/train/rediverse_ft); the MiniMax-H3 adapter is not on the Hugging Face Hub.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY_VIDEO:-python}
read -r -a G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a ARM_LIST <<< "${ARMS:-student rediverse_ft rediverse_tf rediverse teacher}"
OUT=${REDIVERSE_OUT:-outputs}
ADAPTER=${ADAPTER:-$OUT/minimax-h3/train/rediverse_ft}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY_PROBE=${PY_VIDEO:-python}
PROBE_RAM_GB=${PROBE_RAM_GB:-96}

per_gpu() {   # per_gpu <module> <args...>: one worker per GPU, odd workers walk their list in reverse
    local n=0 pids=()
    for g in "${G[@]}"; do
        local rev=""
        if (( n % 2 == 1 )); then rev="$REVERSE_FLAG"; fi
        "$PY" -m "$@" --device "cuda:$g" $rev &
        pids+=("$!")
        n=$((n + 1))
    done
    for p in "${pids[@]}"; do wait "$p"; done
}

for stage in "$@"; do
    case "$stage" in
        encode)
            "$PY" -m h3.encode_prompts --set train --device "cuda:${G[0]}"
            "$PY" -m h3.encode_prompts --set vbench --device "cuda:${G[0]}"
            ;;
        bank)
            REVERSE_FLAG="" per_gpu h3.build_targets
            ;;
        train)
            "$PY" -m h3.train --device "cuda:${G[0]}" --dev1 "cuda:${G[1]}"
            "$PY" -m h3.convert_weights --src "$OUT/minimax-h3/train/adapters/rediverse_ft_0000200.pth" \
                --out-dir "$OUT/minimax-h3/train/rediverse_ft"
            ;;
        directions)
            REVERSE_FLAG="" per_gpu h3.estimate_directions
            ;;
        generate)
            for arm in "${ARM_LIST[@]}"; do
                if [[ "$arm" == rediverse_ft || "$arm" == rediverse ]] && [[ ! -e "$ADAPTER" ]]; then
                    echo "no ReDiverse-FT adapter at $ADAPTER: run the train stage or set ADAPTER" >&2
                    exit 2
                fi
                REVERSE_FLAG="--reverse" per_gpu h3.generate --arm "$arm" --adapter "$ADAPTER"
                REVERSE_FLAG="--reverse" per_gpu h3.generate --arm "$arm" --decode
            done
            ;;
        probe)
            for arm in teacher student; do
                n=${#G[@]}; k=0
                for g in "${G[@]}"; do
                    CUDA_VISIBLE_DEVICES=$g "$PY_PROBE" -m h3.probe --arm "$arm" --ram-gb "$PROBE_RAM_GB" \
                        --shard "$k" --num-shards "$n" --device cuda:0 &
                    k=$((k + 1))
                done
                wait
            done
            "$PY_PROBE" -m rediverse.probe_report --backbone minimax-h3 --tsv "$OUT/minimax-h3/probe/report.tsv" \
                --plot "$OUT/minimax-h3/probe/report.png"
            ;;
        *)
            echo "unknown stage: $stage (encode | bank | train | directions | generate | probe)" >&2
            exit 2
            ;;
    esac
done
