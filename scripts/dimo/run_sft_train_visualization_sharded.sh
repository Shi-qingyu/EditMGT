#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/inter_edit_train/sft_checkpoint_11000_large}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-11000}"
GPU_IDS="${GPU_IDS:-0,2,3,5}"
CACHE_THRESHOLD="${CACHE_THRESHOLD:-0.3}"
IFS=',' read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} < 1 )); then
    echo 'GPU_IDS must contain at least one GPU' >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR/logs"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pids=()
for rank in "${!gpus[@]}"; do
    gpu="${gpus[$rank]}"
    echo "Launching shard $rank/${#gpus[@]} on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m scripts.dimo.visualize_inter_edit_train sft \
        --output-dir "$OUTPUT_DIR" --sft-checkpoint "$SFT_CHECKPOINT" \
        --condition-cache-threshold "$CACHE_THRESHOLD" \
        --reference-kv-cache-threshold "$CACHE_THRESHOLD" \
        --num-shards "${#gpus[@]}" --shard-index "$rank" \
        > "$OUTPUT_DIR/logs/shard_$rank.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
    echo "At least one visualization shard failed; see $OUTPUT_DIR/logs" >&2
    exit 1
fi
echo "All visualization shards completed: $OUTPUT_DIR"
