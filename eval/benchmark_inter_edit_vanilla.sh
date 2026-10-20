#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
NUM_GPUS="${NUM_GPUS:-8}"
INTER_EDIT_DATA_DIR="${INTER_EDIT_DATA_DIR:-/mnt/geminihzceph/public/data/Inter-Edit}"
INTER_EDIT_LANGUAGE="${INTER_EDIT_LANGUAGE:-en}"
ALPHA_CLIP_CKPT="${ALPHA_CLIP_CKPT:-$PROJECT_ROOT/checkpoint/AlphaCLIP/clip_l14_grit1m_fultune_8xe.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/inter_edit_en_vanilla}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# The native entry loads released EditMGT weights, using its original mask-free defaults.
exec .venv/bin/python -m eval.benchmark_inter_edit_vanilla \
    --num-gpus "$NUM_GPUS" \
    --data-dir "$INTER_EDIT_DATA_DIR" --language "$INTER_EDIT_LANGUAGE" \
    --alpha-clip-ckpt "$ALPHA_CLIP_CKPT" \
    --output-dir "$OUTPUT_DIR" "$@"
