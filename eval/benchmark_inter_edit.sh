#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
NUM_GPUS="${NUM_GPUS:-8}"
# Edit these defaults here; environment variables and trailing CLI arguments can override them.
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-7200}"
INTER_EDIT_DATA_DIR="${INTER_EDIT_DATA_DIR:-/mnt/geminihzceph/public/data/Inter-Edit}"
INTER_EDIT_LANGUAGE="${INTER_EDIT_LANGUAGE:-en}"
ALPHA_CLIP_CKPT="${ALPHA_CLIP_CKPT:-$PROJECT_ROOT/checkpoint/AlphaCLIP/clip_l14_grit1m_fultune_8xe.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/inter_edit_en_checkpoint_7200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec .venv/bin/python -m eval.benchmark_inter_edit \
    --num-gpus "$NUM_GPUS" \
    --checkpoint "$CHECKPOINT" \
    --data-dir "$INTER_EDIT_DATA_DIR" --language "$INTER_EDIT_LANGUAGE" \
    --alpha-clip-ckpt "$ALPHA_CLIP_CKPT" \
    --output-dir "$OUTPUT_DIR" "$@"
