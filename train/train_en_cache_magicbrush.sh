#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Preserve previous runs and keep MagicBrush dev/test out of training.
export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-./runs/editmgt_sparse_inter_edit_en_cache_magicbrush}"
# The shared launcher always uses pure cosine. Retain the corrected train/infer
# time units and per-example loss normalization.
export TIME_CONDITIONING="${TIME_CONDITIONING:-mask_ratio}"
export MASK_LOSS_REDUCTION="${MASK_LOSS_REDUCTION:-sample}"
export COND_DROPOUT_PROB="${COND_DROPOUT_PROB:-0.1}"
export SPARSE_TARGET_HALO="${SPARSE_TARGET_HALO:-1}"
exec bash "$PROJECT_ROOT/train/train_en_cache.sh" \
    --magicbrush_data_dir "${MAGICBRUSH_DATA_DIR:-$PROJECT_ROOT/data/magicbrush}" \
    --magicbrush_probability "${MAGICBRUSH_PROBABILITY:-0.2}" \
    "$@"
