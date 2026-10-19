#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Start a separate experiment with corrected MagicBrush alpha masks. The old
# no-dilation run used luminance masks and must not supply a resume checkpoint.
export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix}"
export SPARSE_TARGET_HALO=0
export TIME_CONDITIONING=mask_ratio
export MASK_LOSS_REDUCTION=sample

# Keep these profile settings after user arguments: training and validation
# use exactly the original VQ edit mask, with no halo-supervision expansion.
exec bash "$PROJECT_ROOT/scripts/train_en_cache_magicbrush.sh" "$@" \
    --fresh_start \
    --pretrained_model_name_or_path "$PROJECT_ROOT/checkpoint/EditMGT" \
    --transformer_subfolder editmgt \
    --sparse_target_halo 0 \
    --no-supervise_target_halo \
    --time_conditioning mask_ratio \
    --mask_loss_reduction sample
