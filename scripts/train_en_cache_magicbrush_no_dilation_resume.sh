#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix}"
export SPARSE_TARGET_HALO=0
export TIME_CONDITIONING=mask_ratio
export MASK_LOSS_REDUCTION=sample

# Import the full legacy training state once. Subsequent launches continue the
# corrected-mask run instead of rolling it back to the original step 2600.
if [[ -z "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_FROM_CHECKPOINT="$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation/checkpoint-2600"
    for checkpoint in "$TRAIN_OUTPUT_DIR"/checkpoint-*; do
        if [[ -d "$checkpoint" && "${checkpoint##*/}" =~ ^checkpoint-[0-9]+$ ]]; then
            RESUME_FROM_CHECKPOINT=latest
            break
        fi
    done
fi

if [[ "$RESUME_FROM_CHECKPOINT" != latest ]]; then
    for required in transformer/config.json optimizer.bin scheduler.bin; do
        if [[ ! -f "$RESUME_FROM_CHECKPOINT/$required" ]]; then
            printf 'Cannot resume: missing %s in %s.\n' "$required" "$RESUME_FROM_CHECKPOINT" >&2
            exit 1
        fi
    done
fi

exec bash "$PROJECT_ROOT/scripts/train_en_cache_magicbrush.sh" "$@" \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --no-fresh_start \
    --require_resume \
    --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" \
    --pretrained_model_name_or_path "$PROJECT_ROOT/checkpoint/EditMGT" \
    --transformer_subfolder editmgt \
    --sparse_target_halo 0 \
    --no-supervise_target_halo \
    --time_conditioning mask_ratio \
    --mask_loss_reduction sample
