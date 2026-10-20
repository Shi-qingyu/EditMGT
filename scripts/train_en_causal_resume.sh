#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix}"
export SPARSE_TARGET_HALO=0
export TIME_CONDITIONING=mask_ratio
export MASK_LOSS_REDUCTION=sample
export COND_DROPOUT_PROB="${COND_DROPOUT_PROB:-0.1}"
export MASK_BBOX_PROB="${MASK_BBOX_PROB:-0.1}"
# scripts/train.sh translates these into --cache_conditioning and
# --condition_cache_threshold. The threshold alone does not enable the mask.
export CACHE_CONDITIONING=1
export CONDITION_CACHE_THRESHOLD="${CONDITION_CACHE_THRESHOLD:-0.8}"
export SKIP_INSTALL="${SKIP_INSTALL:-1}"

# Resume the full corrected-mask state at step 6800. Subsequent launches prefer
# newer checkpoints in the output directory instead of rewinding to step 6800.
if [[ -z "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_FROM_CHECKPOINT="$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-6800"
    for checkpoint in "$TRAIN_OUTPUT_DIR"/checkpoint-*; do
        if [[ -d "$checkpoint" && "${checkpoint##*/}" =~ ^checkpoint-[0-9]+$ ]] \
            && (( 10#${checkpoint##*-} > 6800 )); then
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

# Use the existing English launcher directly; keep the mixed-dataset settings
# here so this entry does not depend on the removed cache/mixed wrappers.
exec bash "$PROJECT_ROOT/scripts/train_en.sh" \
    --allow_condition_cache_threshold_change_on_resume \
    --magicbrush_data_dir "${MAGICBRUSH_DATA_DIR:-$PROJECT_ROOT/data/magicbrush}" \
    --magicbrush_probability "${MAGICBRUSH_PROBABILITY:-0.2}" \
    --inter_edit_translations_dir "${INTER_EDIT_TRANSLATIONS_DIR:-$PROJECT_ROOT/data/inter_edit_zh_to_en_qwen35_9b}" \
    --maskedit_data_dir "${MASKEDIT_DATA_DIR:-$PROJECT_ROOT/data/MaskEdit-10K}" \
    --maskedit_probability "${MASKEDIT_PROBABILITY:-0.1}" \
    "$@" \
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
