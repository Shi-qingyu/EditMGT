#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# New time units and loss weighting need a separate run. Existing entry points
# keep their original sampler and remain suitable for resuming old checkpoints.
export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-./runs/editmgt_sparse_inter_edit_en_cache_magicbrush_lognormal}"
export TIMESTEP_SAMPLING="${TIMESTEP_SAMPLING:-logit_normal}"
export TIMESTEP_LOGIT_MEAN="${TIMESTEP_LOGIT_MEAN:-0.0}"
export TIMESTEP_LOGIT_STD="${TIMESTEP_LOGIT_STD:-1.0}"
export TIMESTEP_SHIFT="${TIMESTEP_SHIFT:-1.0}"
export TIMESTEP_COSINE_MIX="${TIMESTEP_COSINE_MIX:-0.2}"
export TIMESTEP_LOW_NOISE_MIX="${TIMESTEP_LOW_NOISE_MIX:-0.2}"
export TIME_CONDITIONING="${TIME_CONDITIONING:-mask_ratio}"
export MASK_LOSS_REDUCTION="${MASK_LOSS_REDUCTION:-sample}"

exec bash "$PROJECT_ROOT/train/train_en_cache_magicbrush.sh" "$@"
