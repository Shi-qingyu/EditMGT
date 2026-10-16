#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use a separate run so automatic resume cannot select an English checkpoint
# trained with fully bidirectional, timestep-dependent context branches.
export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-./runs/editmgt_sparse_inter_edit_en_cache}"
export CACHE_CONDITIONING=1
export CONDITION_CACHE_THRESHOLD="${CONDITION_CACHE_THRESHOLD:-0.3}"
exec bash "$PROJECT_ROOT/train/train_en.sh" "$@"
