#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
# Set a checkpoint directory here, e.g. "${CHECKPOINT:-$PROJECT_ROOT/runs/my_run/checkpoint-1000}".
# Leave empty to use --checkpoint /path/to/checkpoint-N or --vanilla on the command line.
CHECKPOINT="${CHECKPOINT:-runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-7200}"
CHECKPOINT_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$PYTHON_BIN" -m eval.benchmark_magicbrush --num-gpus "$NUM_GPUS" "${CHECKPOINT_ARGS[@]}" "$@"
