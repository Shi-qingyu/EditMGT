#!/usr/bin/env bash
set -euo pipefail

# Inter-Edit test/en: 36 decoding forwards, snapshots at 6/12/18/24/30/36.
# Uncommitted token cells are shown in solid gray.
#
#   bash scripts/visualize_token_prediction_order.sh
#   CUDA_VISIBLE_DEVICES=2 bash scripts/visualize_token_prediction_order.sh
#   bash scripts/visualize_token_prediction_order.sh --sample-keys sample_00997 sample_03770
#   bash scripts/visualize_token_prediction_order.sh --stage render --visual-sample sample_00997
#
# Optional overrides: MOTIVATION_PYTHON, INTER_EDIT_ROOT,
# MOTIVATION_NUM_SAMPLES, MOTIVATION_OUTPUT_DIR.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
MOTIVATION_PYTHON="${MOTIVATION_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

exec "$MOTIVATION_PYTHON" "$PROJECT_ROOT/scripts/plot_mgm_motivation.py" \
    --data-dir "${INTER_EDIT_ROOT:-/mnt/geminihzceph/public/data/Inter-Edit}" \
    --language en \
    --num-inference-steps 36 \
    --snapshot-every 6 \
    --num-samples "${MOTIVATION_NUM_SAMPLES:-8}" \
    --output-dir "${MOTIVATION_OUTPUT_DIR:-$PROJECT_ROOT/docs/motivation_token_order}" \
    "$@"
