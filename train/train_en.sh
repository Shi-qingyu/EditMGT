#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep the mixed-language run untouched. "latest" only searches this new output
# directory, so a fresh English run starts from the released editing weights.
export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-./runs/editmgt_sparse_inter_edit_en}"
export INTER_EDIT_LANGUAGE=en
export SPARSE_TARGET_HALO="${SPARSE_TARGET_HALO:-1}"
exec bash "$PROJECT_ROOT/train/train.sh" "$@"
