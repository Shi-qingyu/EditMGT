#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$PYTHON_BIN" -m eval.benchmark_magicbrush --num-gpus "$NUM_GPUS" "$@"
