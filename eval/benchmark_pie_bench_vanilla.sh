#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

NUM_GPUS="${NUM_GPUS:-8}"
PIE_BENCH_DATA_DIR="${PIE_BENCH_DATA_DIR:-$PROJECT_ROOT/data/PIE-Bench}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/pie_bench_vanilla}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec .venv/bin/python -m eval.benchmark_pie_bench_vanilla \
    --num-gpus "$NUM_GPUS" --data-dir "$PIE_BENCH_DATA_DIR" --output-dir "$OUTPUT_DIR" "$@"
