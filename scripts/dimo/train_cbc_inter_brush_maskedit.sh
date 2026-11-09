#!/usr/bin/env bash
set -euo pipefail

# Start Core--Boundary--Context (CBC) region-aware distillation:
#   bash scripts/dimo/train_cbc_inter_brush_maskedit.sh
#
# Choose visible GPUs (one training worker per visible GPU by default):
#   GPU_IDS=0,1,2,3 bash scripts/dimo/train_cbc_inter_brush_maskedit.sh
#
# Continue this CBC run from its latest complete training checkpoint:
#   bash scripts/dimo/train_cbc_inter_brush_maskedit.sh --resume latest
#
# Optional environment overrides:
#   DIMO_PYTHON       Python interpreter with the training dependencies installed
#   DIMO_CONFIG       Alternative complete JSON training configuration
#   DIMO_TEACHER_DIR  SFT Teacher checkpoint-N or its transformer/ directory
#   DIMO_OUTPUT_DIR   New experiment directory (use the same path when resuming)
#   BASE_MODEL_DIR   Base text encoders, VQ model and scheduler directory
#   GPU_IDS          Comma-separated GPU indices/UUIDs; otherwise respect CUDA_VISIBLE_DEVICES
#   NUM_PROCESSES    Worker count; defaults to the number of visible GPUs
#   MAIN_PROCESS_PORT Distributed launch port; defaults to 25012
#
# Default JSON: SFT-8000 Teacher, BF16, 1024px, micro-batch 1 per GPU,
# accumulation 8, 10000 Student updates, CBC FKL plus weighted pseudo-mask,
# validation previews every 100 Student updates, and a
# 90% Inter-Edit / 5% MagicBrush / 5% MaskEdit training mixture.
# Effective Student batch = NUM_PROCESSES * micro_batch * gradient_accumulation.
# Output: runs/region_dimo_cbc_inter_brush_maskedit.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Core--Boundary--Context Region-DiMO. This is a new experiment identity and
# never resumes or overwrites the historical v3 run.
export DIMO_CONFIG="${DIMO_CONFIG:-$PROJECT_ROOT/configs/dimo/region_dimo_cbc_inter_brush_maskedit.json}"
export DIMO_PYTHON="${DIMO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${1:-}" == --check-only ]]; then
    shift
    exec "$DIMO_PYTHON" -m src.dimo.preflight --config "$DIMO_CONFIG" "$@"
fi

exec bash "$PROJECT_ROOT/scripts/dimo/run_formal.sh" "$@"
