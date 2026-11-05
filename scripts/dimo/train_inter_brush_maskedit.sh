#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Match train_en_causal.sh: Inter-Edit 90%, MagicBrush 5%, MaskEdit 5%,
# translated English prompts and the same training-only text-edit filter.
# The preset pins checkpoint-8000 and starts a separate DiMO v3 run at step 0.
export DIMO_CONFIG="${DIMO_CONFIG:-$PROJECT_ROOT/configs/dimo/region_dimo_inter_brush_maskedit.json}"
export DIMO_PYTHON="${DIMO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == --check-only ]]; then
    shift
    exec "$DIMO_PYTHON" -m src.dimo.preflight --config "$DIMO_CONFIG" "$@"
fi

exec bash "$PROJECT_ROOT/scripts/dimo/run_formal.sh" "$@"
