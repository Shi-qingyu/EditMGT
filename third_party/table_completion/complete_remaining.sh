#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
CHECK="$ROOT/third_party/table_completion/check_completed.py"
UPDATE="$ROOT/third_party/table_completion/update_realtime_tables.py"
VALIDATE="$ROOT/third_party/table_completion/validate_tables.py"
RUN="$ROOT/third_party/table_completion/run_pending_model.sh"

while tmux has-session -t flux2_full 2>/dev/null; do
    sleep 30
done
"$PYTHON" "$CHECK" flux2_klein
"$PYTHON" "$UPDATE"
"$PYTHON" "$VALIDATE"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" bash "$RUN" spotedit_flux2_4b
"$PYTHON" "$CHECK" spotedit_flux2_4b
"$PYTHON" "$UPDATE"
"$PYTHON" "$VALIDATE"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" bash "$RUN" sparse_lavida
"$PYTHON" "$CHECK" sparse_lavida
"$PYTHON" "$UPDATE"
"$PYTHON" "$VALIDATE"
touch "$ROOT/outputs/real_time_baselines/ALL_COMPLETE"
echo "All remaining real-time benchmarks completed."
