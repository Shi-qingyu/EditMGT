#!/usr/bin/env bash
# Re-run UltraEdit after the current serial queue finishes. The first smoke
# attempt failed during tokenizer loading because of a Protobuf version mismatch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
while tmux has-session -t added_eval 2>/dev/null; do
    sleep 60
done
export PYTHONPATH="$ROOT/third_party/UltraEdit/diffusers/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/third_party/table_completion/run_baseline.py"
SMOKE="$ROOT/outputs/real_time_baselines/ultraedit/magicbrush_smoke1"
"$PYTHON" "$SCRIPT" --model ultraedit --benchmark magicbrush --stage generate --limit 1 --resume \
    > "$SMOKE/logs/generate_retry.log" 2>&1
GPU_IDS=0,1,2,3,4,5,6,7 bash third_party/table_completion/run_model.sh ultraedit
"$PYTHON" third_party/table_completion/check_completed.py ultraedit
"$PYTHON" third_party/table_completion/update_realtime_tables.py
"$PYTHON" third_party/table_completion/validate_tables.py
touch "$ROOT/outputs/real_time_baselines/ULTRA_COMPLETE"
