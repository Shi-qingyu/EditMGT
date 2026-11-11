#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
while tmux has-session -t ultra_retry 2>/dev/null; do
    sleep 60
done
PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/third_party/table_completion/run_baseline.py"
SMOKE="$ROOT/outputs/real_time_baselines/dice_paella/pie_bench_smoke1"
mkdir -p "$SMOKE/logs"
"$PYTHON" "$SCRIPT" --model dice_paella --benchmark pie_bench --stage prepare --limit 1 --resume
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --model dice_paella --benchmark pie_bench \
    --stage generate --limit 1 --resume > "$SMOKE/logs/generate.log" 2>&1
GPU_IDS=0,1,2,3,4,5,6,7 bash third_party/table_completion/run_model.sh dice_paella pie_bench
"$PYTHON" third_party/table_completion/update_dice_pie.py
"$PYTHON" third_party/table_completion/validate_tables.py
touch "$ROOT/outputs/real_time_baselines/DICE_PIE_COMPLETE"
