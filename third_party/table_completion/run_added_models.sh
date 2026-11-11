#!/usr/bin/env bash
# Serial evaluation after the pre-existing SpotEdit/Sparse-LaViDa queue.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUTPUT="$ROOT/outputs/real_time_baselines"
SCRIPT="$ROOT/third_party/table_completion/run_baseline.py"
PYTHON="$ROOT/.venv/bin/python"
BASE_PYTHONPATH="${PYTHONPATH:-}"

while [[ ! -e "$OUTPUT/ALL_COMPLETE" ]]; do
    sleep 60
done

failed=0
for model in ultraedit nep_512 lumina_dimoo mmada_parallel_a; do
    echo "Starting $model smoke at $(date -Is)"
    if [[ "$model" == ultraedit ]]; then
        export PYTHONPATH="$ROOT/third_party/UltraEdit/diffusers/src:$ROOT${BASE_PYTHONPATH:+:$BASE_PYTHONPATH}"
    else
        export PYTHONPATH="$ROOT${BASE_PYTHONPATH:+:$BASE_PYTHONPATH}"
    fi
    smoke_dir="$OUTPUT/$model/magicbrush_smoke1"
    mkdir -p "$smoke_dir/logs"
    "$PYTHON" "$SCRIPT" --model "$model" --benchmark magicbrush --stage prepare --limit 1 --resume
    if ! CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        "$PYTHON" "$SCRIPT" --model "$model" --benchmark magicbrush \
        --stage generate --limit 1 --resume > "$smoke_dir/logs/generate.log" 2>&1; then
        echo "$model smoke FAILED: $smoke_dir/logs/generate.log" >&2
        failed=$((failed + 1))
        continue
    fi
    echo "Starting $model full benchmark at $(date -Is)"
    if ! GPU_IDS=0,1,2,3,4,5,6,7 bash third_party/table_completion/run_model.sh "$model"; then
        echo "$model full benchmark FAILED; inspect $OUTPUT/$model" >&2
        failed=$((failed + 1))
        continue
    fi
    "$PYTHON" third_party/table_completion/check_completed.py "$model"
    "$PYTHON" third_party/table_completion/update_realtime_tables.py
    "$PYTHON" third_party/table_completion/validate_tables.py
    echo "$model complete at $(date -Is)"
done

if (( failed )); then
    echo "$failed added models failed smoke or full evaluation" >&2
    exit 1
fi
touch "$OUTPUT/ADDED_COMPLETE"
echo "All added models complete."
