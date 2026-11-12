#!/usr/bin/env bash
# All five models, eight seeds each, one GPU by default:
#   bash third_party/table_completion/run_cherry_pick.sh
# Select GPUs, seeds, output directory and optionally a subset of models:
#   GPU_IDS=0,1,2,3 SEEDS="42 43 44 45" OUTPUT_DIR=outputs/qualitative_comparison_v2 \
#     bash third_party/table_completion/run_cherry_pick.sh spotedit lumina_dimoo gide dreamlite swiftedit
# Optional environment overrides:
#   PREPARE_PYTHON, SPOTEDIT_PYTHON, LUMINA_PYTHON, GIDE_PYTHON,
#   DREAMLITE_PYTHON, SWIFTEDIT_PYTHON, CHERRY_PICK_DIR, CASE_CONFIG, TRAIN_MANIFEST.
# Repeating the same command resumes successful model/case/seed outputs.
# To change the selected models, prompts or seeds, choose a new OUTPUT_DIR.
set -euo pipefail

CHERRY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CHERRY_ROOT"
CHERRY_SCRIPT="$CHERRY_ROOT/third_party/table_completion/run_cherry_pick.py"
CHERRY_OUTPUT="${OUTPUT_DIR:-$CHERRY_ROOT/outputs/qualitative_comparison}"
CHERRY_PREPARE_PYTHON="${PREPARE_PYTHON:-$CHERRY_ROOT/.venv/bin/python}"
CHERRY_GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
CHERRY_SEEDS="${SEEDS:-42 43 44 45 46 47 48 49}"
IFS=',' read -r -a cherry_gpus <<< "$CHERRY_GPU_IDS"
read -r -a cherry_seeds <<< "${CHERRY_SEEDS//,/ }"
if (( ${#cherry_gpus[@]} == 0 || ${#cherry_seeds[@]} == 0 )); then
    echo 'GPU_IDS and SEEDS must be nonempty' >&2
    exit 2
fi
for cherry_gpu in "${cherry_gpus[@]}"; do
    if [[ -z "$cherry_gpu" || "$cherry_gpu" == -1 ]]; then
        echo 'GPU_IDS must contain usable CUDA device IDs or UUIDs' >&2
        exit 2
    fi
done
if (( $# == 0 )); then
    set -- spotedit lumina_dimoo gide dreamlite swiftedit
fi
cherry_models=()
for requested_model in "$@"; do
    case "${requested_model,,}" in
        spotedit|spotedit_flux2_4b) cherry_models+=(spotedit) ;;
        lumina_dimoo|lumina-dimoo|lumina-dimmo) cherry_models+=(lumina_dimoo) ;;
        gide|gide_pie) cherry_models+=(gide) ;;
        dreamlite|dreamlite_mobile) cherry_models+=(dreamlite) ;;
        swiftedit) cherry_models+=(swiftedit) ;;
        *) echo "Unsupported model: $requested_model" >&2; exit 2 ;;
    esac
done

export PYTHONPATH="$CHERRY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$CHERRY_PREPARE_PYTHON" "$CHERRY_SCRIPT" --stage prepare --resume \
    --cherry-pick "${CHERRY_PICK_DIR:-$CHERRY_ROOT/docs/cherry_pick}" \
    --case-config "${CASE_CONFIG:-$CHERRY_ROOT/third_party/table_completion/cherry_pick_cases.json}" \
    --train-manifest "${TRAIN_MANIFEST:-$CHERRY_ROOT/outputs/inter_edit_train/sft_checkpoint_11000_large/manifest.json}" \
    --models "${cherry_models[@]}" --seeds "${cherry_seeds[@]}" \
    --output-dir "$CHERRY_OUTPUT"

cherry_pids=()
stop_workers() {
    for cherry_pid in "${cherry_pids[@]}"; do
        kill "$cherry_pid" 2>/dev/null || true
    done
    for cherry_pid in "${cherry_pids[@]}"; do
        wait "$cherry_pid" 2>/dev/null || true
    done
}
trap 'stop_workers; exit 130' INT
trap 'stop_workers; exit 143' TERM
cherry_failed=0
for cherry_model in "${cherry_models[@]}"; do
    case "$cherry_model" in
        spotedit) cherry_python="${SPOTEDIT_PYTHON:-$CHERRY_ROOT/third_party/table_completion/envs/flux2/bin/python}" ;;
        lumina_dimoo) cherry_python="${LUMINA_PYTHON:-$CHERRY_ROOT/.venv/bin/python}" ;;
        gide) cherry_python="${GIDE_PYTHON:-$CHERRY_ROOT/.venv/bin/python}" ;;
        dreamlite) cherry_python="${DREAMLITE_PYTHON:-$CHERRY_ROOT/third_party/table_completion/envs/dreamlite/bin/python}" ;;
        swiftedit) cherry_python="${SWIFTEDIT_PYTHON:-$CHERRY_ROOT/third_party/table_completion/envs/swift/bin/python}" ;;
    esac
    if [[ ! -x "$cherry_python" ]]; then
        echo "Missing $cherry_model interpreter: $cherry_python; set its *_PYTHON override" >&2
        cherry_failed=1
        continue
    fi
    mkdir -p "$CHERRY_OUTPUT/logs/$cherry_model"
    echo "Starting $cherry_model on GPUs $CHERRY_GPU_IDS; logs: $CHERRY_OUTPUT/logs/$cherry_model"
    cherry_pids=()
    for cherry_rank in "${!cherry_gpus[@]}"; do
        CUDA_VISIBLE_DEVICES="${cherry_gpus[$cherry_rank]}" \
            "$cherry_python" -u "$CHERRY_SCRIPT" --stage generate --resume \
            --model "$cherry_model" --rank "$cherry_rank" \
            --world-size "${#cherry_gpus[@]}" --output-dir "$CHERRY_OUTPUT" \
            >> "$CHERRY_OUTPUT/logs/$cherry_model/rank_${cherry_rank}.log" 2>&1 &
        cherry_pids+=("$!")
    done
    for cherry_pid in "${cherry_pids[@]}"; do
        if ! wait "$cherry_pid"; then
            cherry_failed=1
            echo "$cherry_model worker failed; inspect $CHERRY_OUTPUT/logs/$cherry_model" >&2
        fi
    done
    cherry_pids=()
done

if ! "$CHERRY_PREPARE_PYTHON" "$CHERRY_SCRIPT" --stage summarize --output-dir "$CHERRY_OUTPUT"; then
    cherry_failed=1
fi
echo "Results: $CHERRY_OUTPUT (index.csv, summary.json, cases/, logs/)"
exit "$cherry_failed"
