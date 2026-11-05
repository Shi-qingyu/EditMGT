#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Explicit single-checkpoint / vanilla invocations retain the original runner.
# With no model override, sweep every checkpoint and retain only its metrics.
CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "$CHECKPOINT")
fi
for argument in "$@"; do
    case "$argument" in
        --checkpoint|--checkpoint=*|--vanilla|-h|--help)
            exec "$PYTHON_BIN" -m eval.benchmark_magicbrush \
                --num-gpus "$NUM_GPUS" "${CHECKPOINT_ARGS[@]}" "$@"
            ;;
    esac
done
if [[ ${#CHECKPOINT_ARGS[@]} -gt 0 ]]; then
    exec "$PYTHON_BIN" -m eval.benchmark_magicbrush \
        --num-gpus "$NUM_GPUS" "${CHECKPOINT_ARGS[@]}" "$@"
fi

# --output-dir becomes the sweep root; each checkpoint gets its own child.
# Other benchmark options (metrics, cache thresholds, GPUs, etc.) pass through.
OUTPUT_ROOT="${OUTPUT_DIR:-}"
SPLIT=dev
BENCHMARK_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir|--checkpoint-root|--split|--stage)
            if [[ $# -lt 2 ]]; then
                printf 'Missing value for %s.\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --output-dir) OUTPUT_ROOT="$2" ;;
                --checkpoint-root) CHECKPOINT_ROOT="$2" ;;
                --split) SPLIT="$2"; BENCHMARK_ARGS+=(--split "$2") ;;
                --stage)
                    if [[ "$2" != all ]]; then
                        printf 'Checkpoint sweeps require --stage all before removing images.\n' >&2
                        exit 2
                    fi
                    ;;
            esac
            shift 2
            ;;
        --output-dir=*) OUTPUT_ROOT="${1#*=}"; shift ;;
        --checkpoint-root=*) CHECKPOINT_ROOT="${1#*=}"; shift ;;
        --split=*) SPLIT="${1#*=}"; BENCHMARK_ARGS+=("$1"); shift ;;
        --stage=all) shift ;;
        --stage=*)
            printf 'Checkpoint sweeps require --stage all before removing images.\n' >&2
            exit 2
            ;;
        *) BENCHMARK_ARGS+=("$1"); shift ;;
    esac
done
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/benchmark_magicbrush_${SPLIT}}"

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    printf 'Checkpoint root does not exist: %s\n' "$CHECKPOINT_ROOT" >&2
    exit 1
fi

CHECKPOINTS=()
for checkpoint in "$CHECKPOINT_ROOT"/checkpoint-*; do
    if [[ -d "$checkpoint" && "${checkpoint##*/}" =~ ^checkpoint-[0-9]+$ ]] \
        && (( 10#${checkpoint##*-} > 7200 )); then
        CHECKPOINTS+=("$checkpoint")
    fi
done
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    printf 'No checkpoint-N directories with N > 7200 found in %s.\n' "$CHECKPOINT_ROOT" >&2
    exit 1
fi
mapfile -t CHECKPOINTS < <(printf '%s\n' "${CHECKPOINTS[@]}" | sort -V)

FAILED_CHECKPOINTS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_output="$OUTPUT_ROOT/${checkpoint##*/}"
    printf 'Evaluating %s -> %s\n' "$checkpoint" "$checkpoint_output"
    if "$PYTHON_BIN" -m eval.benchmark_magicbrush \
        --num-gpus "$NUM_GPUS" "${BENCHMARK_ARGS[@]}" \
        --checkpoint "$checkpoint" --output-dir "$checkpoint_output" --stage all; then
        # The runner exits successfully only after generation and scoring finish.
        # summary.json retains the checkpoint identity, settings and timing.
        # Delete only known benchmark artifacts, never checkpoint/model files.
        rm -rf -- "$checkpoint_output/images" "$checkpoint_output/inputs" \
            "$checkpoint_output/records" "$checkpoint_output/workers"
        rm -f -- "$checkpoint_output/run.json" "$checkpoint_output/selection.json" \
            "$checkpoint_output/generation_summary.json" "$checkpoint_output/execution.json" \
            "$checkpoint_output/parallel_execution.json" "$checkpoint_output/.benchmark.lock" \
            "$checkpoint_output/.metric_models.lock"
        printf 'Retained metrics: %s/metrics\n' "$checkpoint_output"
    else
        status=$?
        # Do not start another checkpoint after an interrupt or termination.
        if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
            exit "$status"
        fi
        FAILED_CHECKPOINTS+=("${checkpoint##*/}")
        printf 'Evaluation failed for %s; preserving images for --resume.\n' "$checkpoint" >&2
    fi
done

if [[ ${#FAILED_CHECKPOINTS[@]} -gt 0 ]]; then
    printf 'Failed checkpoints: %s\n' "${FAILED_CHECKPOINTS[*]}" >&2
    exit 1
fi
