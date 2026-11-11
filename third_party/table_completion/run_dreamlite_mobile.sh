#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
WORLD_SIZE="${#GPUS[@]}"
if (( WORLD_SIZE < 1 )); then
    echo 'GPU_IDS must contain at least one GPU' >&2
    exit 2
fi

PYTHON="$ROOT/third_party/table_completion/envs/dreamlite/bin/python"
METRIC_PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/third_party/table_completion/run_baseline.py"
OUTPUT="$ROOT/outputs/real_time_baselines/dreamlite_mobile"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

BENCHMARKS=("${@:-magicbrush inter_edit pie_bench}")
if (( $# == 0 )); then BENCHMARKS=(magicbrush inter_edit pie_bench); fi
for benchmark in "${BENCHMARKS[@]}"; do
    case "$benchmark" in
        magicbrush|inter_edit|pie_bench) ;;
        *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;;
    esac
    run_dir="$OUTPUT/$benchmark"
    "$PYTHON" "$SCRIPT" --model dreamlite_mobile --benchmark "$benchmark" --stage prepare --resume
    mkdir -p "$run_dir/logs"
    pids=()
    for rank in "${!GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "$PYTHON" "$SCRIPT" \
            --model dreamlite_mobile --benchmark "$benchmark" --stage generate \
            --rank "$rank" --world-size "$WORLD_SIZE" --resume \
            > "$run_dir/logs/generate_rank${rank}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    "$PYTHON" "$SCRIPT" --model dreamlite_mobile --benchmark "$benchmark" --stage finalize \
        > "$run_dir/logs/finalize.log" 2>&1
    case "$benchmark" in
        magicbrush)
            CUDA_VISIBLE_DEVICES="$GPU_IDS" "$METRIC_PYTHON" -m eval.benchmark_magicbrush \
                --split test --mode both --stage metrics --num-gpus "$WORLD_SIZE" \
                --bf16 --output-dir "$run_dir" > "$run_dir/logs/metrics.log" 2>&1 ;;
        inter_edit)
            CUDA_VISIBLE_DEVICES="$GPU_IDS" "$METRIC_PYTHON" -m eval.benchmark_inter_edit \
                --language en --stage metrics --num-gpus "$WORLD_SIZE" --bf16 \
                --tar-cache-dir "$ROOT/data/inter_edit_tar_index" \
                --alpha-clip-ckpt "$ROOT/checkpoint/AlphaCLIP/clip_l14_grit1m_fultune_8xe.pth" \
                --output-dir "$run_dir" > "$run_dir/logs/metrics.log" 2>&1 ;;
        pie_bench)
            CUDA_VISIBLE_DEVICES="$GPU_IDS" "$METRIC_PYTHON" -m eval.benchmark_pie_bench \
                --stage metrics --num-gpus "$WORLD_SIZE" --bf16 --report-limit 0 \
                --output-dir "$run_dir" > "$run_dir/logs/metrics.log" 2>&1 ;;
    esac
    "$PYTHON" "$SCRIPT" --model dreamlite_mobile --benchmark "$benchmark" --stage annotate
    echo "Finished $benchmark: $run_dir"
done
