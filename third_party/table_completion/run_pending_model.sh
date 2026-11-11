#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODEL="${1:?Usage: run_model.sh MODEL [magicbrush|inter_edit|pie_bench ...]}"
shift
case "$MODEL" in
    dreamlite_mobile) PYTHON="$ROOT/third_party/table_completion/envs/dreamlite/bin/python" ;;
    chordedit) PYTHON="$ROOT/.venv/bin/python" ;;
    flux2_klein) PYTHON="$ROOT/third_party/table_completion/envs/flux2/bin/python" ;;
    turboedit) PYTHON="$ROOT/third_party/table_completion/envs/turbo/bin/python" ;;
    swiftedit) PYTHON="$ROOT/third_party/table_completion/envs/swift/bin/python" ;;
    sparse_lavida) PYTHON="$ROOT/third_party/table_completion/envs/sparse/bin/python" ;;
    spotedit_flux2_4b) PYTHON="$ROOT/third_party/table_completion/envs/flux2/bin/python" ;;
    *) echo "Unsupported model: $MODEL" >&2; exit 2 ;;
esac
METRIC_PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/third_party/table_completion/run_baseline.py"
OUTPUT="$ROOT/outputs/real_time_baselines/$MODEL${RUN_TAG:+_$RUN_TAG}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
WORLD_SIZE="${#GPUS[@]}"
if (( WORLD_SIZE < 1 )); then echo 'Empty GPU_IDS' >&2; exit 2; fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if (( $# == 0 )); then set -- magicbrush inter_edit pie_bench; fi
for benchmark in "$@"; do
    case "$benchmark" in
        magicbrush|inter_edit|pie_bench) ;;
        *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;;
    esac
    run_dir="$OUTPUT/$benchmark"
    "$PYTHON" "$SCRIPT" --model "$MODEL" --benchmark "$benchmark" --stage prepare --output-dir "$run_dir" --resume
    mkdir -p "$run_dir/logs"
    pids=()
    for rank in "${!GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES="${GPUS[$rank]}" "$PYTHON" "$SCRIPT" \
            --model "$MODEL" --benchmark "$benchmark" --stage generate \
            --rank "$rank" --world-size "$WORLD_SIZE" --output-dir "$run_dir" --resume \
            > "$run_dir/logs/generate_rank${rank}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if (( failed )); then
        echo "Some $MODEL $benchmark workers failed; inspect $run_dir/logs" >&2
        exit 1
    fi
    "$PYTHON" "$SCRIPT" --model "$MODEL" --benchmark "$benchmark" --stage finalize --output-dir "$run_dir" \
        > "$run_dir/logs/finalize.log" 2>&1
    case "$benchmark" in
        magicbrush)
            CUDA_VISIBLE_DEVICES="$GPU_IDS" "$METRIC_PYTHON" -m eval.benchmark_magicbrush \
                --split test --mode both --stage metrics --num-gpus "$WORLD_SIZE" --bf16 \
                --output-dir "$run_dir" > "$run_dir/logs/metrics.log" 2>&1 ;;
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
    "$PYTHON" "$SCRIPT" --model "$MODEL" --benchmark "$benchmark" --stage annotate --output-dir "$run_dir"
    echo "Finished $MODEL $benchmark: $run_dir"
done
