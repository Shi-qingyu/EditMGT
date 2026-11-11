#!/usr/bin/env bash
set -euo pipefail

# Score the existing full-split, BF16 vanilla EditMGT predictions. This does
# not regenerate images or change the released model's inference protocol.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT="$ROOT/outputs/dimo_step4000_sft8000_comparison/vanilla36"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -n "${GPU_IDS:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_IDS"; fi
NUM_GPUS="${NUM_GPUS:-8}"

"$PYTHON" -m eval.benchmark_magicbrush_vanilla \
    --split test --mode both --stage metrics --num-gpus "$NUM_GPUS" \
    --bf16 --output-dir "$OUTPUT/magicbrush_test"

"$PYTHON" -m eval.benchmark_inter_edit_vanilla \
    --language en --stage metrics --num-gpus "$NUM_GPUS" \
    --bf16 --tar-cache-dir "$ROOT/data/inter_edit_tar_index" \
    --alpha-clip-ckpt "$ROOT/checkpoint/AlphaCLIP/clip_l14_grit1m_fultune_8xe.pth" \
    --output-dir "$OUTPUT/inter_edit"

"$PYTHON" -m eval.benchmark_pie_bench_vanilla \
    --stage metrics --num-gpus "$NUM_GPUS" --bf16 --report-limit 0 \
    --output-dir "$OUTPUT/pie_bench"
