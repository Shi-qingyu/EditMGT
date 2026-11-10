#!/usr/bin/env bash
set -euo pipefail

# One command runs and scores dense -> sparse -> sparse+cache -> distilled EMA
# on the SAME Inter-Edit test/en and PIE-Bench samples, then writes matched-key
# quality/latency tables and natural mask-area bins.
#
#   bash scripts/ablations/run_efficiency_ablation.sh
#   GPU_IDS=0,1,2,3 NUM_GPUS=4 bash scripts/ablations/run_efficiency_ablation.sh
#   DISTILL_CHECKPOINT=/path/to/checkpoint-4000 bash scripts/ablations/run_efficiency_ablation.sh
#
# Defaults: first three rows share SFT weights, 18 steps and CFG=1. Distill uses
# one step, CFG=1. Set TEACHER_CFG=6 for the operational teacher protocol; actual
# steps/CFG are always reported. HALO=1 and its supervision are shared by all.
# Dense means full target computation with the SAME mask/source hard lock and
# ROI conditioning, not the released mask-free vanilla model.
# MASK_AREA_BINS partition existing annotations; annotations/GT are not mutated.
# LIMIT=100 limits EACH benchmark identically across all four variants.
# Rerunning resumes the same settings. Choose a new ABLATION_OUTPUT_DIR if any
# model/sampling setting changes. No package installation is performed.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
ABLATION_PYTHON="${ABLATION_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$PROJECT_ROOT/checkpoint/EditMGT}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-$PROJECT_ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-8000}"
DISTILL_CHECKPOINT="${DISTILL_CHECKPOINT:-$PROJECT_ROOT/runs/region_dimo_cbc_inter_brush_maskedit/checkpoint-4000}"
ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-$PROJECT_ROOT/outputs/efficiency_ablation_cbc}"
INTER_EDIT_ROOT="${INTER_EDIT_ROOT:-/mnt/geminihzceph/public/data/Inter-Edit}"
PIE_BENCH_ROOT="${PIE_BENCH_ROOT:-$PROJECT_ROOT/data/PIE-Bench}"
ALPHA_CLIP_CKPT="${ALPHA_CLIP_CKPT:-$PROJECT_ROOT/checkpoint/AlphaCLIP/clip_l14_grit1m_fultune_8xe.pth}"
RESOLUTION="${RESOLUTION:-1024}"
TEACHER_STEPS="${TEACHER_STEPS:-18}"
TEACHER_CFG="${TEACHER_CFG:-1}"
CACHE_THRESHOLD="${CACHE_THRESHOLD:-0.3}"
HALO="${HALO:-1}"
SEED="${SEED:-42}"
WARMUP="${WARMUP:-1}"
MASK_AREA_BINS="${MASK_AREA_BINS:-0,0.05,0.10,0.20,0.40,0.60,0.80,1}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
if [[ -n "${GPU_IDS+x}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
# Device discovery happens only when this launcher is actually started.
NUM_GPUS="${NUM_GPUS:-$("$ABLATION_PYTHON" -c 'import torch; print(torch.cuda.device_count())')}"
mkdir -p "$ABLATION_OUTPUT_DIR/logs"
benchmarks=(inter_edit pie_bench)
if [[ -n "${BENCHMARKS:-}" ]]; then
    read -r -a benchmarks <<< "$BENCHMARKS"
fi

for benchmark in "${benchmarks[@]}"; do
    for variant in dense sparse sparse_cache distill; do
        checkpoint="$SFT_CHECKPOINT"
        steps="$TEACHER_STEPS"
        cfg="$TEACHER_CFG"
        if [[ "$variant" == distill ]]; then
            checkpoint="$DISTILL_CHECKPOINT"
            steps=1
            cfg=1
        fi
        common=(--benchmark "$benchmark" --variant "$variant" --checkpoint "$checkpoint"
            --model-path "$BASE_MODEL_DIR" --output-dir "$ABLATION_OUTPUT_DIR/$variant/$benchmark"
            --num-gpus "$NUM_GPUS" --resolution "$RESOLUTION" --bf16
            --num-inference-steps "$steps" --guidance-scale "$cfg"
            --reference-strength 1 --negative-prompt '' --seed "$SEED" --warmup "$WARMUP"
            --mask-mode provided --sparse-target-halo "$HALO" --no-preserve-unmasked-pixels
            --condition-cache-threshold "$CACHE_THRESHOLD" --reference-kv-cache-threshold "$CACHE_THRESHOLD"
            --resume)
        if [[ "$variant" != sparse_cache ]]; then
            common+=(--disable-reference-kv-cache --disable-condition-kv-cache)
        fi
        if [[ -n "${LIMIT:-}" ]]; then
            common+=(--limit "$LIMIT")
        fi
        if [[ -n "${MODEL_CACHE_DIR:-}" ]]; then
            common+=(--model-cache-dir "$MODEL_CACHE_DIR")
        fi
        case "$benchmark" in
            inter_edit)
                common+=(--data-dir "$INTER_EDIT_ROOT" --language en
                    --tar-cache-dir "$PROJECT_ROOT/data/inter_edit_tar_index"
                    --metrics "${INTER_EDIT_METRICS:-lpips,alpha-in,alpha-out,boundary}"
                    --alpha-clip-ckpt "$ALPHA_CLIP_CKPT")
                ;;
            pie_bench)
                common+=(--data-dir "$PIE_BENCH_ROOT" --report-limit 0)
                if [[ -n "${PIE_METRICS:-}" ]]; then common+=(--metrics "$PIE_METRICS"); fi
                if [[ -n "${DINO_REPO:-}" ]]; then common+=(--dino-repo "$DINO_REPO"); fi
                if [[ -n "${DINO_WEIGHTS:-}" ]]; then common+=(--dino-weights "$DINO_WEIGHTS"); fi
                ;;
            *) printf 'Unsupported benchmark: %s\n' "$benchmark" >&2; exit 2 ;;
        esac
        printf 'Efficiency ablation: %s / %s, steps=%s, CFG=%s\n' "$benchmark" "$variant" "$steps" "$cfg"
        "$ABLATION_PYTHON" -m scripts.ablations.benchmark_efficiency "${common[@]}" \
            2>&1 | tee "$ABLATION_OUTPUT_DIR/logs/$benchmark.$variant.log"
    done
done

"$ABLATION_PYTHON" -m scripts.ablations.summarize_efficiency_ablation \
    --output-root "$ABLATION_OUTPUT_DIR" --benchmarks "${benchmarks[@]}" \
    --mask-area-bins "$MASK_AREA_BINS"
printf 'Saved report: %s/summary.md\n' "$ABLATION_OUTPUT_DIR"
