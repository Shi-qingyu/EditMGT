#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Five independent fresh runs: a mask-only control and four one-factor ablations.
RUN_ROOT="$ROOT/runs/sft_arch_ablation_2000"
EVAL_ROOT="$ROOT/outputs/sft_arch_ablation_2000"
mkdir -p "$RUN_ROOT/logs" "$EVAL_ROOT"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export NUM_GPUS="$NUM_PROCESSES"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-25012}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export REPORT_TO=none
export MASK_BBOX_PROB=0
export CACHE_CONDITIONING=1
export CONDITION_CACHE_THRESHOLD=0.8

variants=(mask_control no_micro_conditions bbox_only no_token_type_rope no_condition_isolation)

for variant in "${variants[@]}"; do
    run="$RUN_ROOT/$variant"
    checkpoint="$run/checkpoint-2000"
    flags=()
    case "$variant" in
        mask_control) ;;
        no_micro_conditions) flags+=(--disable_micro_conditions) ;;
        bbox_only) export MASK_BBOX_PROB=1 ;;
        no_token_type_rope) flags+=(--disable_token_type_rope) ;;
        no_condition_isolation) export CACHE_CONDITIONING=0 ;;
    esac

    if [[ ! -f "$checkpoint/transformer/config.json" ]]; then
        if [[ -d "$run" && -n "$(ls -A "$run")" ]]; then
            printf 'Incomplete run already exists: %s; inspect and resume it explicitly.\n' "$run" >&2
            exit 1
        fi
        printf 'Training %s: 2000 updates, 8 GPUs, bbox_prob=%s, condition_isolation=%s\n' \
            "$variant" "$MASK_BBOX_PROB" "$CACHE_CONDITIONING" | tee -a "$RUN_ROOT/logs/suite.log"
        TRAIN_OUTPUT_DIR="$run" bash scripts/train_en_causal.sh \
            --fresh_start --max_train_steps 2000 --checkpointing_steps 2500 \
            --validation_steps 0 --report_to none --seed 42 "${flags[@]}" \
            >"$RUN_ROOT/logs/$variant.train.log" 2>&1
        if [[ ! -f "$checkpoint/transformer/config.json" ]]; then
            printf 'Training finished without expected checkpoint: %s\n' "$checkpoint" >&2
            exit 1
        fi
    fi
    export MASK_BBOX_PROB=0
    export CACHE_CONDITIONING=1
done

for variant in "${variants[@]}"; do
    checkpoint="$RUN_ROOT/$variant/checkpoint-2000"
    # All variants receive the original annotated evaluation mask. Disabling
    # both inference KV caches holds the generation protocol constant; the
    # condition-isolation model still uses its learned attention topology.
    common=(--checkpoint "$checkpoint" --num-gpus "$NUM_GPUS" \
        --num-inference-steps 18 --guidance-scale 6 --seed 42 \
        --condition-cache-threshold 0.8 --reference-kv-cache-threshold 0.8 \
        --disable-condition-kv-cache --disable-reference-kv-cache)
    for benchmark in magicbrush inter_edit pie_bench; do
        eval_dir="$EVAL_ROOT/$variant/$benchmark"
        if [[ -f "$eval_dir/metrics/summary.json" ]]; then
            continue
        fi
        mkdir -p "$EVAL_ROOT/$variant"
        eval_args=("${common[@]}" --output-dir "$eval_dir")
        if [[ -d "$eval_dir" ]]; then
            eval_args+=(--resume)
        fi
        if [[ "$benchmark" == inter_edit ]]; then
            eval_args+=(--language en --alpha-clip-ckpt \
                "$ROOT/checkpoint/AlphaCLIP/clip_l14_grit1m_fultune_8xe.pth")
        elif [[ "$benchmark" == magicbrush ]]; then
            eval_args+=(--split dev --mode both --metrics l1,l2,clip-i,dino)
        fi
        printf 'Evaluating %s on %s\n' "$variant" "$benchmark" | tee -a "$RUN_ROOT/logs/suite.log"
        .venv/bin/python -m "eval.benchmark_$benchmark" "${eval_args[@]}" \
            >"$RUN_ROOT/logs/$variant.$benchmark.log" 2>&1
    done
done

printf 'All SFT ablation training and benchmark runs completed.\n' | tee -a "$RUN_ROOT/logs/suite.log"
