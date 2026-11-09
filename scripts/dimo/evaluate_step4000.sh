#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUTPUT="$ROOT/outputs/dimo_step4000_sft8000_comparison"
SFT="$ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-8000"
DIMO_RUN="$ROOT/runs/region_dimo_v3_inter_brush_maskedit_bs256"
DIMO="$DIMO_RUN/export-ema-step4000"
BASE="$ROOT/checkpoint/EditMGT"
ALPHA="$ROOT/checkpoint/AlphaCLIP/clip_l14_grit1m_fultune_8xe.pth"

[[ -f "$SFT/transformer/config.json" && -f "$DIMO_RUN/checkpoint-4000/COMPLETE" && -f "$DIMO/export.json" ]] || {
    printf 'Missing a requested complete checkpoint or EMA export.\n' >&2; exit 1;
}
mkdir -p "$OUTPUT/logs"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline

for method in sft18 ema1 ema2 ema4; do
    for benchmark in magicbrush inter_edit pie_bench; do
        result="$OUTPUT/$method/$benchmark"
        if [[ -f "$result/metrics/summary.json" ]] && \
            "$ROOT/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("complete") and s.get("failed_predictions")==0 else 1)' \
                "$result/metrics/summary.json"; then
            printf 'Already complete: %s %s\n' "$method" "$benchmark"
            continue
        fi
        common=(--benchmark "$benchmark" --output-dir "$result" --model-path "$BASE"
            --num-gpus 8 --bf16 --seed 42 --warmup 1 --resume)
        if [[ "$method" == sft18 ]]; then
            common+=(--backend teacher --checkpoint "$SFT" --num-inference-steps 18
                     --guidance-scale 6 --condition-cache-threshold 0.8
                     --reference-kv-cache-threshold 0.8)
        else
            steps="${method#ema}"
            common+=(--backend ema --checkpoint "$DIMO" --num-inference-steps "$steps"
                     --guidance-scale 1)
            if (( steps > 1 )); then
                common+=(--cache-first-step)
            else
                common+=(--disable-reference-kv-cache --disable-condition-kv-cache)
            fi
        fi
        case "$benchmark" in
            magicbrush) common+=(--split dev --mode both --metrics l1,l2,clip-i,dino) ;;
            inter_edit) common+=(--language en --alpha-clip-ckpt "$ALPHA") ;;
            pie_bench) common+=(--report-limit 0) ;;
        esac
        printf '%s: evaluating %s on %s\n' "$(date -Is)" "$method" "$benchmark"
        "$ROOT/.venv/bin/python" -m src.dimo.benchmark "${common[@]}" \
            >"$OUTPUT/logs/$method.$benchmark.log" 2>&1
        printf '%s: complete %s on %s\n' "$(date -Is)" "$method" "$benchmark"
    done
done

"$ROOT/.venv/bin/python" -m scripts.dimo.visualize_inter_edit_train sft \
    >"$OUTPUT/logs/visual_sft.log" 2>&1
"$ROOT/.venv/bin/python" -m scripts.dimo.visualize_inter_edit_train dimo \
    >"$OUTPUT/logs/visual_dimo.log" 2>&1
"$ROOT/.venv/bin/python" -m scripts.dimo.summarize_step4000 \
    --output-root "$OUTPUT"
printf '%s: all benchmark and visualization results complete.\n' "$(date -Is)"
