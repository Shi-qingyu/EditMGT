#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUTPUT="$ROOT/outputs/dimo_step4000_sft8000_comparison"
SFT="$ROOT/runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-8000"
DIMO="$ROOT/runs/region_dimo_v3_inter_brush_maskedit_bs256/export-ema-step4000"
BASE="$ROOT/checkpoint/EditMGT"
[[ -f "$SFT/transformer/config.json" && -f "$DIMO/export.json" ]] || {
    printf 'Missing checkpoint or EMA export.\n' >&2; exit 1;
}
[[ $(find data/magicbrush/data -maxdepth 1 -name 'test-*.parquet' | wc -l) -eq 6 ]] || {
    printf 'MagicBrush test parquet shards are incomplete.\n' >&2; exit 1;
}
mkdir -p "$OUTPUT/logs"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline

# The first runner occupies all eight GPUs. Start this runner now and let it
# wait, so the newly added test split runs automatically after it finishes.
if tmux has-session -t dimo_step4000_eval 2>/dev/null; then
    printf '%s: waiting for dev/Inter-Edit/PIE runner to release GPUs\n' "$(date -Is)"
    while tmux has-session -t dimo_step4000_eval 2>/dev/null; do sleep 15; done
fi

for method in sft18 ema1 ema2 ema4; do
    result="$OUTPUT/$method/magicbrush_test"
    if [[ -f "$result/metrics/summary.json" ]] && \
        "$ROOT/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("complete") and s.get("failed_predictions")==0 else 1)' \
            "$result/metrics/summary.json"; then
        printf 'Already complete: %s MagicBrush test\n' "$method"
        continue
    fi
    common=(--benchmark magicbrush --split test --mode both --metrics l1,l2,clip-i,dino
        --output-dir "$result" --model-path "$BASE" --num-gpus 8 --bf16
        --seed 42 --warmup 1 --resume)
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
    printf '%s: evaluating %s on MagicBrush test\n' "$(date -Is)" "$method"
    "$ROOT/.venv/bin/python" -m src.dimo.benchmark "${common[@]}" \
        >"$OUTPUT/logs/$method.magicbrush_test.log" 2>&1
    printf '%s: complete %s on MagicBrush test\n' "$(date -Is)" "$method"
done

"$ROOT/.venv/bin/python" -m scripts.dimo.summarize_step4000 \
    --output-root "$OUTPUT" --include-magicbrush-test
printf '%s: MagicBrush test and final report complete.\n' "$(date -Is)"
