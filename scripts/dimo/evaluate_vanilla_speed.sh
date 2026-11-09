#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUTPUT="$ROOT/outputs/dimo_step4000_sft8000_comparison"
BASE="$ROOT/checkpoint/EditMGT"
[[ -f "$BASE/editmgt/diffusion_pytorch_model.safetensors" ]] || {
    printf 'Missing released vanilla EditMGT weights.\n' >&2; exit 1;
}
mkdir -p "$OUTPUT/logs"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline

for benchmark in magicbrush magicbrush_test inter_edit pie_bench; do
    result="$OUTPUT/vanilla36/$benchmark"
    if [[ -f "$result/generation_summary.json" ]] && \
        "$ROOT/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("complete") and s.get("skipped")==0 else 1)' \
            "$result/generation_summary.json"; then
        printf 'Already complete: vanilla EditMGT %s\n' "$benchmark"
        continue
    fi
    common=(--output-dir "$result" --model-path "$BASE" --num-gpus 8
        --stage generate --num-inference-steps 36 --guidance-scale 6
        --reference-strength 1.1 --bf16 --seed 42 --warmup 1 --resume)
    case "$benchmark" in
        magicbrush) module=eval.benchmark_magicbrush_vanilla
            common+=(--split dev --mode both) ;;
        magicbrush_test) module=eval.benchmark_magicbrush_vanilla
            common+=(--split test --mode both) ;;
        inter_edit) module=eval.benchmark_inter_edit_vanilla
            common+=(--language en) ;;
        pie_bench) module=eval.benchmark_pie_bench_vanilla
            common+=(--report-limit 0) ;;
    esac
    printf '%s: measuring vanilla EditMGT on %s\n' "$(date -Is)" "$benchmark"
    "$ROOT/.venv/bin/python" -m "$module" "${common[@]}" \
        >"$OUTPUT/logs/vanilla36.$benchmark.log" 2>&1
    printf '%s: complete vanilla EditMGT on %s\n' "$(date -Is)" "$benchmark"
done

"$ROOT/.venv/bin/python" -m scripts.dimo.summarize_step4000 \
    --output-root "$OUTPUT" --include-magicbrush-test
printf '%s: vanilla speed comparison and corrected report complete.\n' "$(date -Is)"
