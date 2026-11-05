#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SUITE_LOG="$ROOT/runs/sft_arch_ablation_2000/logs/suite.log"
EVAL_ROOT="$ROOT/outputs/sft_arch_ablation_2000"
RUN="$ROOT/runs/region_dimo_v3_inter_brush_maskedit_bs256"
exec 9>"$RUN/.resume_after_ablations.lock"
flock -n 9 || { printf 'DiMO continuation is already queued.\n'; exit 1; }
printf '%s\n' "$$" >"$RUN/logs/resume_after_ablations.pid"

variants=(mask_control no_micro_conditions bbox_only no_token_type_rope no_condition_isolation)
benchmarks=(magicbrush inter_edit pie_bench)
printf 'Waiting for all 15 SFT ablation benchmark summaries and suite completion.\n'
while true; do
    ready=0
    for variant in "${variants[@]}"; do
        for benchmark in "${benchmarks[@]}"; do
            [[ -s "$EVAL_ROOT/$variant/$benchmark/metrics/summary.json" ]] && ((ready+=1)) || true
        done
    done
    if (( ready == 15 )) && [[ -f "$SUITE_LOG" ]] && \
        grep -Fq 'All SFT ablation training and benchmark runs completed.' "$SUITE_LOG"; then
        break
    fi
    printf '%s: %s/15 benchmark summaries ready.\n' "$(date -Is)" "$ready"
    sleep 60
done

[[ -f "$RUN/checkpoint-2000/COMPLETE" ]] || {
    printf 'Missing complete DiMO checkpoint-2000.\n' >&2; exit 1;
}
for checkpoint in "$RUN"/checkpoint-*; do
    [[ -d "$checkpoint" && -f "$checkpoint/COMPLETE" ]] || continue
    step="${checkpoint##*-}"
    if (( step > 2000 )); then
        printf 'Newer DiMO checkpoint already exists: %s; refusing an ambiguous resume.\n' "$checkpoint" >&2
        exit 1
    fi
done

export DIMO_CONFIG="$ROOT/configs/dimo/region_dimo_inter_brush_maskedit_bs256.json"
export GPU_IDS="0,1,2,3,4,5,6,7"
export NUM_PROCESSES=8
export MAIN_PROCESS_PORT=25013
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
printf '%s: Ablation benchmarks complete; resuming DiMO from checkpoint-2000.\n' "$(date -Is)"
exec bash "$ROOT/scripts/dimo/train_inter_brush_maskedit.sh" --resume latest
