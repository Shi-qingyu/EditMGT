#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPORT="$ROOT/outputs/dimo_step4000_sft8000_comparison/logs/evaluate_vanilla_speed.log"
RUN="$ROOT/runs/inter_brush_maskedit_en_no_dilation_mrcache0.8_d0.1_fresh"

while tmux has-session -t editmgt_vanilla_speed 2>/dev/null; do
    sleep 15
done
if [[ ! -f "$REPORT" ]] || ! rg -Fq 'vanilla speed comparison and corrected report complete.' "$REPORT"; then
    printf 'Vanilla speed evaluation did not complete successfully; training was not started.\n' >&2
    exit 1
fi
if [[ ! -d "$RUN/checkpoint-6000" ]]; then
    printf 'Expected training checkpoint-6000 is missing.\n' >&2
    exit 1
fi

export GPU_IDS=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export WANDB_MODE=offline
export SKIP_INSTALL=1
export TRAIN_OUTPUT_DIR="$RUN"
printf '%s: resuming Inter-Edit training with filtered test/en at 5%% sampling probability.\n' "$(date -Is)"
exec bash "$ROOT/scripts/train_en_causal.sh" \
    --inter_edit_test_probability 0.05 --require_resume
