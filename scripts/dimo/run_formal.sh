#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
: "${DIMO_CONFIG:?Set DIMO_CONFIG to an explicit private JSON configuration}"
DIMO_PYTHON="${DIMO_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$DIMO_PYTHON" ]]; then
    printf 'Interpreter missing: %s. Prepare .venv or explicitly set DIMO_PYTHON.\n' "$DIMO_PYTHON" >&2
    exit 1
fi
# Same GPU selection contract as scripts/train.sh. Never assume GPU count/type.
if [[ -n "${GPU_IDS+x}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
VISIBLE_GPU_COUNT="$("$DIMO_PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
NUM_PROCESSES="${NUM_PROCESSES:-$VISIBLE_GPU_COUNT}"
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || (( NUM_PROCESSES > VISIBLE_GPU_COUNT )); then
    printf 'Invalid NUM_PROCESSES=%s; visible GPUs=%s; CUDA_VISIBLE_DEVICES=%s\n' \
        "$NUM_PROCESSES" "$VISIBLE_GPU_COUNT" "${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    exit 1
fi
export DIMO_NUM_PROCESSES="$NUM_PROCESSES"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LAUNCH=("$DIMO_PYTHON")
if (( NUM_PROCESSES > 1 )); then
    LAUNCH+=( -m torch.distributed.run --nnodes=1 --nproc_per_node="$NUM_PROCESSES" \
        --master_addr=127.0.0.1 --master_port="${MAIN_PROCESS_PORT:-25012}" )
fi
printf 'Region-DiMO: %s worker(s), CUDA_VISIBLE_DEVICES=%s, config=%s\n' \
    "$NUM_PROCESSES" "${CUDA_VISIBLE_DEVICES:-<unset>}" "$DIMO_CONFIG"
exec "${LAUNCH[@]}" -m src.dimo.train --config "$DIMO_CONFIG" "$@"
