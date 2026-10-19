#!/usr/bin/env bash
set -euo pipefail

# Shared training launcher; the Python implementation lives in train/train.py.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use SKIP_INSTALL=1 after the environment has been prepared. Set UV_BIN to
# the uv executable if it is not on PATH.
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    UV_BIN="${UV_BIN:-uv}"
    "$UV_BIN" venv .venv --python 3.10 --allow-existing
    TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
    PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    "$UV_BIN" pip install --python .venv/bin/python \
      "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
      --index-url "$PYTORCH_INDEX_URL"
    "$UV_BIN" pip install --python .venv/bin/python -r requirements.txt
fi

# Respect scheduler/container CUDA_VISIBLE_DEVICES. GPU_IDS is an explicit
# override; these IDs use the CUDA namespace visible inside the container.
if [[ -n "${GPU_IDS+x}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi
VISIBLE_GPU_COUNT="$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count())')"
NUM_PROCESSES="${NUM_PROCESSES:-$VISIBLE_GPU_COUNT}"
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || (( NUM_PROCESSES > VISIBLE_GPU_COUNT )); then
    printf 'Invalid NUM_PROCESSES=%s: %s CUDA devices are visible (CUDA_VISIBLE_DEVICES=%s).\n' \
        "$NUM_PROCESSES" "$VISIBLE_GPU_COUNT" "${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
TRAIN_LAUNCH=(.venv/bin/python)
if (( NUM_PROCESSES > 1 )); then
    TRAIN_LAUNCH=(.venv/bin/accelerate launch --multi_gpu --gpu_ids all \
        --num_machines 1 --num_processes "$NUM_PROCESSES" \
        --main_process_port "${MAIN_PROCESS_PORT:-25012}")
fi
printf 'Launching %s worker(s), visible GPUs=%s, CUDA_VISIBLE_DEVICES=%s\n' \
    "$NUM_PROCESSES" "$VISIBLE_GPU_COUNT" "${CUDA_VISIBLE_DEVICES:-<unset>}"

TRAIN_EXTRA_ARGS=()
if [[ "${COMPILE_MODEL:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--compile_model --compile_backend "${COMPILE_BACKEND:-inductor}")
fi
if [[ "${USE_8BIT_ADAM:-1}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--use_8bit_adam)
fi
if [[ "${CACHE_CONDITIONING:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--cache_conditioning --condition_cache_threshold "${CONDITION_CACHE_THRESHOLD:-0.3}")
fi
TRAIN_EXTRA_ARGS+=(
    --timestep_sampling cosine
    --time_conditioning "${TIME_CONDITIONING:-legacy}"
    --mask_loss_reduction "${MASK_LOSS_REDUCTION:-token}"
    --cond_dropout_prob "${COND_DROPOUT_PROB:-0.0}"
)

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-./runs/editmgt_sparse_inter_edit}"
INTER_EDIT_LANGUAGE="${INTER_EDIT_LANGUAGE:-all}"

exec "${TRAIN_LAUNCH[@]}" train/train.py \
    "${TRAIN_EXTRA_ARGS[@]}" \
    --train_edit_model True \
    --edit_only_diffusion \
    --sparse_target \
    --sparse_target_halo "${SPARSE_TARGET_HALO:-0}" \
    --mask_debug_dir "$TRAIN_OUTPUT_DIR/mask_debug" \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --train_batch_size 4 \
    --dataloader_num_workers 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_grad_norm 10 \
    --pretrained_model_name_or_path "./checkpoint/EditMGT" \
    --transformer_subfolder "editmgt" \
    --text_encoder_architecture 'CLIP_Gemma2' \
    --instance_dataset "InterEditDataset" \
    --instance_data_dir "/mnt/geminihzceph/public/data/Inter-Edit" \
    --dataset_cache_dir "./data/inter_edit_tar_index" \
    --dataset_split train \
    --dataset_language "$INTER_EDIT_LANGUAGE" \
    --resume_from_checkpoint "latest" \
    --resolution 1024 \
    --mixed_precision no \
    --lr_scheduler constant \
    --max_train_steps 500000 \
    --checkpointing_steps 200 \
    --validation_steps 200 \
    --validation_at_start \
    --validation_num_images 5 \
    --validation_num_inference_steps 18 \
    --validation_inference_shift 1.0 \
    --validation_guidance_scale 6 \
    --validation_negative_prompt "${VALIDATION_NEGATIVE_PROMPT:-}" \
    --validation_reference_kv_cache_threshold "${CONDITION_CACHE_THRESHOLD:-0.3}" \
    --validation_seed 42 \
    --report_to "${REPORT_TO:-wandb}" \
    --logging_steps 10 \
    "$@"
