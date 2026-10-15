#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

UV_BIN="${UV_BIN:-uv}"
"$UV_BIN" venv .venv --python 3.10 --allow-existing
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
"$UV_BIN" pip install --python .venv/bin/python \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "$PYTORCH_INDEX_URL"
"$UV_BIN" pip install --python .venv/bin/python -r requirements.txt

PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python scripts/visualize_generation_order.py \
  --checkpoint checkpoint/EditMGT \
  --data-dir /mnt/geminihzceph/public/data/Inter-Edit \
  --sample-index 0 \
  --num-cases 4 \
  --num-inference-steps 36 \
  --reference-strength 1.1 \
  --sdpa-backend math \
  --output-dir outputs/generation_order \
  "$@"
