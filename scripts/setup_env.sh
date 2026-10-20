#!/usr/bin/env bash
set -euo pipefail

# Prepare the training environment without starting a training process.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# An existing interpreter is the only condition needed to reuse the
# environment, including a symlink to Python managed outside this checkout.
# Return before downloading uv, inspecting Python or creating cache files.
if [[ -e .venv/bin/python || -L .venv/bin/python ]]; then
    printf 'Found .venv/bin/python; skipping all environment installation.\n'
    exit 0
fi

# Keep the executable, managed Python and download cache in this checkout.
# Explicit overrides allow using an existing uv executable or shared cache.
UV_INSTALL_DIR="${UV_INSTALL_DIR:-$PROJECT_ROOT/.tools/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PROJECT_ROOT/.tools/python}"
export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-$PROJECT_ROOT/.tools/bin}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export UV_PYTHON_CACHE_DIR="${UV_PYTHON_CACHE_DIR:-$UV_CACHE_DIR/python}"
mkdir -p "$UV_CACHE_DIR"
SETUP_TMP_DIR="$(mktemp -d "$UV_CACHE_DIR/setup.XXXXXX")"
trap 'rm -rf -- "$SETUP_TMP_DIR"' EXIT

if [[ -n "${UV_BIN:-}" ]]; then
    if ! UV_BIN="$(command -v "$UV_BIN")"; then
        printf 'Cannot find the explicitly configured UV_BIN.\n' >&2
        exit 1
    fi
else
    UV_BIN="$UV_INSTALL_DIR/uv"
    if [[ ! -x "$UV_BIN" ]]; then
        printf 'Installing project-local uv into %s\n' "$UV_INSTALL_DIR"
        if command -v curl >/dev/null 2>&1; then
            curl --fail --location --silent --show-error --retry 3 \
                https://astral.sh/uv/install.sh -o "$SETUP_TMP_DIR/install-uv.sh"
        elif command -v wget >/dev/null 2>&1; then
            wget -q https://astral.sh/uv/install.sh -O "$SETUP_TMP_DIR/install-uv.sh"
        else
            printf 'Installing uv requires curl or wget.\n' >&2
            exit 1
        fi
        # Unmanaged installation does not edit shell profiles or write the
        # normal per-user installation receipt under /root/.local/share.
        UV_UNMANAGED_INSTALL="$UV_INSTALL_DIR" sh "$SETUP_TMP_DIR/install-uv.sh"
    fi
fi
printf 'Using uv: %s\n' "$UV_BIN"

# The interpreter is missing: prepare Python locally and preserve any
# incomplete environment directory before creating the new environment.
"$UV_BIN" python install 3.10 --no-bin
if [[ -e .venv || -L .venv ]]; then
    ENV_BACKUP="$PROJECT_ROOT/.venv.backup-$(date +%Y%m%d-%H%M%S)-$$"
    mv -- .venv "$ENV_BACKUP"
    printf 'Previous environment preserved at %s\n' "$ENV_BACKUP"
fi
"$UV_BIN" venv .venv --python 3.10 --managed-python

TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

# Install CUDA wheels from the selected PyTorch index. uv audits packages
# already satisfying the requirements instead of reinstalling them each run.
"$UV_BIN" pip install --python .venv/bin/python \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$PYTORCH_INDEX_URL"

# Keep version overrides consistent with requirements.txt during the second
# install, rather than silently restoring the default torch/torchvision pins.
awk -v torch_version="$TORCH_VERSION" -v torchvision_version="$TORCHVISION_VERSION" '
    /^torch==/ { print "torch==" torch_version; next }
    /^torchvision==/ { print "torchvision==" torchvision_version; next }
    { print }
' requirements.txt > "$SETUP_TMP_DIR/requirements.txt"
"$UV_BIN" pip install --python .venv/bin/python -r "$SETUP_TMP_DIR/requirements.txt"
printf 'Environment ready: %s/.venv/bin/python\n' "$PROJECT_ROOT"
