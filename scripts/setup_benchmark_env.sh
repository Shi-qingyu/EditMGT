#!/usr/bin/env bash
set -euo pipefail

# Explicitly install metric dependencies into the existing project environment.
# Training and benchmark launchers do not invoke this installer automatically.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ ! -x .venv/bin/python ]]; then
    printf 'Missing .venv/bin/python. Prepare the project environment with: bash scripts/setup_env.sh\n' >&2
    exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
if [[ -n "${UV_BIN:-}" ]]; then
    if ! UV_BIN="$(command -v "$UV_BIN")"; then
        printf 'Cannot find the explicitly configured UV_BIN.\n' >&2
        exit 1
    fi
    PACKAGE_INSTALL=("$UV_BIN" pip install --python .venv/bin/python)
elif [[ -x "$PROJECT_ROOT/.tools/uv/uv" ]]; then
    PACKAGE_INSTALL=("$PROJECT_ROOT/.tools/uv/uv" pip install --python .venv/bin/python)
elif command -v uv >/dev/null 2>&1; then
    PACKAGE_INSTALL=(uv pip install --python .venv/bin/python)
else
    # uv-created environments may not contain pip. Bootstrap it locally when
    # neither a configured uv nor a project/PATH uv executable is available.
    if ! .venv/bin/python -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("pip") is None)'; then
        .venv/bin/python -m ensurepip --upgrade
    fi
    PACKAGE_INSTALL=(.venv/bin/python -m pip install)
fi

# AlphaCLIP imports pkg_resources in setup.py and at runtime. Install its
# legacy build requirements first, then build with isolation disabled so the
# compatible setuptools is also used during package metadata generation.
"${PACKAGE_INSTALL[@]}" 'setuptools<82' wheel
"${PACKAGE_INSTALL[@]}" --no-build-isolation -r "$PROJECT_ROOT/requirements-benchmark.txt"
printf 'Benchmark dependencies installed into %s/.venv\n' "$PROJECT_ROOT"
