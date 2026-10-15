#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use the existing environment; this launcher does not install dependencies.
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
exec "$PYTHON_BIN" "$PROJECT_ROOT/infer_inter_edit_vanilla.py" "$@"
