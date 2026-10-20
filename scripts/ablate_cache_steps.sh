#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
exec "$PYTHON_BIN" "$PROJECT_ROOT/tests/ablate_cache_steps.py" "$@"
