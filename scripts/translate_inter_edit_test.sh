#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PROJECT_ROOT/scripts/translate_inter_edit.sh" \
    --output-dir "${INTER_EDIT_TEST_TRANSLATIONS_DIR:-$PROJECT_ROOT/data/inter_edit_test_zh_to_en_qwen35_9b}" \
    "$@" \
    --split test
