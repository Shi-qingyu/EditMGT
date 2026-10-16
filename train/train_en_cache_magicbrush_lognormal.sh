#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Historical filename retained as an alias. It now uses the same cosine recipe
# and default output directory as train_en_cache_magicbrush.sh. An explicit old
# checkpoint may still be resumed; its historical sampler is not restored.
exec bash "$PROJECT_ROOT/train/train_en_cache_magicbrush.sh" "$@"
