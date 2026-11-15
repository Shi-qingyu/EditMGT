#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export RUN_TAG=blip_flan GPU_IDS

for benchmark in magicbrush inter_edit; do
    "$ROOT/.venv/bin/python" - "$benchmark" <<'PY'
import json
import sys
from pathlib import Path
root = Path('outputs/real_time_baselines/caption_protocol')
benchmark = sys.argv[1]
manifest = json.loads((root / f'{benchmark}.json').read_text())
expected = {'magicbrush': 1571, 'inter_edit': 3413}[benchmark]
assert len(manifest) == expected, (benchmark, len(manifest), expected)
assert all(row['source_caption'] and row['caption'] for row in manifest.values())
print(f'Validated {benchmark} input-only caption manifest: {expected} tasks', flush=True)
PY
done

bash third_party/table_completion/run_model.sh dice_paella magicbrush inter_edit
bash third_party/table_completion/run_model.sh gide_pie magicbrush inter_edit
"$ROOT/.venv/bin/python" third_party/table_completion/update_dice_gide_all.py
"$ROOT/.venv/bin/python" third_party/table_completion/write_dice_gide_report.py
"$ROOT/.venv/bin/python" third_party/table_completion/validate_tables.py
echo "DICE/GIDE three-benchmark evaluations and table updates complete"
