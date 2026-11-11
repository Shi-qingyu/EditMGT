"""Fail unless a model has complete raw summaries for all three benchmarks."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "outputs/real_time_baselines"
EXPECTED = {"magicbrush": (1571, 1571), "inter_edit": (3413, 3411),
            "pie_bench": (700, 700)}


def main():
    model_dir = sys.argv[1]
    for benchmark, (selected, scored) in EXPECTED.items():
        path = ROOT / model_dir / benchmark / "metrics/summary.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["complete"] and result["failed_predictions"] == 0, path
        assert result["selected_predictions"] == selected, path
        assert result["scored_predictions"] == scored, path
        print(f"{model_dir} {benchmark}: complete, {scored} scored", flush=True)


if __name__ == "__main__":
    main()
