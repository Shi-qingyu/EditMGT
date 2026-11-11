"""Exercise GIDE's PIE-Bench adapter on representative edit types.

This is a functional smoke check and does not contribute benchmark scores.
"""

import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_data import BenchmarkImages  # noqa: E402
from src.benchmark_utils import atomic_json, sample_seed, save_png  # noqa: E402
from run_baseline import load_backend  # noqa: E402


KEYS = {
    "replace_object": "111000000000",
    "add": "211000000000",
    "remove": "311000000000",
    "background": "811000000000",
    "style": "911000000000",
}
SELECTION = ROOT / "outputs/dimo_step4000_sft8000_comparison/vanilla36/pie_bench/selection.json"
OUTPUT = ROOT / "outputs/real_time_baselines/gide_pie/pie_bench_smoke_types"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tasks = {task["key"]: task for task in json.loads(SELECTION.read_text(encoding="utf-8"))}
    backend = load_backend("gide_pie")
    resolver = BenchmarkImages(ROOT / "data/inter_edit_tar_index")
    results = {}
    try:
        for label, key in KEYS.items():
            task = tasks[key]
            source = backend.prepare_source(resolver.image(task["source"], "RGB"))
            try:
                started = time.perf_counter()
                result = backend.generate(source, task, sample_seed(42, task["seed_key"]))
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                save_png(OUTPUT / f"{label}_{key}.png", result)
                results[label] = {"key": key, "status": "ok", "seconds": seconds}
            except Exception as error:
                results[label] = {"key": key, "status": "error", "error": repr(error)}
            atomic_json(OUTPUT / "results.json", results)
            print(label, results[label], flush=True)
    finally:
        resolver.close()
    if any(item["status"] != "ok" for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
