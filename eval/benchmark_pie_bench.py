"""Generate, score and visualize PIE-Bench v1 with trained or released EditMGT."""
import argparse
import json
from pathlib import Path

from src.benchmark_data import BenchmarkImages
from src.benchmark_utils import (
    PROJECT_ROOT, BenchmarkRun, add_benchmark_arguments, generate_benchmark,
    score_benchmark, validate_arguments, validate_metric_inputs,
)
from src.pie_bench_data import EDIT_TYPES, select_pie_cases, write_pie_report


def parse_args(native_editmgt=False):
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, "pie_bench", native_editmgt=native_editmgt)
    parser.set_defaults(metric_resolution=512)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/PIE-Bench")
    parser.add_argument("--annotation-file", type=Path, default=None,
                        help="Defaults to <data-dir>/mapping_file.json.")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Defaults to <data-dir>/annotation_images.")
    parser.add_argument("--edit-category-list", "--categories", nargs="+", choices=tuple(EDIT_TYPES),
                        default=list(EDIT_TYPES), help="PIE categories 0 through 9; default includes every category.")
    parser.add_argument("--sample-ids", nargs="+", default=None,
                        help="Exact mapping-file IDs, including leading zeros; preserves annotation order.")
    parser.add_argument("--prompt-field", choices=["editing_instruction", "editing_prompt"],
                        default="editing_instruction", help="Text sent to EditMGT; CLIP always uses global captions.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="One editing/scoring replica per GPU. Shell launchers default to 8.")
    parser.add_argument("--report-limit", type=int, default=50,
                        help="Number of source/mask/prediction rows in index.html; 0 disables HTML.")
    args = parser.parse_args()
    if args.num_gpus < 1 or args.report_limit < 0:
        parser.error("--num-gpus must be positive; --report-limit must be nonnegative")
    validate_arguments(parser, args)
    return args


def main(args):
    selection, dataset = (None, None) if args.stage == "metrics" else select_pie_cases(args)
    names = validate_metric_inputs(args, "pie_bench", selection) if args.stage != "metrics" else None
    run = BenchmarkRun(args, "pie_bench", selection, dataset)
    images = BenchmarkImages()
    try:
        if args.stage == "metrics":
            names = validate_metric_inputs(args, "pie_bench", run.selection)
        if args.num_gpus > 1:
            from src.benchmark_parallel import run_pie_bench_data_parallel

            complete = run_pie_bench_data_parallel(args, run, names)
        else:
            if args.stage != "metrics":
                generate_benchmark(args, run, images)
            if args.stage != "generate":
                complete = score_benchmark(args, run, images, names)
            else:
                complete = json.loads((run.root / "generation_summary.json").read_text())["complete"]
        write_pie_report(run, args.report_limit)
        print(f"Saved PIE-Bench outputs to {run.root}", flush=True)
        return 0 if complete else 1
    finally:
        images.close()
        run.close()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
