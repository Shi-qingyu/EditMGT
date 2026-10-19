"""Generate and score held-out MagicBrush dev edits and autoregressive sessions."""
import argparse
import json
from pathlib import Path

from src.benchmark_data import (
    BenchmarkImages, attach_magicbrush_captions, magicbrush_sessions, magicbrush_tasks,
)
from src.benchmark_utils import (
    PROJECT_ROOT, BenchmarkRun, add_benchmark_arguments, generate_benchmark,
    score_benchmark, validate_arguments, validate_metric_inputs,
)


def parse_args(native_editmgt=False):
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, "magicbrush", native_editmgt=native_editmgt)
    parser.set_defaults(output_dir=None, metrics="l1,l2,clip-i,dino")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/magicbrush",
                        help="Defaults to the existing local MagicBrush parquet release.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="Default: held-out dev; test requires its separate official release.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional edit_sessions JSON or flat JSON/JSONL turn manifest.")
    parser.add_argument("--caption-file", type=Path, default=None,
                        help="Official global_description.json for CLIP-T; edit instructions are not a substitute.")
    parser.add_argument("--mode", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Local data-parallel replicas; whole sessions stay on one GPU. Shell launchers default to 8.")
    args = parser.parse_args()
    if args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    if args.output_dir is None:
        suffix = "_vanilla" if native_editmgt else ""
        args.output_dir = PROJECT_ROOT / "outputs" / f"benchmark_magicbrush_{args.split}{suffix}"
    validate_arguments(parser, args)
    return args


def find_captions(args, root):
    if args.caption_file is None:
        root = Path(root)
        args.caption_file = next((path for path in (root / "global_description.json", root.parent / "global_description.json")
                                  if path.is_file()), None)


def main(args):
    selection, dataset = None, None
    if args.stage != "metrics":
        sessions, dataset = magicbrush_sessions(args.data_dir, args.split, args.manifest)
        selection, count = magicbrush_tasks(sessions, args.mode, args.limit)
        dataset.update(mode=args.mode, selected_sessions=count,
                       selected_turns=sum(len(turns) for _, turns in list(sessions.items())[:args.limit]),
                       full_requested_split=count == len(sessions))
        find_captions(args, dataset["root"])
        attach_magicbrush_captions(selection, args.caption_file)
    names = validate_metric_inputs(args, "magicbrush", selection) if args.stage != "metrics" else None
    run = BenchmarkRun(args, "magicbrush", selection, dataset)
    images = BenchmarkImages()
    try:
        if args.stage == "metrics":
            find_captions(args, run.info["settings"]["dataset"]["root"])
            attach_magicbrush_captions(run.selection, args.caption_file)
            names = validate_metric_inputs(args, "magicbrush", run.selection)
        if args.num_gpus > 1:
            from src.benchmark_parallel import run_magicbrush_data_parallel

            complete = run_magicbrush_data_parallel(args, run, names)
        else:
            if args.stage != "metrics":
                generate_benchmark(args, run, images)
            if args.stage != "generate":
                complete = score_benchmark(args, run, images, names)
            else:
                complete = json.loads((run.root / "generation_summary.json").read_text())["complete"]
        print(f"Saved MagicBrush outputs to {run.root}", flush=True)
        return 0 if complete else 1
    finally:
        images.close()
        run.close()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
