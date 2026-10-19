"""Generate and score the official Inter-Edit test benchmark or its English subset."""
import argparse
import json
from pathlib import Path

from src.benchmark_data import BenchmarkImages
from src.benchmark_utils import (
    BenchmarkRun, add_benchmark_arguments, file_identity, generate_benchmark,
    score_benchmark, validate_arguments, validate_metric_inputs,
)
from src.inter_edit_dataset import (
    DEFAULT_INTER_EDIT_ROOT, InterEditReader, inter_edit_prompt_language, matches_inter_edit_language,
)


def parse_args(native_editmgt=False):
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_arguments(parser, "inter_edit", native_editmgt=native_editmgt)
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_INTER_EDIT_ROOT))
    parser.add_argument("--language", choices=["all", "en", "zh"], default="en")
    parser.add_argument("--sample-indices", nargs="+", type=int, default=None,
                        help="Optional original test indices, all of which must match --language.")
    parser.add_argument("--tar-cache-dir", type=Path, default=None)
    args = parser.parse_args()
    validate_arguments(parser, args)
    return args


def select_cases(args):
    reader = InterEditReader(args.data_dir, "test", args.tar_cache_dir)
    eligible = [row for row in reader.test_rows if matches_inter_edit_language(row, args.language)]
    if args.sample_indices is not None:
        indices = set(args.sample_indices)
        chosen = [row for row in eligible if row["index"] in indices]
        if {row["index"] for row in chosen} != indices:
            raise ValueError("Some --sample-indices are missing or outside the requested test language")
    else:
        chosen = eligible
    chosen = chosen[:args.limit]
    selection = []
    for row in chosen:
        index = int(row["index"])
        def spec(archive, field):
            return {"kind": "tar", "root": str(reader.root.resolve()), "archive": archive, "member": row[field]}
        selection.append({
            "key": f"sample_{index:05d}", "test_index": index, "seed_key": f"inter_edit:{index}",
            "instruction": row["instruction"], "language": inter_edit_prompt_language(row),
            "edit_type": row["edit_type"], "prediction": f"images/{index:05d}.png",
            "source": spec("source_images.tar", "source_path"),
            "target": spec("target_images.tar", "gt_path"), "mask": spec("masks.tar", "mask_path"),
        })
    dataset = {
        "root": str(reader.root.resolve()), "split": "test", "language": args.language,
        "total_test_samples": len(reader.test_rows), "eligible_samples": len(eligible),
        "selected_samples": len(selection), "full_language_subset": len(selection) == len(eligible),
        "full_test_benchmark": len(selection) == len(reader.test_rows),
        "metadata_file": file_identity(reader.root / "metadata.json"),
        "archive_files": [file_identity(reader.root / name) for name in
                          ("source_images.tar", "target_images.tar", "masks.tar") if (reader.root / name).is_file()],
    }
    return selection, dataset


def main(args):
    selection, dataset = (None, None) if args.stage == "metrics" else select_cases(args)
    names = validate_metric_inputs(args, "inter_edit", selection) if args.stage != "metrics" else None
    run = BenchmarkRun(args, "inter_edit", selection, dataset)
    images = BenchmarkImages(args.tar_cache_dir)
    try:
        if args.stage == "metrics":
            names = validate_metric_inputs(args, "inter_edit", run.selection)
        if args.stage != "metrics":
            generate_benchmark(args, run, images)
        if args.stage != "generate":
            complete = score_benchmark(args, run, images, names)
        else:
            complete = json.loads((run.root / "generation_summary.json").read_text())["complete"]
        print(f"Saved Inter-Edit outputs to {run.root}", flush=True)
        return 0 if complete else 1
    finally:
        images.close()
        run.close()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
