#!/usr/bin/env python
"""Create a deterministic, edit-type-stratified parquet file split."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/CrispEdit-mask-100k")
    parser.add_argument("--output", default="data/CrispEdit-mask-100k/split.json")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def edit_type(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def main():
    args = parse_args()
    if not 0 < args.test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1")
    data_dir = Path(args.data_dir)
    groups = defaultdict(list)
    for path in sorted(data_dir.glob("*.parquet")):
        groups[edit_type(path)].append(path.name)
    if not groups:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    train, test = [], []
    rng = random.Random(args.seed)
    counts = {}
    for category, files in sorted(groups.items()):
        rng.shuffle(files)
        test_count = max(1, round(len(files) * args.test_ratio))
        test.extend(files[:test_count])
        train.extend(files[test_count:])
        counts[category] = {"train_files": len(files) - test_count, "test_files": test_count}
    train.sort()
    test.sort()
    manifest = {
        "version": 1,
        "seed": args.seed,
        "split_unit": "parquet_file",
        "test_ratio": args.test_ratio,
        "train": train,
        "test": test,
        "counts_by_edit_type": counts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(train)} train and {len(test)} test files to {output}")


if __name__ == "__main__":
    main()
