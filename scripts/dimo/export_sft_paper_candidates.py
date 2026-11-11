"""Copy manually reviewed Inter-Edit SFT demos as separate images."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil


FILES = ("source.png", "mask.png", "target.png", "sft18.png", "sft18.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Full visualization pool")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=6)
    args = parser.parse_args()
    root = args.output_dir
    destination = root / "paper_candidates"
    original = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in original["samples"]}
    choices = json.loads(args.selection.read_text(encoding="utf-8"))["samples"]
    if len(choices) != args.per_type * 4 or len({row["name"] for row in choices}) != len(choices):
        raise ValueError("Selection needs unique samples and the requested number per type")
    selected = []
    for choice in choices:
        name = choice["name"]
        row = dict(by_name[name], selection_note=choice["selection_note"])
        source = root / name
        target = destination / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            path = source / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            shutil.copy2(path, target / filename)
        (target / "instruction.txt").write_text(row["instruction"] + "\n", encoding="utf-8")
        selected.append(row)
    counts = Counter(row["edit_type"] for row in selected)
    if counts != {category: args.per_type for category in ("Add", "Local", "Remove", "Texture")}:
        raise ValueError(f"Unbalanced selection: {counts}")
    payload = dict(original, samples=selected, per_type=args.per_type,
                   selection="manual visual review of complete 400-sample train pool",
                   source_pool=str(root.resolve()))
    (destination / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.selection, destination / "selection.json")
    print(f"Exported {len(selected)} separate-file paper candidates to {destination}", flush=True)


if __name__ == "__main__":
    main()
