"""Choose test examples by annotation area alone for a labeled speed stress test."""

import argparse
import json
from pathlib import Path

from src.pie_bench_data import decode_pie_mask


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-benchmark", type=int, default=50)
    parser.add_argument("--max-mask-fraction", type=float, default=0.05)
    parser.add_argument("--inter-edit-records", type=Path,
                        default=ROOT / "outputs/efficiency_ablation_cbc/dense/inter_edit/records")
    parser.add_argument("--pie-mapping", type=Path, default=ROOT / "data/PIE-Bench/mapping_file.json")
    args = parser.parse_args()
    if args.per_benchmark < 1 or not 0 < args.max_mask_fraction < 1:
        parser.error("Use a positive sample count and a mask fraction in (0,1)")

    inter = []
    for path in args.inter_edit_records.glob("sample_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "ok":
            continue
        fraction = float(row["ablation"]["original_mask_fraction"])
        if 0 < fraction <= args.max_mask_fraction:
            inter.append((fraction, int(path.stem.removeprefix("sample_"))))
    inter.sort()
    pie = []
    mapping = json.loads(args.pie_mapping.read_text(encoding="utf-8"))
    for sample_id, row in mapping.items():
        mask = decode_pie_mask(row["mask"])
        fraction = sum(mask.histogram()[128:]) / (mask.width * mask.height)
        if 0 < fraction <= args.max_mask_fraction:
            pie.append((fraction, sample_id))
    pie.sort()
    if min(len(inter), len(pie)) < args.per_benchmark:
        raise ValueError(f"Only {len(inter)} Inter-Edit and {len(pie)} PIE-Bench eligible masks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = {"inter_edit": inter[:args.per_benchmark], "pie_bench": pie[:args.per_benchmark]}
    (args.output_dir / "inter_edit_indices.txt").write_text(
        "".join(f"{index}\n" for _, index in selected["inter_edit"]), encoding="utf-8")
    (args.output_dir / "pie_sample_ids.txt").write_text(
        "".join(f"{sample_id}\n" for _, sample_id in selected["pie_bench"]), encoding="utf-8")
    payload = {
        "selection": "smallest original annotation masks, sorted by fraction then sample ID",
        "per_benchmark": args.per_benchmark,
        "maximum_mask_fraction": args.max_mask_fraction,
        "inter_edit_source_records": str(args.inter_edit_records.resolve()),
        "pie_annotation_file": str(args.pie_mapping.resolve()),
        "samples": {
            benchmark: [{"fraction": fraction, "sample_id": sample_id} for fraction, sample_id in rows]
            for benchmark, rows in selected.items()
        },
    }
    (args.output_dir / "subset_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for benchmark, rows in selected.items():
        print(f"{benchmark}: {len(rows)} samples; mask range {rows[0][0]:.4%}–{rows[-1][0]:.4%}", flush=True)


if __name__ == "__main__":
    main()
