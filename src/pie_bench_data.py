"""PIE-Bench v1 annotations, zero-based run-length masks, and local reports.

Dataset and mask conventions:
https://github.com/cure-lab/PnPInversion
https://github.com/cure-lab/PnPInversion/blob/main/evaluation/evaluate.py
"""
import csv
import html
import json
from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image


EDIT_TYPES = {
    "0": "random", "1": "change_object", "2": "add_object", "3": "delete_object",
    "4": "change_attribute_content", "5": "change_attribute_pose",
    "6": "change_attribute_color", "7": "change_attribute_material",
    "8": "change_background", "9": "change_style",
}


def decode_pie_mask(encoded, shape=(512, 512), fix_boundary=True):
    """Decode (zero-based start, length) pairs in row-major order, not COCO RLE."""
    if (len(shape) != 2 or any(not isinstance(size, Integral) or size < 1 for size in shape)
            or not isinstance(encoded, (list, tuple)) or len(encoded) % 2):
        raise ValueError("PIE mask requires a positive (height, width) and start/length pairs")
    height, width = map(int, shape)
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, count in zip(encoded[::2], encoded[1::2]):
        if (not isinstance(start, Integral) or not isinstance(count, Integral)
                or start < 0 or count < 0 or start > mask.size):
            raise ValueError(f"Invalid PIE mask run: start={start}, length={count}")
        mask[start:min(start + count, mask.size)] = 255
    mask = mask.reshape(height, width)
    if fix_boundary:
        # The published evaluator marks the outer border as edited to avoid
        # boundary annotation errors. Apply this consistently in both stages.
        mask[[0, -1], :] = 255
        mask[:, [0, -1]] = 255
    return Image.fromarray(mask, mode="L")


def clean_pie_caption(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("PIE-Bench captions and editing instructions must be nonempty strings")
    return value.replace("[", "").replace("]", "").strip()


def select_pie_cases(args):
    from src.benchmark_utils import file_identity, safe_name

    root = args.data_dir.resolve()
    annotation = (args.annotation_file or root / "mapping_file.json").resolve()
    images = (args.images_dir or root / "annotation_images").resolve()
    if not annotation.is_file():
        raise FileNotFoundError(f"PIE annotations not found: {annotation}")
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("PIE mapping_file.json must be a nonempty {sample_id: annotation} object")
    requested = set(args.sample_ids) if args.sample_ids is not None else None
    if requested is not None and len(requested) != len(args.sample_ids):
        raise ValueError("--sample-ids must be unique")
    if requested is not None and requested - set(payload):
        raise ValueError(f"Unknown PIE sample IDs: {sorted(requested - set(payload))}")
    categories = set(args.edit_category_list)
    selection = []
    for sample_id, row in payload.items():
        category = str(row["editing_type_id"])
        if requested is not None and sample_id not in requested:
            continue
        if category not in categories:
            if requested is not None:
                raise ValueError(f"PIE sample {sample_id} is outside --edit-category-list")
            continue
        relative = Path(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"PIE image_path must stay inside --images-dir: {relative}")
        encoded = row.get("mask")
        if not isinstance(encoded, list) or len(encoded) % 2:
            raise ValueError(f"Missing/invalid RLE mask for PIE sample {sample_id}")
        source_caption = clean_pie_caption(row["original_prompt"])
        target_caption = clean_pie_caption(row["editing_prompt"])
        if args.prompt_field == "editing_instruction":
            instruction = row.get("editing_instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Missing editing_instruction for PIE sample {sample_id}")
            instruction = instruction.strip()
        else:
            instruction = target_caption
        selection.append({
            "key": safe_name(sample_id), "sample_id": sample_id, "seed_key": f"pie_bench:{sample_id}",
            "instruction": instruction, "source_caption": source_caption, "caption": target_caption,
            "editing_instruction": row.get("editing_instruction"), "image_path": relative.as_posix(),
            "edit_type_id": category, "edit_type": EDIT_TYPES.get(category, category),
            "prediction": f"images/{safe_name(sample_id)}.png",
            "source": {"kind": "file", "path": str(images / relative)},
            "mask": {"kind": "pie_rle", "runs": encoded, "shape": [512, 512], "fix_boundary": True},
        })
    selection = selection[:args.limit]
    if not selection:
        raise ValueError("The selected PIE-Bench subset is empty")
    dataset = {
        "root": str(root), "images_dir": str(images), "annotation_file": file_identity(annotation),
        "benchmark_release": "PIE-Bench v1", "total_samples": len(payload),
        "selected_samples": len(selection), "full_benchmark": len(selection) == len(payload),
        "edit_category_list": sorted(categories), "prompt_field": args.prompt_field,
        "mask_encoding": "zero-based start/length pairs, row-major 512x512, official border correction",
        "preservation_reference": "source image; PIE-Bench does not provide an edited ground-truth image",
    }
    return selection, dataset


def write_pie_report(run, limit=50):
    """Export flat scores and bounded source/mask/prediction thumbnails."""
    from src.benchmark_data import BenchmarkImages

    metrics_dir = run.root / "metrics"
    summary_path = metrics_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    score_path = metrics_dir / "per_sample.jsonl"
    scores = {}
    if score_path.is_file():
        with score_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    scores[row["key"]] = row
        names = sorted({name for row in scores.values() for name in row["metrics"]})
        with (metrics_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "edit_type", "instruction", "inference_seconds", *names])
            writer.writeheader()
            for task in run.selection:
                row = scores.get(task["key"])
                if row is not None:
                    writer.writerow({"sample_id": task["sample_id"], "edit_type": task["edit_type"],
                                     "instruction": task["instruction"], "inference_seconds": row["inference_seconds"],
                                     **row["metrics"]})
        with (metrics_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "std", "count", "undefined_count"])
            writer.writeheader()
            for name, stats in summary.get("overall", {}).items():
                writer.writerow({"metric": name, **stats})
    if limit == 0:
        return
    directory = run.root / "report"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    resolver = BenchmarkImages()
    try:
        for task in run.selection[:limit]:
            record = run.record(task["key"])
            try:
                source = resolver.image(task["source"], "RGB")
                mask = resolver.edit_mask(task["mask"], "pie_bench").resize(source.size, Image.Resampling.NEAREST)
                overlay = Image.composite(Image.blend(source, Image.new("RGB", source.size, (255, 45, 45)), 0.4),
                                          source, mask)
                thumbnails = [("source", source), ("mask", overlay)]
                if record and record.get("status") == "ok" and run.image_path(task).is_file():
                    with Image.open(run.image_path(task)) as image:
                        thumbnails.append(("generated", image.convert("RGB")))
                figures = []
                for label, image in thumbnails:
                    image = image.copy()
                    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    name = f"{task['key']}_{label}.png"
                    image.save(directory / name)
                    figures.append(f'<figure><figcaption>{label}</figcaption><img loading="lazy" src="report/{name}"></figure>')
                details = json.dumps(scores.get(task["key"], {}).get("metrics", {}), ensure_ascii=False)
                status = record.get("status", "missing") if record else "missing"
                rows.append(f'<section><h2>{html.escape(task["sample_id"])} · {html.escape(task["edit_type"])}</h2>'
                            f'<p>{html.escape(task["instruction"])}</p><p>Target description: {html.escape(task["caption"])}</p>'
                            f'<p>Status: {html.escape(status)}</p><div class="images">{"".join(figures)}</div>'
                            f'<pre>{html.escape(details)}</pre></section>')
            except Exception as error:
                rows.append(f'<section><h2>{html.escape(task["sample_id"])}</h2><pre>{html.escape(str(error))}</pre></section>')
    finally:
        resolver.close()
    stats = "".join(f'<tr><td>{html.escape(name)}</td><td>{html.escape(str(value["mean"]))}</td>'
                    f'<td>{value["count"]}</td></tr>' for name, value in summary.get("overall", {}).items())
    page = ('<!doctype html><meta charset="utf-8"><title>PIE-Bench</title>'
            '<style>body{font-family:sans-serif;margin:24px;max-width:1200px}.images{display:flex;gap:16px}'
            'figure{margin:0}img{max-width:256px;width:100%}section{border-top:1px solid #ddd;padding:16px 0}'
            'td,th{padding:6px;text-align:left}pre{white-space:pre-wrap}</style><h1>PIE-Bench</h1>'
            f'<p>Selected: {len(run.selection)}; scored: {summary.get("scored_predictions", 0)}; '
            f'complete: {summary.get("complete", False)}. Showing the first {min(limit, len(run.selection))} cases.</p>'
            '<p>Red marks the annotation edit region. Preservation metrics compare generated and source images; '
            'text metrics use the target description. The dataset has no edited ground-truth images.</p>'
            '<p><a href="metrics/summary.json">Metric summary</a> · <a href="metrics/per_sample.csv">Per-image CSV</a>'
            ' · <a href="generation_summary.json">Generation timing</a></p>'
            '<table><tr><th>Metric (raw scale)</th><th>Mean</th><th>Valid count</th></tr>' + stats + '</table>'
            + "".join(rows))
    (run.root / "index.html").write_text(page, encoding="utf-8")
