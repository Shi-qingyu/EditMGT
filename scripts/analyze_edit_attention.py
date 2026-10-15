#!/usr/bin/env python
"""Generate edit-only test cases and visualize instruction localization attention."""
import argparse
import io
import json
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.attention_utils import save_attention_case  # noqa: E402
from src.editmgt import init_edit_mgt  # noqa: E402
from src.v2_model import negative_prompt  # noqa: E402
from src.inter_edit_dataset import DEFAULT_INTER_EDIT_ROOT, InterEditReader  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="checkpoint/EditMGT")
    parser.add_argument("--transformer-subfolder", default="editmgt")
    parser.add_argument(
        "--transformer-path",
        default=None,
        help="Optional trained transformer directory, e.g. runs/editmgt/checkpoint-200/transformer.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_INTER_EDIT_ROOT)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--output-dir", default="outputs/attention_debug")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=18)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-cases", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def decode_image(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    return Image.open(value)


def iter_test_rows(data_dir, manifest_path):
    data_dir = Path(data_dir)
    if (data_dir / "test" / "metadata.json").is_file() or (data_dir / "metadata.json").is_file():
        for _, filename, row_index, row in InterEditReader(data_dir, "test").iter_samples():
            yield filename, row_index, row
        return
    if manifest_path is None:
        raise ValueError("A parquet dataset requires --split-manifest")
    manifest = json.loads(Path(manifest_path).read_text())
    for filename in manifest["test"]:
        table = pq.read_table(Path(data_dir) / filename)
        for row_index, row in enumerate(table.to_pylist()):
            yield filename, row_index, row


def main():
    args = parse_args()
    pipe = init_edit_mgt(
        args.device,
        enable_bf16=args.bf16,
        base_model_path=args.model_path,
        transformer_subfolder=args.transformer_subfolder,
        transformer_path=args.transformer_path,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "attention_metrics.jsonl").write_text("")
    metrics = []
    for case_index, (filename, row_index, row) in enumerate(iter_test_rows(args.data_dir, args.split_manifest)):
        if case_index >= args.num_cases:
            break
        source = decode_image(row.get("input_img") or row.get("input") or row.get("image_file"))
        target = decode_image(row.get("output_img") or row.get("output") or row.get("edited_file"))
        mask = decode_image(row.get("mask_png") or row.get("edit_mask") or row.get("mask"))
        instruction = row.get("instruction") or row.get("edit_instruction")
        source_for_model = source.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        target_for_viz = target.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        mask_for_model = mask.resize((args.resolution, args.resolution), Image.Resampling.NEAREST)

        result = pipe(
            prompt=[instruction],
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            reference_image=[source_for_model],
            mask_image=mask_for_model,
            edit_only_diffusion=True,
            return_attention=True,
            attention_layers=args.layers,
            attention_keywords=args.keywords,
            negative_prompt=negative_prompt or None,
        )
        pixel_mask = torch.from_numpy(__import__("numpy").array(mask_for_model)).ge(128).float()
        sample_id = f"{Path(filename).stem}_row_{row_index}"
        metadata = save_attention_case(
            str(output_dir), sample_id, instruction, source_for_model, target_for_viz,
            result.images[0], pixel_mask, pipe.last_attention_analysis,
        )
        metrics.append(metadata)
        with (output_dir / "attention_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metadata) + "\n")

    if not metrics:
        raise RuntimeError("No test samples were analyzed")
    enrichments = [item["attention_enrichment"] for item in metrics]
    masses = [item["attention_mass"] for item in metrics]
    areas = [item["edit_area_ratio"] for item in metrics]
    layer_ids = metrics[0]["layers"].keys()
    summary = {
        "num_samples": len(metrics),
        "mean_attention_enrichment": statistics.mean(enrichments),
        "median_attention_enrichment": statistics.median(enrichments),
        "std_attention_enrichment": statistics.pstdev(enrichments),
        "mean_edit_attention_mass": statistics.mean(masses),
        "mean_edit_area_ratio": statistics.mean(areas),
        "per_layer_mean_attention_enrichment": {
            layer: statistics.mean(item["layers"][layer]["attention_enrichment"] for item in metrics)
            for layer in layer_ids
        },
    }
    (output_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
