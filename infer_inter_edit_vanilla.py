"""Run the original EditMGT on the fixed training-preview cases, without masks."""
import argparse
import html
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.editmgt import init_edit_mgt
from src.v2_model import negative_prompt


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoint/EditMGT"),
                        help="Original EditMGT checkpoint containing the editmgt subfolder.")
    parser.add_argument(
        "--samples-manifest", type=Path,
        default=PROJECT_ROOT / "runs/editmgt_sparse_inter_edit/validation/samples.json",
        help="Reuse the exact preview sources and instructions listed in this manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/inter_edit_vanilla")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--reference-strength", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42, help="Each case uses seed + sample_id, as in validation.")
    parser.add_argument("--bf16", action="store_true", help="Opt into BF16; default matches infer.py FP32.")
    return parser.parse_args()


def load_cases(manifest_path, resolution):
    """Read only exported source/GT PNGs and instructions; never open a mask."""
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"No cases in {manifest_path}")
    cases = []
    for row in rows:
        input_dir = manifest_path.parent / row["input_dir"]
        images = {}
        for name in ("source", "target"):
            with Image.open(input_dir / f"{name}.png") as image:
                images[name] = image.convert("RGB").resize(
                    (resolution, resolution), Image.Resampling.BILINEAR,
                )
        cases.append({
            "test_index": row["test_index"], "sample_id": int(row["sample_id"]),
            "instruction": row["instruction"], "edit_type": row["edit_type"],
            "language": row.get("language"), "input_dir": str(input_dir.resolve()),
            **images,
        })
    return cases


def comparison_panel(case, generated, seed):
    tile = min(512, generated.width)
    panel = Image.new("RGB", (3 * tile, tile + 52), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), f"Sample {case['sample_id']} | {case['edit_type']} | Seed {seed}", fill="black")
    for column, (label, image) in enumerate((
        ("Source", case["source"]), ("Vanilla EditMGT", generated), ("Ground truth", case["target"]),
    )):
        draw.text((column * tile + 8, 32), label, fill="black")
        panel.paste(image.resize((tile, tile), Image.Resampling.BILINEAR), (column * tile, 52))
    return panel


def save_report(output_dir, records, panels, settings):
    sheet = Image.new("RGB", (max(panel.width for panel in panels), sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    sheet.save(output_dir / "contact_sheet.png")
    sections = []
    for record in records:
        figures = "".join(
            f'<figure><figcaption>{label}</figcaption>'
            f'<a href="{record["output_dir"]}/{name}.png"><img src="{record["output_dir"]}/{name}.png"></a></figure>'
            for label, name in (("Source", "source"), ("Vanilla EditMGT", "generated"), ("Ground truth", "target"))
        )
        sections.append(
            f'<section><h2>Sample {record["sample_id"]} · {html.escape(record["edit_type"])}</h2>'
            f'<p>{html.escape(record["instruction"])}</p><p>Seed: {record["seed"]}</p>'
            f'<div class="images">{figures}</div></section>'
        )
    style = "body{font-family:sans-serif;margin:24px}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%}"
    (output_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Vanilla EditMGT · Inter-Edit test</title>'
        f'<style>{style}</style><h1>Vanilla EditMGT · Inter-Edit test</h1>'
        '<p>Inputs: source image and original instruction. Ground truth is for display only.</p>'
        + "".join(sections), encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps({**settings, "cases": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


@torch.no_grad()
def main(args):
    cases = load_cases(args.samples_manifest, args.resolution)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Same initialization and sampling parameters as infer.py, using local base weights.
    pipe = init_edit_mgt(
        device=args.device, enable_bf16=args.bf16, base_model_path=args.model_path,
    )
    if pipe.transformer.config.sparse_target or pipe.transformer.config.token_type_rope:
        raise ValueError("Use the original EditMGT checkpoint for this vanilla baseline.")
    for model in (pipe.transformer, pipe.vqvae, pipe.text_encoder, pipe.text_encoder_t5):
        if model is not None:
            model.eval()
    pipe.local_guidance = 0.0
    records, panels = [], []
    started = time.perf_counter()
    for case in cases:
        seed = args.seed + case["sample_id"]
        generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)
        print(f"Sample {case['sample_id']} | seed={seed} | {case['instruction']}", flush=True)
        generated = pipe(
            prompt=[case["instruction"]],
            height=args.resolution, width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            reference_strength=args.reference_strength,
            reference_image=[case["source"]],
            negative_prompt=negative_prompt or None,
            generator=generator,
            edit_only_diffusion=False,
            sparse_target=False,
            use_reference_kv_cache=False,
        ).images[0].convert("RGB")
        case_dir = args.output_dir / f"sample_{case['sample_id']:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case["source"].save(case_dir / "source.png")
        case["target"].save(case_dir / "target.png")
        generated.save(case_dir / "generated.png")
        panel = comparison_panel(case, generated, seed)
        panel.save(case_dir / "comparison.png")
        panels.append(panel)
        record = {key: value for key, value in case.items() if key not in ("source", "target")}
        records.append({**record, "seed": seed, "output_dir": case_dir.name})
    save_report(args.output_dir, records, panels, {
        "model_path": args.model_path, "transformer_subfolder": "editmgt", "split": "test",
        "samples_manifest": str(args.samples_manifest.resolve()),
        "resolution": args.resolution, "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale, "reference_strength": args.reference_strength,
        "negative_prompt": negative_prompt, "seed": args.seed, "bf16": args.bf16,
        "mask_input": False, "edit_only_diffusion": False, "sparse_target": False,
        "token_type_rope": False, "use_reference_kv_cache": False,
        "seconds": time.perf_counter() - started,
    })
    print(f"Saved {len(cases)} cases to {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main(parse_args())
