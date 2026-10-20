"""Visualize original EditMGT with the official demo's default inference settings."""
import argparse
import html
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    from src.mask_schedule import validate_inference_shift

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoint/EditMGT"),
                        help="Original EditMGT checkpoint containing the editmgt subfolder.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional original Transformer directory or full checkpoint; "
                             "default: --model-path/editmgt. Sparse fine-tuned models are rejected.")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/mnt/geminihzceph/public/data/Inter-Edit"),
                        help="Inter-Edit root or its train/ directory.")
    parser.add_argument("--tar-cache-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--sample-indices", nargs="+", type=int,
                        help="Zero-based global train indices before language filtering; overrides --num-samples.")
    parser.add_argument("--language", choices=("en", "zh", "all"), default="en")
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--samples-manifest", type=Path, default=None,
        help="Reuse all exported train/test cases, prompts and order from samples.json instead of selecting train rows.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/inter_edit_train_vanilla")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--inference-shift", type=validate_inference_shift, default=1.0,
                        help="Compatibility option; only 1 is accepted, and inference always uses cosine.")
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--negative-prompt", default=None,
                        help="Default: official src.v2_model.negative_prompt. Pass an empty string to override it.")
    parser.add_argument("--reference-strength", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=None,
                        help="Official default: no fixed seed. Opt in with --seed N for N + sample_id per case.")
    parser.add_argument("--bf16", action="store_true", help="Opt into BF16; the official demo defaults to FP32.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Export selected inputs and their comparison grid without loading model weights.")
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.resolution < 32 or args.resolution % 32:
        parser.error("--resolution must be a positive multiple of 32")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    if args.samples_manifest is not None and args.sample_indices is not None:
        parser.error("--samples-manifest reuses all exported cases; do not combine it with --sample-indices")
    return args


def load_cases(manifest_path, resolution, image_resample=None):
    """Reuse train or legacy test exports; masks are optional and used for display only."""
    from PIL import Image

    from src.mask_utils import normalize_edit_mask_image, resize_binary_edit_mask
    from src.train_validation import mask_overlay

    image_resample = Image.Resampling.BICUBIC if image_resample is None else Image.Resampling(image_resample)
    manifest_path = Path(manifest_path).expanduser().resolve()
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"No cases in {manifest_path}")
    cases = []
    for row in rows:
        original_instruction = row["instruction"]
        instruction = row.get("instruction_en") or original_instruction
        input_dir = manifest_path.parent / row["input_dir"]
        images, original_sizes = {}, {}
        for name in ("source", "target"):
            with Image.open(input_dir / f"{name}.png") as image:
                original_sizes[f"{name}_original_size"] = row.get(f"{name}_original_size", list(image.size))
                images[name] = image.convert("RGB").resize(
                    (resolution, resolution), image_resample,
                )
        if (input_dir / "mask.png").is_file():
            with Image.open(input_dir / "mask.png") as image:
                images["mask"] = resize_binary_edit_mask(normalize_edit_mask_image(image), (resolution, resolution))
            images["overlay"] = mask_overlay(images["source"], images["mask"])
        cases.append({
            **row, **original_sizes, "split": row.get("split", "train" if "train_index" in row else "test"),
            "sample_id": int(row["sample_id"]), "resolution": resolution,
            "instruction": instruction, "original_instruction": original_instruction,
            "source_input_dir": str(input_dir.resolve()),
            "image_resampling": image_resample.name.lower(),
            **images,
        })
    if len({case["split"] for case in cases}) != 1 or cases[0]["split"] not in ("train", "test"):
        raise ValueError("samples.json must contain cases from a single Inter-Edit train or test split")
    return cases


def case_metadata(case):
    return {key: value for key, value in case.items() if key not in ("source", "target", "mask", "overlay")}


def prepare_cases(args, image_resample=None):
    from PIL import Image

    from infer import prepare_cases as prepare_train_cases
    from src.train_validation import _stack_panels

    # Shared callers (e.g. cache ablation) retain the training reader's filter.
    # The vanilla main explicitly selects the official RGB resize default.
    image_resample = Image.Resampling.BILINEAR if image_resample is None else Image.Resampling(image_resample)
    if args.samples_manifest is None:
        # Share selection, language filtering and empty-mask handling with the trained model.
        return prepare_train_cases(args, image_resample=image_resample)
    cases = load_cases(args.samples_manifest, args.resolution, image_resample=image_resample)
    for case in cases:
        case_dir = args.output_dir / "inputs" / f"sample_{case['sample_id']:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for key, name in (("source", "source"), ("target", "target"), ("mask", "mask"), ("overlay", "mask_overlay")):
            if key in case:
                case[key].save(case_dir / f"{name}.png")
        case["input_dir"] = str(case_dir.relative_to(args.output_dir))
    split = cases[0]["split"]
    selection = {
        "split": split, "selection": "manifest", "samples_manifest": str(args.samples_manifest.resolve()),
        "selected_sample_ids": [case["sample_id"] for case in cases],
        f"selected_{split}_indices": [case.get(f"{split}_index") for case in cases],
    }
    for name, contents in (("samples.json", [case_metadata(case) for case in cases]), ("selection.json", selection)):
        (args.output_dir / name).write_text(json.dumps(contents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _stack_panels([comparison_panel(case, None, None) for case in cases]).save(args.output_dir / f"{split}_inputs.png")
    return cases


def comparison_panel(case, generated, seed):
    from PIL import Image, ImageDraw

    tile = min(512, case["source"].width)
    images = [("Source", case["source"])]
    if "overlay" in case:
        images.append(("Original edit mask (display only)", case["overlay"]))
    images.append(("Ground truth", case["target"]))
    if generated is not None:
        images.append(("Vanilla EditMGT", generated))
    panel = Image.new("RGB", (len(images) * tile, tile + 52), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    title = f"Sample {case['sample_id']} | {case['edit_type']}"
    if generated is not None:
        title += f" | Seed {seed}" if seed is not None else " | Unfixed seed (official default)"
    draw.text((8, 8), title, fill="black")
    for column, (label, image) in enumerate(images):
        draw.text((column * tile + 8, 32), label, fill="black")
        panel.paste(image.resize((tile, tile), Image.Resampling.BILINEAR), (column * tile, 52))
    return panel


def save_report(output_dir, records, panels, settings):
    from src.train_validation import _stack_panels

    _stack_panels(panels).save(output_dir / "contact_sheet.png")
    sections = []
    for record in records:
        images = [("Source", f'{record["input_dir"]}/source.png')]
        if record["mask_display"]:
            images.append(("Original edit mask (display only)", f'{record["input_dir"]}/mask_overlay.png'))
        images.extend((("Ground truth", f'{record["input_dir"]}/target.png'),
                       ("Vanilla EditMGT", record.get("model_generated_path", f'{record["output_dir"]}/generated.png'))))
        figures = "".join(
            f'<figure><figcaption>{label}</figcaption>'
            f'<a href="{html.escape(path)}"><img src="{html.escape(path)}"></a></figure>'
            for label, path in images
        )
        original = record.get("original_instruction")
        original_caption = (f'<p>Original instruction: {html.escape(original)}</p>'
                            if original and original != record["instruction"] else "")
        sections.append(
            f'<section><h2>Sample {record["sample_id"]} · {html.escape(record["edit_type"])}</h2>'
            f'{original_caption}<p>Instruction: {html.escape(record["instruction"])}</p>'
            f'<p>Seed: {record["seed"] if record["seed"] is not None else "unfixed (official default)"}</p>'
            f'<div class="images">{figures}</div>'
            f'<p><a href="{html.escape(record.get("generated_path", record["output_dir"] + "/generated.png"))}">'
            'Download result at original source size</a></p></section>'
        )
    title = f'Vanilla EditMGT · Inter-Edit {settings["split"]}'
    style = "body{font-family:sans-serif;margin:24px}.images{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:12px}figure{margin:0}img{width:100%}"
    (output_dir / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
        f'<style>{style}</style><h1>{title}</h1>'
        '<p>Model inputs: source image and instruction. Ground truth and annotation masks are for display only. '
        'Generation uses the full token grid without sparse targets, KV reuse or RGB compositing.</p>'
        + "".join(sections), encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps({**settings, "cases": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main(args):
    from PIL import Image

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transformer_path = None
    if args.checkpoint is not None:
        from infer import resolve_checkpoint

        transformer_path, _, _ = resolve_checkpoint(args.checkpoint)
    # Official Image.resize() uses BICUBIC by default for RGB input images.
    cases = prepare_cases(args, image_resample=Image.Resampling.BICUBIC)
    split = cases[0]["split"]
    if args.prepare_only:
        print(f"Saved {len(cases)} {split} inputs to {args.output_dir / f'{split}_inputs.png'}", flush=True)
        return

    import torch

    from src.editmgt import init_edit_mgt
    from src.v2_model import negative_prompt as official_negative_prompt

    negative_prompt = official_negative_prompt if args.negative_prompt is None else args.negative_prompt

    pipe = init_edit_mgt(
        device=args.device, enable_bf16=args.bf16, base_model_path=args.model_path,
        transformer_path=str(transformer_path) if transformer_path is not None else None,
    )
    config = pipe.transformer.config
    if (getattr(config, "sparse_target", False) or getattr(config, "token_type_rope", False)
            or getattr(config, "cache_conditioning", False) or getattr(config, "time_conditioning", "legacy") != "legacy"):
        raise ValueError("Use the original EditMGT checkpoint for this vanilla baseline.")
    for model in (pipe.transformer, pipe.vqvae, pipe.text_encoder, pipe.text_encoder_t5):
        if model is not None:
            model.eval().requires_grad_(False)
    pipe.local_guidance = 0.0
    records, panels = [], []
    started = time.perf_counter()
    for case in cases:
        seed = args.seed + int(case["sample_id"]) if args.seed is not None else None
        generator = torch.Generator(device=pipe._execution_device).manual_seed(seed) if seed is not None else None
        print(f"Sample {case['sample_id']} | seed={seed if seed is not None else 'unfixed'} | {case['instruction']}", flush=True)
        with torch.inference_mode():
            generated = pipe(
                prompt=[case["instruction"]],
                height=args.resolution, width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                inference_shift=args.inference_shift,
                guidance_scale=args.guidance_scale,
                reference_strength=args.reference_strength,
                reference_image=[case["source"]],
                negative_prompt=negative_prompt or None,
                generator=generator,
                num_images_per_prompt=1,
                edit_only_diffusion=False,
                sparse_target=False,
                use_reference_kv_cache=False,
                use_condition_kv_cache=False,
                preserve_unmasked_pixels=False,
            ).images[0].convert("RGB")
        case_dir = args.output_dir / f"sample_{case['sample_id']:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        # Preserve the model-resolution result for aligned comparison panels;
        # the official exported result is resized back to the source dimensions.
        generated.save(case_dir / "generated_model.png")
        native_size = tuple(case["source_original_size"])
        generated.resize(native_size).save(case_dir / "generated.png")
        panel = comparison_panel(case, generated, seed)
        panel.save(case_dir / "comparison.png")
        panels.append(panel)
        records.append({
            **case_metadata(case), "seed": seed, "output_dir": case_dir.name, "mask_display": "mask" in case,
            "generated_path": f"{case_dir.name}/generated.png", "generated_size": list(native_size),
            "model_generated_path": f"{case_dir.name}/generated_model.png",
            "model_generated_size": list(generated.size),
        })
    save_report(args.output_dir, records, panels, {
        "model_path": args.model_path, "transformer_subfolder": None if transformer_path else "editmgt", "split": split,
        "checkpoint": str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None,
        "transformer_path": str(transformer_path) if transformer_path else None,
        "samples_manifest": str(args.output_dir / "samples.json"),
        "source_samples_manifest": str(args.samples_manifest.expanduser().resolve()) if args.samples_manifest else None,
        "data_root": str(args.data_dir.expanduser().resolve()) if args.samples_manifest is None else None,
        "selection": "manifest" if args.samples_manifest else ("indices" if args.sample_indices is not None else args.selection),
        "selection_seed": args.selection_seed, "device": args.device,
        "resolution": args.resolution, "num_inference_steps": args.num_inference_steps,
        "inference_shift": pipe.scheduler.inference_shift,
        "inference_schedule": getattr(pipe, "last_inference_schedule_info", None),
        "guidance_scale": args.guidance_scale, "reference_strength": args.reference_strength,
        "negative_prompt": negative_prompt, "negative_prompt_source": "official" if args.negative_prompt is None else "cli",
        "seed": args.seed, "seed_policy": "base_plus_sample_id" if args.seed is not None else "unfixed_global_rng",
        "bf16": args.bf16, "image_resampling": "bicubic", "output_size_policy": "original_source_size",
        "language_filter": args.language if args.samples_manifest is None else None,
        "mask_input": False, "edit_only_diffusion": False, "sparse_target": False,
        "token_type_rope": False, "use_reference_kv_cache": False,
        "use_condition_kv_cache": False, "cache_conditioning": False, "time_conditioning": "legacy",
        "preserve_unmasked_pixels": False, "output_compositing": "none",
        "seconds": time.perf_counter() - started,
    })
    print(f"Saved {len(cases)} cases to {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main(parse_args())
