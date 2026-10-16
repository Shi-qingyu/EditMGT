"""Infer the fixed English-prompt cases with a trained sparse EditMGT checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from infer_inter_edit_vanilla import PROJECT_ROOT, load_cases
from src.editmgt import init_edit_mgt
from src.mask_utils import normalize_edit_mask_image, resize_binary_edit_mask
from src.mask_schedule import validate_inference_shift
from src.train_validation import mask_overlay, run_inter_edit_validation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "runs/editmgt_sparse_inter_edit_en_cache_magicbrush_lognormal/checkpoint-1200",
        help="Training checkpoint directory containing transformer/config.json and weights.",
    )
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoint/EditMGT"),
                        help="Base checkpoint supplying frozen text encoders, VQ and sampler.")
    parser.add_argument(
        "--samples-manifest", type=Path,
        default=PROJECT_ROOT / "runs/editmgt_sparse_inter_edit/validation/samples.json",
        help="Reuse the same five preview images, masks and English prompt translations.",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/inter_edit_sparse_checkpoint_1000_en")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=18)
    parser.add_argument("--inference-shift", type=validate_inference_shift, default=1.0,
                        help="Compatibility option; only 1 is accepted, and inference always uses cosine.")
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--negative-prompt", default="",
                        help="Negative text; empty matches CFG's empty-text training condition.")
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparse-target-halo", type=int, default=1,
                        help="Transformer-cell halo; default 1 matches train/train_en.sh, including for older checkpoints.")
    parser.add_argument("--preserve-unmasked-pixels", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy source pixels outside the original mask; disable to inspect the raw VQ output.")
    parser.add_argument("--disable-reference-kv-cache", action="store_true")
    parser.add_argument("--disable-condition-kv-cache", action="store_true",
                        help="Recompute text/reference while retaining the checkpoint's trained low-noise masks.")
    parser.add_argument("--reference-kv-cache-threshold", type=float, default=0.3)
    parser.add_argument("--bf16", action="store_true", help="Opt into BF16; default is FP32.")
    return parser.parse_args()


def prepare_cases(manifest_path, output_dir, resolution):
    cases = load_cases(manifest_path, resolution)
    for case in cases:
        # This is the exact test annotation exported for training previews.
        # Only source, English prompt and mask are model inputs; GT is for display.
        source_input_dir = Path(case["input_dir"])
        with Image.open(source_input_dir / "mask.png") as image:
            mask = resize_binary_edit_mask(normalize_edit_mask_image(image), (resolution, resolution))
        if mask.getbbox() is None:
            raise ValueError(f"Sample {case['sample_id']} has an empty edit mask")
        overlay = mask_overlay(case["source"], mask)
        case_dir = output_dir / "inputs" / f"sample_{case['sample_id']:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        for name, image in (("source", case["source"]), ("target", case["target"]),
                            ("mask", mask), ("mask_overlay", overlay)):
            image.save(case_dir / f"{name}.png")
        case.update({
            "source_input_dir": str(source_input_dir),
            "input_dir": str(case_dir.relative_to(output_dir)),
            "resolution": resolution, "mask": mask, "overlay": overlay,
        })
    return cases


@torch.no_grad()
def main(args):
    transformer_path = args.checkpoint / "transformer"
    if not (transformer_path / "config.json").is_file():
        raise FileNotFoundError(f"No trained Transformer config at {transformer_path / 'config.json'}")
    cases = prepare_cases(args.samples_manifest, args.output_dir, args.resolution)
    pipe = init_edit_mgt(
        device=args.device, enable_bf16=args.bf16,
        base_model_path=args.model_path, transformer_path=str(transformer_path),
    )
    if not pipe.transformer.config.sparse_target or not pipe.transformer.config.token_type_rope:
        raise ValueError("Expected a trained sparse-target checkpoint with token-type RoPE enabled.")
    for model in (pipe.transformer, pipe.vqvae, pipe.text_encoder, pipe.text_encoder_t5):
        if model is not None:
            model.eval()
    pipe.local_guidance = 0.0
    halo = (pipe.transformer.config.sparse_target_halo
            if args.sparse_target_halo is None else args.sparse_target_halo)
    step_label = args.checkpoint.name.removeprefix("checkpoint-")
    checkpoint_step = int(step_label) if step_label.isdigit() else 0
    print(f"Transformer: {transformer_path}", flush=True)
    for case in cases:
        print(f"Sample {case['sample_id']} | seed={args.seed + case['sample_id']} | {case['instruction']}", flush=True)
    report = run_inter_edit_validation(
        pipe=pipe, cases=cases, output_dir=args.output_dir, step=checkpoint_step,
        num_inference_steps=args.num_inference_steps,
        inference_shift=args.inference_shift,
        guidance_scale=args.guidance_scale, reference_strength=args.reference_strength,
        seed=args.seed, edit_only_diffusion=True, sparse_target=True,
        sparse_target_halo=halo, negative_prompt=args.negative_prompt,
        use_reference_kv_cache=not args.disable_reference_kv_cache,
        reference_kv_cache_threshold=args.reference_kv_cache_threshold,
        use_condition_kv_cache=False if args.disable_condition_kv_cache else None,
        preserve_unmasked_pixels=args.preserve_unmasked_pixels,
    )
    metadata_path = report["step_dir"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "checkpoint": str(args.checkpoint.resolve()),
        "transformer_path": str(transformer_path.resolve()), "model_path": args.model_path,
        "samples_manifest": str(args.samples_manifest.resolve()),
        "prompt_language": "en", "mask_input": True, "token_type_rope": True, "bf16": args.bf16,
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(cases)} cases to {report['step_dir'] / 'index.html'}", flush=True)


if __name__ == "__main__":
    main(parse_args())
