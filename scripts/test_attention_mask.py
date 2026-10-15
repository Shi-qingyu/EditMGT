#!/usr/bin/env python
"""Two-pass zero-shot editing using instruction attention as the edit mask."""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_zero_shot import (  # noqa: E402
    configure_torch_runtime,
    decode_image,
    get_first,
    iter_test_samples,
)
from src.attention_utils import num_attention_layers  # noqa: E402
from src.editmgt import init_edit_mgt  # noqa: E402
from src.v2_model import negative_prompt  # noqa: E402
from src.inter_edit_dataset import DEFAULT_INTER_EDIT_ROOT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover an edit mask from cross attention, then run edit-only inference."
    )
    parser.add_argument("--checkpoint", default="checkpoint/EditMGT")
    parser.add_argument("--data-dir", default=DEFAULT_INTER_EDIT_ROOT)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument(
        "--assets-manifest",
        default=None,
        help="Optional JSON cases with image/instruction; bypasses the parquet dataset.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/attention_mask_zero_shot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--discovery-steps", type=int, default=18)
    parser.add_argument("--num-inference-steps", type=int, default=18)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.4,
        help="Select positions whose min-max-normalized attention is at least this value.",
    )
    parser.add_argument(
        "--mask-closing",
        type=int,
        default=1,
        help="Closing radius on the token grid to fill holes and join nearby regions.",
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=2,
        help="Final dilation radius on the token grid to preserve the full edit boundary.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Attention layers; default is the model's last eight layers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--sdpa-backend", choices=("math", "efficient", "auto"), default="math"
    )
    return parser.parse_args()


def attention_to_mask(
    attention: torch.Tensor, threshold: float, closing: int, dilation: int
):
    if not 0 <= threshold <= 1:
        raise ValueError("--mask-threshold must be in [0, 1]")
    if closing < 0 or dilation < 0:
        raise ValueError("--mask-closing and --mask-dilation must be non-negative")
    flat_attention = attention.detach().float().flatten().cpu()
    side = math.isqrt(flat_attention.numel())
    if side * side != flat_attention.numel():
        raise ValueError(f"Attention map has non-square length {flat_attention.numel()}")
    attention_min = flat_attention.min()
    attention_range = flat_attention.max() - attention_min
    if attention_range <= torch.finfo(flat_attention.dtype).eps:
        raise ValueError("Cross-attention map is constant; cannot derive a threshold mask")
    normalized_attention = (flat_attention - attention_min) / attention_range
    mask = normalized_attention >= threshold
    selected_count = int(mask.sum().item())
    if selected_count == 0:
        raise ValueError(
            f"No attention positions met --mask-threshold {threshold}; lower the threshold"
        )
    mask = mask.reshape(1, 1, side, side).float()
    if closing:
        kernel_size = 2 * closing + 1
        mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=closing)
        # Binary erosion after dilation completes morphological closing. Explicit
        # zero padding keeps the image boundary outside the edit region.
        padded = F.pad(mask, (closing, closing, closing, closing), value=0)
        mask = 1 - F.max_pool2d(1 - padded, kernel_size=kernel_size, stride=1)
    closed_count = int(mask.sum().item())
    if dilation:
        mask = F.max_pool2d(mask, kernel_size=2 * dilation + 1, stride=1, padding=dilation)
    return mask[0, 0].bool(), selected_count, closed_count


def resize_mask_to_pil(mask: torch.Tensor, resolution: int) -> Image.Image:
    mask = F.interpolate(
        mask[None, None].float(), size=(resolution, resolution), mode="nearest"
    )[0, 0]
    return Image.fromarray(mask.byte().mul(255).numpy(), mode="L")


def save_attention_heatmap(attention: torch.Tensor, output_path: Path):
    side = math.isqrt(attention.numel())
    raw = attention.detach().float().reshape(side, side).cpu().numpy()
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    plt.imsave(output_path, normalized, cmap="jet", vmin=0, vmax=1)


def iter_input_samples(args):
    if args.assets_manifest is None:
        yield from iter_test_samples(
            Path(args.data_dir), Path(args.split_manifest) if args.split_manifest else None, args.sample_index, args.num_cases
        )
        return

    cases = json.loads(Path(args.assets_manifest).read_text())
    if args.sample_index < 0 or args.num_cases < 1:
        raise ValueError("--sample-index must be non-negative and --num-cases must be positive")
    selected = cases[args.sample_index:args.sample_index + args.num_cases]
    if len(selected) != args.num_cases:
        raise IndexError(
            f"Requested {args.num_cases} asset cases from index {args.sample_index}, "
            f"but manifest contains {len(cases)} cases"
        )
    for index, case in enumerate(selected, start=args.sample_index):
        image_path = Path(case["image"])
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        row = {
            "case_id": case.get("id", image_path.stem),
            "image_file": str(image_path),
            "instruction": case["instruction"],
        }
        yield index, image_path.name, 0, row


def main():
    args = parse_args()
    configure_torch_runtime(args)
    pipe = init_edit_mgt(
        device=args.device,
        enable_bf16=args.bf16,
        base_model_path=args.checkpoint,
        transformer_subfolder="editmgt",
    )
    total_layers = num_attention_layers(pipe.transformer)
    attention_layers = args.layers or list(range(max(0, total_layers - 8), total_layers))
    invalid_layers = [layer for layer in attention_layers if layer < 0 or layer >= total_layers]
    if invalid_layers:
        raise ValueError(f"Invalid attention layers {invalid_layers}; model has {total_layers} layers")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = []
    samples = iter_input_samples(args)
    for global_index, filename, row_index, row in samples:
        instruction = get_first(row, ("instruction", "edit_instruction"))
        source = decode_image(get_first(row, ("input_img", "input", "image_file"))).convert("RGB")
        target_value = next(
            (row[key] for key in ("output_img", "output", "edited_file") if row.get(key) is not None),
            None,
        )
        target = decode_image(target_value).convert("RGB") if target_value is not None else None
        model_size = (args.resolution, args.resolution)
        source_model = source.resize(model_size, Image.Resampling.BILINEAR)
        target_model = target.resize(model_size, Image.Resampling.BILINEAR) if target is not None else None
        case_seed = args.seed + global_index

        # Pass 1: no GT mask is read or passed. Generate the full image while
        # capturing instruction localization from generation queries to text keys.
        discovery = pipe(
            prompt=[instruction],
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.discovery_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=negative_prompt or None,
            generator=torch.Generator(device=args.device).manual_seed(case_seed),
            reference_image=[source_model],
            reference_strength=args.reference_strength,
            edit_only_diffusion=False,
            return_attention=True,
            attention_layers=attention_layers,
        )
        analysis = pipe.last_attention_analysis
        # Global attention pooling: average across the selected layers, all
        # attention heads, and every valid (non-padding, non-special) instruction
        # token. No keyword/token selection participates in mask discovery.
        attention = analysis["all_tokens"]
        token_mask, selected_count, closed_count = attention_to_mask(
            attention, args.mask_threshold, args.mask_closing, args.mask_dilation
        )
        attention_mask = resize_mask_to_pil(token_mask, args.resolution)

        # Original EditMGT baseline: use the same seed and denoising settings as
        # the masked run, but generate the full image without an edit mask.
        edit_seed = case_seed + 1
        original_editmgt = pipe(
            prompt=[instruction],
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=negative_prompt or None,
            generator=torch.Generator(device=args.device).manual_seed(edit_seed),
            reference_image=[source_model],
            reference_strength=args.reference_strength,
            edit_only_diffusion=False,
            return_attention=False,
        )

        # Pass 2: the discovered attention mask is the only edit mask. Reusing
        # the baseline seed makes the original/masked comparison controlled.
        edited = pipe(
            prompt=[instruction],
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=negative_prompt or None,
            generator=torch.Generator(device=args.device).manual_seed(edit_seed),
            reference_image=[source_model],
            reference_strength=args.reference_strength,
            mask_image=attention_mask,
            edit_only_diffusion=True,
            return_attention=False,
        )

        case_id = row.get(
            "case_id", f"sample_{global_index:06d}_{Path(filename).stem}_row_{row_index}"
        )
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        stale_pixel_result = case_dir / "generated_pixel_preserve.png"
        if stale_pixel_result.exists():
            stale_pixel_result.unlink()
        source_model.save(case_dir / "source.png")
        if target_model is not None:
            target_model.save(case_dir / "target.png")
        discovery.images[0].save(case_dir / "discovery_full_generation.png")
        original_editmgt.images[0].save(case_dir / "original_editmgt.png")
        save_attention_heatmap(attention, case_dir / "cross_attention.png")
        attention_mask.save(case_dir / "attention_mask.png")
        edited.images[0].save(case_dir / "generated_token_preserve.png")

        metadata = {
            "case_id": case_id,
            "sample_index": global_index,
            "dataset_file": filename,
            "row_index": row_index,
            "instruction": instruction,
            "checkpoint": args.checkpoint,
            "mask_source": "cross_attention_only_no_gt_mask",
            "attention_direction": analysis["direction"],
            "attention_layers": attention_layers,
            "attention_pooling": "mean_over_layers_heads_and_all_valid_instruction_tokens",
            "pooled_instruction_token_positions": analysis["valid_token_positions"],
            "input_source": "assets_manifest" if args.assets_manifest else "parquet_test_split",
            "attention_grid_size": analysis["token_grid_size"],
            "attention_normalization": "per_case_min_max_to_0_1",
            "mask_threshold": args.mask_threshold,
            "selected_attention_tokens_before_morphology": selected_count,
            "mask_ratio_before_morphology": selected_count / attention.numel(),
            "mask_closing": args.mask_closing,
            "mask_ratio_after_closing": closed_count / attention.numel(),
            "mask_ratio_after_dilation": token_mask.float().mean().item(),
            "mask_dilation": args.mask_dilation,
            "seed_discovery": case_seed,
            "seed_original_editmgt": edit_seed,
            "seed_edit": edit_seed,
            "discovery_steps": args.discovery_steps,
            "num_inference_steps": args.num_inference_steps,
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        run_metadata.append(metadata)
        print(f"[{len(run_metadata)}/{args.num_cases}] Attention-mask edit completed: {case_dir}")

    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
