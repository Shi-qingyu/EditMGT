#!/usr/bin/env python
"""Visualize token reveal order for the original full-generation EditMGT."""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_zero_shot import (  # noqa: E402
    configure_torch_runtime,
    decode_image,
    get_first,
    iter_test_samples,
)
from src.editmgt import init_edit_mgt  # noqa: E402
from src.v2_model import negative_prompt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record every masked-token sampling step and compare reveal progress "
            "inside/outside the dataset GT edit mask."
        )
    )
    parser.add_argument("--checkpoint", default="checkpoint/EditMGT")
    parser.add_argument("--data-dir", default="/mnt/geminihzceph/public/data/Inter-Edit")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/generation_order")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--visualization-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=36)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--reference-strength", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gif-duration-ms", type=int, default=450)
    parser.add_argument("--grid-columns", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--sdpa-backend", choices=("math", "efficient", "auto"), default="math"
    )
    return parser.parse_args()


def pil_mask_to_token_mask(mask: Image.Image, token_height: int, token_width: int):
    array = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array)[None, None]
    return F.interpolate(tensor, (token_height, token_width), mode="nearest")[0, 0].ge(0.5)


def decode_trace(pipe, token_states, args):
    images = []
    with torch.no_grad():
        for tokens in token_states:
            decoded = pipe.safe_decode_latents(
                tokens.to(args.device),
                batch_size=1,
                height=args.resolution,
                width=args.resolution,
                output_type="pil",
            )
            images.append(decoded[0])
    return images


def hide_unknown_pixels(image, tokens, mask_token_id):
    unknown = tokens.eq(mask_token_id).float()
    unknown = F.interpolate(
        unknown[:, None], size=(image.height, image.width), mode="nearest"
    )[0, 0].bool().numpy()
    array = np.asarray(image.convert("RGB")).copy()
    # Neutral gray denotes tokens that have not been generated yet. This overlay
    # is visualization-only and is never fed back to the model.
    array[unknown] = np.array([88, 88, 88], dtype=np.uint8)
    return Image.fromarray(array)


def overlay_edit_boundary(image, edit_mask):
    mask = edit_mask[None, None].float()
    mask = F.interpolate(mask, size=(image.height, image.width), mode="nearest")
    dilated = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)[0, 0].bool()
    eroded = ~F.max_pool2d(1 - mask, kernel_size=5, stride=1, padding=2)[0, 0].bool()
    boundary = (dilated ^ eroded).numpy()
    array = np.asarray(image.convert("RGB")).copy()
    array[boundary] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(array)


def add_frame_label(image, label, preserve_progress, edit_progress):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    text = f"{label} | preserve {preserve_progress:.0%} | edit {edit_progress:.0%}"
    box_height = max(30, image.height // 24)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255))
    return canvas


def region_progress(token_states, edit_mask, mask_token_id):
    edit_mask = edit_mask.flatten()
    preserve_mask = ~edit_mask
    preserve_values, edit_values = [], []
    for state in token_states:
        known = state[0].flatten().ne(mask_token_id)
        preserve_values.append(known[preserve_mask].float().mean().item())
        edit_values.append(known[edit_mask].float().mean().item())
    return preserve_values, edit_values


def token_reveal_steps(token_states, mask_token_id):
    height, width = token_states[0].shape[-2:]
    reveal = torch.full((height, width), -1, dtype=torch.long)
    reveal[token_states[0][0].ne(mask_token_id)] = 0
    for frame_index, state in enumerate(token_states[1:], start=1):
        known = state[0].ne(mask_token_id)
        reveal[(reveal < 0) & known] = frame_index
    reveal[reveal < 0] = len(token_states) - 1
    return reveal


def save_progress_plot(preserve_progress, edit_progress, output_path):
    steps = np.arange(len(edit_progress))
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(steps, preserve_progress, marker="o", markersize=3, label="Preserve region")
    plt.plot(steps, edit_progress, marker="o", markersize=3, label="Edit region")
    plt.xlabel("Generation frame (0 = initialization)")
    plt.ylabel("Visible/generated token ratio")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_order_map(reveal, edit_mask, output_path):
    order = reveal.float().numpy()
    plt.figure(figsize=(6, 6))
    image = plt.imshow(order, cmap="turbo", vmin=0, vmax=max(1, int(reveal.max())))
    plt.contour(edit_mask.float().numpy(), levels=[0.5], colors="white", linewidths=1.2)
    plt.title("Token reveal frame (white = GT edit boundary)")
    plt.axis("off")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_grid(frames, output_path, columns):
    columns = max(1, columns)
    thumb_size = 256
    rows = math.ceil(len(frames) / columns)
    grid = Image.new("RGB", (columns * thumb_size, rows * thumb_size), "white")
    for index, frame in enumerate(frames):
        thumb = frame.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        grid.paste(thumb, ((index % columns) * thumb_size, (index // columns) * thumb_size))
    grid.save(output_path)


def main():
    args = parse_args()
    if args.gif_duration_ms < 1:
        raise ValueError("--gif-duration-ms must be positive")
    configure_torch_runtime(args)
    pipe = init_edit_mgt(
        device=args.device,
        enable_bf16=args.bf16,
        base_model_path=args.checkpoint,
        transformer_subfolder="editmgt",
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_metadata = []

    samples = iter_test_samples(
        Path(args.data_dir), Path(args.split_manifest) if args.split_manifest else None, args.sample_index, args.num_cases
    )
    for case_number, (global_index, filename, row_index, row) in enumerate(samples, start=1):
        instruction = get_first(row, ("instruction", "edit_instruction"))
        source = decode_image(get_first(row, ("input_img", "input", "image_file"))).convert("RGB")
        target = decode_image(get_first(row, ("output_img", "output", "edited_file"))).convert("RGB")
        gt_mask = decode_image(get_first(row, ("mask_png", "edit_mask", "mask"))).convert("L")
        size = (args.resolution, args.resolution)
        source = source.resize(size, Image.Resampling.BILINEAR)
        target = target.resize(size, Image.Resampling.BILINEAR)
        gt_mask = gt_mask.resize(size, Image.Resampling.NEAREST)

        trace = []

        def capture_step(step_index, timestep, latents):
            trace.append({
                "step_index": int(step_index),
                "timestep": int(timestep.item()),
                "tokens": latents.detach().to("cpu").clone(),
            })

        case_seed = args.seed + global_index
        result = pipe(
            prompt=[instruction],
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=negative_prompt or None,
            generator=torch.Generator(device=args.device).manual_seed(case_seed),
            reference_image=[source],
            reference_strength=args.reference_strength,
            mask_image=None,
            edit_only_diffusion=False,
            return_attention=False,
            callback=capture_step,
            callback_steps=1,
        )
        if len(trace) != args.num_inference_steps:
            raise RuntimeError(
                f"Expected {args.num_inference_steps} trace states, captured {len(trace)}"
            )

        first_tokens = trace[0]["tokens"]
        token_mask = pil_mask_to_token_mask(gt_mask, first_tokens.shape[-2], first_tokens.shape[-1])
        if not token_mask.any() or token_mask.all():
            raise ValueError("GT mask must contain both edit and preserve tokens for order analysis")
        initial_tokens = torch.full_like(first_tokens, pipe.scheduler.config.mask_token_id)
        initial_label = "initial: all image tokens masked"
        token_states = [initial_tokens] + [item["tokens"] for item in trace]
        decoded = decode_trace(pipe, token_states, args)
        preserve_progress, edit_progress = region_progress(
            token_states, token_mask, pipe.scheduler.config.mask_token_id
        )

        labels = [initial_label] + [
            f"step {item['step_index'] + 1}/{args.num_inference_steps}, t={item['timestep']}"
            for item in trace
        ]
        visualization_size = (args.visualization_size, args.visualization_size)
        frames = [
            add_frame_label(
                overlay_edit_boundary(
                    hide_unknown_pixels(
                        image.resize(visualization_size, Image.Resampling.LANCZOS),
                        tokens,
                        pipe.scheduler.config.mask_token_id,
                    ),
                    token_mask,
                ),
                label,
                preserve_progress[index],
                edit_progress[index],
            )
            for index, (image, tokens, label) in enumerate(zip(decoded, token_states, labels))
        ]

        case_id = f"sample_{global_index:06d}_{Path(filename).stem}_row_{row_index}"
        case_dir = output_root / case_id
        frames_dir = case_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        source.save(case_dir / "source.png")
        target.save(case_dir / "target.png")
        gt_mask.save(case_dir / "gt_mask.png")
        result.images[0].save(case_dir / "generated.png")
        for index, frame in enumerate(frames):
            frame.save(frames_dir / f"frame_{index:03d}.png")
        frames[0].save(
            case_dir / "generation.gif",
            save_all=True,
            append_images=frames[1:],
            duration=args.gif_duration_ms,
            loop=0,
        )
        save_grid(frames, case_dir / "generation_grid.png", args.grid_columns)
        save_progress_plot(
            preserve_progress, edit_progress, case_dir / "region_generation_progress.png"
        )
        reveal = token_reveal_steps(token_states, pipe.scheduler.config.mask_token_id)
        save_order_map(reveal, token_mask, case_dir / "token_generation_order.png")
        mean_preserve_reveal = reveal[~token_mask].float().mean().item()
        mean_edit_reveal = reveal[token_mask].float().mean().item()

        metadata = {
            "case_id": case_id,
            "sample_index": global_index,
            "dataset_file": filename,
            "row_index": row_index,
            "instruction": instruction,
            "checkpoint": args.checkpoint,
            "mask_source": "dataset_gt_mask",
            "mask_convention": "white/1=edit, black/0=preserve",
            "generation_mode": "original_editmgt_full_generation",
            "generation_semantics": "all tokens start masked; GT mask is analysis-only",
            "seed": case_seed,
            "resolution": args.resolution,
            "visualization_size": args.visualization_size,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "reference_strength": args.reference_strength,
            "token_grid_size": list(token_mask.shape),
            "edit_token_ratio": token_mask.float().mean().item(),
            "preserve_visible_ratio_by_frame": preserve_progress,
            "edit_generated_ratio_by_frame": edit_progress,
            "mean_preserve_reveal_frame": mean_preserve_reveal,
            "mean_edit_reveal_frame": mean_edit_reveal,
            "edit_minus_preserve_mean_reveal_frame": (
                mean_edit_reveal - mean_preserve_reveal
            ),
            "positive_order_gap_means_preserve_region_was_revealed_earlier": True,
            "timesteps": [None] + [item["timestep"] for item in trace],
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        run_metadata.append(metadata)
        print(f"[{case_number}/{args.num_cases}] Generation-order visualization: {case_dir}")

    (output_root / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
