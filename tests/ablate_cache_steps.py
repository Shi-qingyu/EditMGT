"""Compare image quality and speed for exact text/reference KV reuse counts, including early snapshots."""
import argparse
import copy
import csv
import html
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mask_schedule import scheduled_mask_ratio


def build_cache_plan(num_steps, phase_threshold, reuse_counts=None, allow_early_cache=True):
    """Convert requested REUSE counts to thresholds; the prefill forward is not counted."""
    if num_steps < 1 or not 0 < phase_threshold <= 1:
        raise ValueError("Expected positive inference steps and a condition phase threshold in (0, 1]")
    ratios = [scheduled_mask_ratio(i, num_steps, "cosine") for i in range(num_steps)]
    first_capture = next((i for i in range(num_steps - 1) if ratios[i] <= phase_threshold), None)
    maximum = num_steps - first_capture - 1 if first_capture is not None else 0
    if reuse_counts is None:
        reuse_counts = ([k for k in (17, 8, 3, 0) if k < num_steps] if allow_early_cache else
                        sorted({0, maximum, *(k for k in (1, 2, 4, 8, 12) if k <= maximum)}))
    if len(set(reuse_counts)) != len(reuse_counts):
        raise ValueError("--cache-steps must be unique")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 0 or k >= num_steps for k in reuse_counts):
        raise ValueError(f"--cache-steps must be integers in [0, {num_steps - 1}]; prefill is excluded")
    if not allow_early_cache and any(k > maximum for k in reuse_counts):
        raise ValueError(
            f"With {num_steps} forwards and the checkpoint's fixed phase threshold {phase_threshold}, "
            f"the maximum native KV reuse count is {maximum}. Enable --allow-early-cache for approximate "
            "early snapshots, or increase --num-inference-steps."
        )
    plans = []
    counts = list(reuse_counts)
    if 0 not in counts:
        counts.append(0)
    for count in counts:
        capture = num_steps - count - 1 if count else None
        early = bool(count and ratios[capture] > phase_threshold)
        threshold = min(1.0, math.nextafter(ratios[capture], math.inf)) if count else 0.0
        actual = next((i for i in range(num_steps - 1) if ratios[i] <= threshold), None)
        if actual != capture:
            raise ValueError("The requested reuse count cannot be represented by this cosine schedule")
        plans.append({
            "key": f"reuse_{count:02d}", "reuse_count": count, "capture_step": capture,
            "capture_mask_ratio": ratios[capture] if count else None,
            "capture_threshold": threshold, "reuse_steps": list(range(capture + 1, num_steps)) if count else [],
            "fresh_forwards": num_steps - count,
            "approximate_early_snapshot": early,
            "cache_mode": "early_snapshot" if early else "trained_low_noise" if count else "none",
        })
    return plans, maximum


def parse_args(argv=None):
    from infer import parse_args as inference_args

    defaults = inference_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=defaults.checkpoint)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--tar-cache-dir", type=Path)
    parser.add_argument("--samples-manifest", type=Path,
                        help="Reuse all exported cases in manifest order; masks must be present.")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample-indices", type=int, nargs="+")
    parser.add_argument("--language", choices=("en", "zh", "all"), default="en")
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--cache-steps", type=int, nargs="+",
                        help="Exact numbers of later forwards reusing KV, excluding prefill. "
                             "Zero is always included. Default at 18 forwards: 17 8 3 0.")
    parser.add_argument("--condition-cache-threshold", type=float, default=0.8,
                        help="Fixed prefix phase for every variant, including no cache; default: 0.8.")
    parser.add_argument("--allow-early-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow approximate snapshots before the trained phase for the 17/8/3-step sweep. "
                             "--no-allow-early-cache restricts capture to the configured low-noise phase.")
    parser.add_argument("--num-inference-steps", type=int, default=18)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--sparse-target-halo", type=int, default=0)
    parser.add_argument("--preserve-unmasked-pixels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmups per variant on the first case.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions per case and variant.")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--difference-gain", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/cache_steps_ablation")
    parser.add_argument("--plan-only", action="store_true", help="Write plan.json without images or model loading.")
    parser.add_argument("--prepare-only", action="store_true", help="Export plan and inputs without model loading.")
    args = parser.parse_args(argv)
    if min(args.num_samples, args.num_inference_steps, args.repeats, args.tile_size) < 1 or args.warmup < 0:
        parser.error("Sample/step/repeat/tile counts must be positive and --warmup nonnegative")
    if args.resolution < 32 or args.resolution % 32 or args.sparse_target_halo < 0:
        parser.error("Resolution must be a positive multiple of 32 and halo nonnegative")
    if not math.isfinite(args.difference_gain) or args.difference_gain <= 0:
        parser.error("--difference-gain must be positive and finite")
    if not 0 < args.condition_cache_threshold <= 1:
        parser.error("--condition-cache-threshold must be in (0, 1]")
    if args.samples_manifest and args.sample_indices:
        parser.error("--samples-manifest already fixes the sample selection")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def image_metrics(generated, baseline, target, source, mask):
    import numpy as np

    pixels = [np.asarray(image.convert("RGB"), dtype=np.float32) / 255 for image in (generated, baseline, target, source)]
    generated, baseline, target, source = pixels
    roi = np.asarray(mask) > 0
    if not roi.any():
        raise ValueError("Ablation metrics require a nonempty annotation mask")
    delta = np.abs(generated - baseline)
    return {
        "mse_gt": float(np.square(generated - target).mean()),
        "mse_edit_gt": float(np.square(generated - target)[roi].mean()),
        "mae_gt": float(np.abs(generated - target).mean()),
        "mae_edit_gt": float(np.abs(generated - target)[roi].mean()),
        "mae_outside_source": float(np.abs(generated - source)[~roi].mean()) if (~roi).any() else None,
        "mae_no_cache": float(delta.mean()), "mae_edit_no_cache": float(delta[roi].mean()),
        "max_difference_no_cache": float(delta.max()),
        "changed_pixel_fraction": float((delta.max(axis=-1) > 0).mean()),
    }


def check_cache_info(pipe, plan):
    info = copy.deepcopy(getattr(pipe, "last_condition_cache_info", None))
    if (info is None or bool(info.get("enabled")) != bool(plan["reuse_count"])
            or info.get("capture_step") != plan["capture_step"] or info.get("reuse_steps") != plan["reuse_steps"]
            or bool(info.get("approximate_early_snapshot")) != plan["approximate_early_snapshot"]):
        raise RuntimeError(f"Actual cache capture/reuse differs from the plan: expected {plan}, observed {info}")
    return info


def generate(pipe, case, plan, args):
    import torch

    device = torch.device(pipe._execution_device)
    generator = torch.Generator(device=device).manual_seed(args.seed + int(case["sample_id"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = pipe(
            prompt=[case["instruction"]], reference_image=[case["source"]], mask_image=case["mask"],
            height=args.resolution, width=args.resolution, num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale, reference_strength=args.reference_strength,
            negative_prompt=args.negative_prompt, generator=generator, inference_shift=1.0,
            edit_only_diffusion=True, sparse_target=bool(pipe.transformer.config.sparse_target),
            sparse_target_halo=args.sparse_target_halo, lora_part_enable=True,
            preserve_unmasked_pixels=args.preserve_unmasked_pixels,
            use_reference_kv_cache=bool(plan["reuse_count"]), use_condition_kv_cache=bool(plan["reuse_count"]),
            reference_kv_cache_threshold=plan["capture_threshold"],
            condition_kv_cache_reuse_steps=plan["reuse_count"],
            allow_early_condition_cache=plan["approximate_early_snapshot"],
        ).images[0].convert("RGB")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return generated, elapsed, check_cache_info(pipe, plan)


def comparison_images(case, images, plans, output_dir, args):
    import numpy as np
    from PIL import Image, ImageDraw

    tiles = [("Source", case["source"]), ("Edit mask", case["overlay"]), ("Ground truth", case["target"])]
    tiles.extend((f"Reuse {p['reuse_count']}" + (" (approx.)" if p["approximate_early_snapshot"] else ""),
                  images[p["key"]]) for p in plans)
    box = case["mask"].getbbox()
    box = (max(0, box[0] - 16), max(0, box[1] - 16),
           min(case["source"].width, box[2] + 16), min(case["source"].height, box[3] + 16))

    def panel(labeled_images, crop=False):
        tile = args.tile_size
        sheet = Image.new("RGB", (len(labeled_images) * tile, tile + 28), "#f5f5f5")
        draw = ImageDraw.Draw(sheet)
        for i, (label, image) in enumerate(labeled_images):
            draw.text((i * tile + 5, 6), label, fill="black")
            if crop:
                image = image.crop(box)
            sheet.paste(image.resize((tile, tile), Image.Resampling.BILINEAR), (i * tile, 28))
        return sheet

    full = panel(tiles)
    full.save(output_dir / "comparison.png")
    panel(tiles, crop=True).save(output_dir / "roi_comparison.png")
    baseline = np.asarray(images["reuse_00"], dtype=np.float32)
    differences = []
    for plan in plans:
        difference = np.abs(np.asarray(images[plan["key"]], dtype=np.float32) - baseline).mean(axis=-1)
        intensity = np.clip(difference * args.difference_gain, 0, 255).astype(np.uint8)
        heatmap = Image.fromarray(np.stack((intensity, intensity // 2, np.zeros_like(intensity)), axis=-1))
        heatmap.save(output_dir / f"{plan['key']}_difference.png")
        differences.append((f"Reuse {plan['reuse_count']} | diff x{args.difference_gain:g}", heatmap))
    panel(differences).save(output_dir / "differences.png")
    return full


def aggregate(records, plans):
    baseline_seconds = sum(r["median_seconds"] for r in records if r["reuse_count"] == 0)
    summary = []
    for plan in plans:
        selected = [r for r in records if r["variant"] == plan["key"]]
        timings = [r["median_seconds"] for r in selected]
        metrics = {name: statistics.mean(r["metrics"][name] for r in selected) for name in (
            "mae_gt", "mae_edit_gt", "mae_no_cache", "mae_edit_no_cache", "changed_pixel_fraction",
            "mse_gt", "mse_edit_gt",
        )}
        outside = [r["metrics"]["mae_outside_source"] for r in selected
                   if r["metrics"]["mae_outside_source"] is not None]
        metrics["mae_outside_source"] = statistics.mean(outside) if outside else None
        metrics["psnr_gt"] = -10 * math.log10(metrics["mse_gt"]) if metrics["mse_gt"] else None
        metrics["psnr_edit_gt"] = -10 * math.log10(metrics["mse_edit_gt"]) if metrics["mse_edit_gt"] else None
        summary.append({**plan, "num_cases": len(selected), "mean_seconds": statistics.mean(timings),
                        "std_seconds": statistics.pstdev(timings), "speedup": baseline_seconds / sum(timings),
                        "images_per_second": 1 / statistics.mean(timings), **metrics})
    return summary


def plot_results(output_dir, plans, summary, num_steps, phase_threshold=.3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    schedule = np.zeros((len(plans), num_steps), dtype=int)
    for row, plan in enumerate(plans):
        if plan["capture_step"] is not None:
            schedule[row, plan["capture_step"]] = 1
            schedule[row, plan["reuse_steps"]] = 2
    fig, ax = plt.subplots(figsize=(max(7, num_steps * .35), 1.6 + .45 * len(plans)))
    ax.imshow(schedule, cmap=ListedColormap(["#dbe4ef", "#f6af45", "#4cab83"]), vmin=0, vmax=2, aspect="auto")
    ax.set(xticks=range(num_steps), yticks=range(len(plans)), xlabel="Forward index (zero-based)",
           yticklabels=[f"Reuse {p['reuse_count']}" + (" (approx.)" if p["approximate_early_snapshot"] else "")
                        for p in plans], title="KV reuse schedule; early snapshots are approximate")
    phase_start = next((i for i in range(num_steps)
                        if scheduled_mask_ratio(i, num_steps, "cosine") <= phase_threshold), None)
    if phase_start is not None:
        ax.axvline(phase_start - .5, color="#27364b", linestyle="--", linewidth=1.2)
        ax.text(phase_start - .5, -.65, "trained low-noise phase", fontsize=8, ha="right")
    ax.legend(handles=[Patch(color=c, label=t) for c, t in (
        ("#dbe4ef", "Recompute"), ("#f6af45", "Prefill"), ("#4cab83", "Reuse KV"))],
        loc="upper center", bbox_to_anchor=(.5, -.3), ncol=3)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"cache_schedule.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    counts = [r["reuse_count"] for r in summary]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    axes[0].errorbar(counts, [r["mean_seconds"] for r in summary],
                     yerr=[r["std_seconds"] for r in summary], marker="o", capsize=3)
    axes[0].set(title="End-to-end pipeline time", ylabel="Seconds (mean +/- case std)")
    axes[1].plot(counts, [r["speedup"] for r in summary], "o-")
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[1].set(title="Paired speedup over no cache", ylabel="Speedup")
    axes[2].plot(counts, [r["mae_edit_gt"] for r in summary], "o-", label="Edit ROI vs GT")
    axes[2].plot(counts, [r["mae_edit_no_cache"] for r in summary], "s-", label="Edit ROI vs no cache")
    axes[2].set(title="Image difference (lower is smaller)", ylabel="MAE on RGB [0,1]")
    axes[2].legend()
    axes[3].plot(counts, [r["psnr_edit_gt"] if r["psnr_edit_gt"] is not None else np.nan for r in summary], "o-")
    axes[3].set(title="Edit ROI PSNR vs GT (higher is better)", ylabel="dB; exact match = infinity")
    for ax in axes:
        ax.set(xlabel="KV reuse forwards", xticks=counts)
        ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"speed_quality.{suffix}", dpi=160)
    plt.close(fig)
    columns = ("Reuse", "Fresh forwards", "Mode", "Seconds", "Speedup", "Images/s", "Edit PSNR", "Edit MAE")
    cells = [[r["reuse_count"], r["fresh_forwards"], "Approx. snapshot" if r["approximate_early_snapshot"] else r["cache_mode"],
              f'{r["mean_seconds"]:.3f}', f'{r["speedup"]:.3f}x', f'{r["images_per_second"]:.3f}',
              f'{r["psnr_edit_gt"]:.2f}' if r["psnr_edit_gt"] is not None else "inf", f'{r["mae_edit_gt"]:.6f}']
             for r in summary]
    fig, ax = plt.subplots(figsize=(14, 1.4 + .4 * len(summary)))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Image quality and speed by KV reuse count", pad=14)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"summary_table.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_report(output_dir, cases, plans, records, summary, metadata):
    def display(value, digits=6):
        return f"{value:.{digits}f}" if value is not None else "n/a"

    rows = "".join(
        f'<tr><td>{r["reuse_count"]}</td><td>{r["fresh_forwards"]}</td><td>{r["capture_step"]}</td>'
        f'<td>{r["cache_mode"]}</td><td>{r["mean_seconds"]:.3f} ± {r["std_seconds"]:.3f}</td>'
        f'<td>{r["speedup"]:.3f}x</td><td>{r["images_per_second"]:.3f}</td>'
        f'<td>{display(r["psnr_edit_gt"], 2) if r["psnr_edit_gt"] is not None else "∞"}</td>'
        f'<td>{r["mae_edit_gt"]:.6f}</td><td>{display(r["mae_outside_source"])}</td>'
        f'<td>{r["mae_edit_no_cache"]:.6f}</td></tr>'
        for r in summary
    )
    sections = []
    for case in cases:
        directory = f"sample_{case['sample_id']:05d}"
        case_records = [r for r in records if r["sample_id"] == case["sample_id"]]
        baseline_seconds = next(r["median_seconds"] for r in case_records if r["reuse_count"] == 0)
        details = "".join(
            f'<tr><td>{r["reuse_count"]}</td><td>{len(r["cache"]["reuse_steps"])}</td>'
            f'<td>{r["median_seconds"]:.3f}</td><td>{baseline_seconds / r["median_seconds"]:.3f}x</td>'
            f'<td>{r["metrics"]["mae_edit_gt"]:.6f}</td><td>{r["metrics"]["mae_edit_no_cache"]:.6f}</td></tr>'
            for r in case_records
        )
        sections.append(
            f'<section><h2>Sample {case["sample_id"]}</h2><p>{html.escape(case["instruction"])}</p>'
            f'<p>Seed: {metadata["seed"] + int(case["sample_id"])}</p>'
            '<table><tr><th>Requested reuse</th><th>Actual reuse</th><th>Seconds</th><th>Speedup</th>'
            '<th>Edit MAE vs GT</th><th>Edit MAE vs no cache</th></tr>' + details + '</table>'
            f'<a href="{directory}/comparison.png"><img src="{directory}/comparison.png"></a>'
            f'<details><summary>ROI crop and differences from no cache (gain {metadata["difference_gain"]:g})</summary>'
            f'<img src="{directory}/roi_comparison.png"><img src="{directory}/differences.png"></details></section>'
        )
    (output_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>KV reuse ablation</title>'
        '<style>body{font-family:sans-serif;margin:24px}img{max-width:100%}section{margin-top:32px}'
        'td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
        '<h1>Text/reference KV reuse ablation</h1>'
        f'<p>Checkpoint: {html.escape(metadata["checkpoint"])}. Split: {html.escape(cases[0]["split"])}. '
        f'{metadata["num_inference_steps"]} forwards; trained phase threshold '
        f'{metadata["condition_cache_threshold"]:g} held fixed across every variant, including no cache.</p>'
        '<p>Early-snapshot variants freeze context from a high-noise forward. Its text/reference states can depend '
        'on evolving target tokens and time conditioning, so early reuse is approximate. The snapshot is not '
        'refreshed at the low-noise transition. Saved attention thresholds are never raised to simulate longer reuse.</p>'
        '<p>Prefill is a full forward and is excluded from the reuse count. Seeds, inputs, masks, precision and '
        'sampling settings match. Timing includes text/VQ work and decoding, excludes model loading, disk writes '
        'and metrics; CUDA is synchronized. Each case uses the median of repeated timings; error bars show '
        'variation across cases. MAE and PSNR measure pixel fidelity, not semantic edit quality. '
        'Edit PSNR is computed from the mean per-case edit-region MSE; infinity means zero MSE.</p>'
        '<img src="cache_schedule.png"><img src="speed_quality.png">'
        '<p><a href="summary.csv">Download quality/speed CSV</a> · '
        '<a href="summary_table.png">Summary table image</a></p>'
        '<table><tr><th>Reuse steps</th><th>Fresh forwards</th><th>Capture index</th><th>Mode</th>'
        '<th>Seconds ± case std</th><th>Speedup</th><th>Images/s</th><th>Edit PSNR (dB) ↑</th>'
        '<th>Edit MAE vs GT ↓</th><th>Outside MAE vs source ↓</th><th>Edit MAE vs no cache ↓</th></tr>'
        + rows + '</table>' + "".join(sections),
        encoding="utf-8",
    )
    write_json(output_dir / "records.json", records)
    write_json(output_dir / "summary.json", {"metadata": metadata, "variants": summary})
    fields = ("reuse_count", "fresh_forwards", "capture_step", "capture_threshold", "cache_mode",
              "approximate_early_snapshot", "num_cases", "mean_seconds", "std_seconds", "speedup", "images_per_second",
              "psnr_gt", "psnr_edit_gt", "mae_gt", "mae_edit_gt", "mae_outside_source",
              "mae_no_cache", "mae_edit_no_cache", "changed_pixel_fraction")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, **{key: "inf" if row[key] is None else row[key]
                                    for key in ("psnr_gt", "psnr_edit_gt")}} for row in summary)


def main(args):
    from infer import resolve_checkpoint

    transformer_path, _, _ = resolve_checkpoint(args.checkpoint)
    config = json.loads((transformer_path / "config.json").read_text(encoding="utf-8"))
    if not config.get("cache_conditioning", False):
        raise ValueError("This experiment requires a cache_conditioning checkpoint with text/reference KV support")
    phase_threshold = args.condition_cache_threshold
    plans, native_maximum = build_cache_plan(args.num_inference_steps, phase_threshold, args.cache_steps,
                                            args.allow_early_cache)
    args.output_dir = args.output_dir.expanduser().resolve()
    if (args.output_dir / "records.json").exists():
        raise FileExistsError("This output directory already contains measurements; choose a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "checkpoint": str(args.checkpoint.expanduser().resolve()), "transformer_path": str(transformer_path),
                "condition_cache_threshold": phase_threshold,
                "checkpoint_condition_cache_threshold": config.get("condition_cache_threshold", .3),
                "maximum_native_reuse_steps": native_maximum,
                "maximum_reuse_steps": args.num_inference_steps - 1 if args.allow_early_cache else native_maximum,
                "psnr_definition": "-10*log10(mean per-case MSE); JSON null / CSV inf means zero MSE",
                "timing_scope": "end_to_end_pipeline_excluding_model_load_io_metrics", "status": "planned"}
    write_json(args.output_dir / "plan.json", {"metadata": metadata, "variants": plans})
    print(f"Native reuse limit: {native_maximum}; requested variants: {[p['reuse_count'] for p in plans]} "
          f"(early snapshots allowed: {args.allow_early_cache})", flush=True)
    if args.plan_only:
        return

    from infer_vanilla import prepare_cases

    cases = prepare_cases(args)
    if len({case["sample_id"] for case in cases}) != len(cases):
        raise ValueError("Ablation samples must have unique IDs")
    if any("mask" not in case or case["mask"].getbbox() is None for case in cases):
        raise ValueError("Every ablation sample must include a nonempty mask.png")
    if args.prepare_only:
        return

    from PIL import Image
    from src.editmgt import init_edit_mgt
    from src.train_validation import _stack_panels

    pipe = init_edit_mgt(args.device, enable_bf16=args.bf16, base_model_path=args.model_path,
                         transformer_path=str(transformer_path), condition_cache_threshold=phase_threshold)
    for module in (pipe.transformer, pipe.vqvae, pipe.text_encoder, pipe.text_encoder_t5):
        if module is not None:
            module.eval().requires_grad_(False)
    if not pipe.transformer.config.cache_conditioning or pipe.transformer.config.condition_cache_threshold != phase_threshold:
        raise ValueError("Loaded Transformer cache policy differs from the planned runtime phase")
    pipe.local_guidance = 0.0
    pipe.set_progress_bar_config(disable=True)
    for plan in plans:
        for _ in range(args.warmup):
            generate(pipe, cases[0], plan, args)
    metadata["status"] = "running"
    write_json(args.output_dir / "metadata.json", metadata)
    records, panels = [], []
    for case in cases:
        case_dir = args.output_dir / f"sample_{case['sample_id']:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        timings = {plan["key"]: [] for plan in plans}
        observations, execution_order = {}, []
        rng = random.Random(args.selection_seed + int(case["sample_id"]))
        for repeat in range(args.repeats):
            order = plans.copy()
            rng.shuffle(order)
            execution_order.append([p["key"] for p in order])
            for plan in order:
                generated, elapsed, info = generate(pipe, case, plan, args)
                timings[plan["key"]].append(elapsed)
                observations[plan["key"]] = info
                if repeat == 0:
                    generated.save(case_dir / f"{plan['key']}.png")
                print(f"Sample {case['sample_id']} | reuse={plan['reuse_count']} | repeat={repeat + 1} | {elapsed:.3f}s", flush=True)
        images = {}
        for plan in plans:
            with Image.open(case_dir / f"{plan['key']}.png") as generated:
                images[plan["key"]] = generated.convert("RGB")
        for plan in plans:
            records.append({
                "sample_id": case["sample_id"], "instruction": case["instruction"],
                "split": case["split"], "train_index": case.get("train_index"),
                "seed": args.seed + int(case["sample_id"]), "variant": plan["key"], "reuse_count": plan["reuse_count"],
                "cache_mode": plan["cache_mode"], "fresh_forwards": plan["fresh_forwards"],
                "approximate_early_snapshot": plan["approximate_early_snapshot"],
                "seconds": timings[plan["key"]], "median_seconds": statistics.median(timings[plan["key"]]),
                "execution_order": execution_order, "cache": observations[plan["key"]],
                "image": f"{case_dir.name}/{plan['key']}.png",
                "metrics": image_metrics(images[plan["key"]], images["reuse_00"], case["target"], case["source"], case["mask"]),
            })
        panels.append(comparison_images(case, images, plans, case_dir, args))
        write_json(args.output_dir / "records.json", records)
    _stack_panels(panels).save(args.output_dir / "contact_sheet.png")
    summary = aggregate(records, plans)
    plot_results(args.output_dir, plans, summary, args.num_inference_steps, phase_threshold)
    metadata["status"] = "complete"
    save_report(args.output_dir, cases, plans, records, summary, metadata)
    write_json(args.output_dir / "metadata.json", metadata)
    print(f"Saved ablation report: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main(parse_args())
