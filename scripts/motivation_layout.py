"""Publication figure for measured dense-MGM decoding trajectories.

This module renders the sampler's committed token state at six-step intervals.
Cells that remain masked are shown as solid gray, including in the magnified
edit-region crop. Numerical prediction statistics are exported separately.
"""

import json
from pathlib import Path
import textwrap


EDIT_COLOR = "#D55E00"
CONTEXT_COLOR = "#0072B2"
UNRESOLVED_GRAY = (166, 166, 166)


def apply_unresolved_gray(image, unresolved, gray=UNRESOLVED_GRAY):
    """Cover unresolved token cells with exact, opaque RGB gray pixel blocks.

    ``unresolved`` is the sampler's boolean [token_height, token_width] mask
    AFTER the specified decoding step. Nearest-neighbor expansion preserves
    discrete token boundaries and never blends a prediction into a masked cell.
    """
    import numpy as np
    from PIL import Image

    mask = np.asarray(unresolved)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError("Unresolved tokens must have a nonempty [height,width] mask")
    if len(gray) != 3 or any(type(channel) is not int or not 0 <= channel <= 255 for channel in gray):
        raise ValueError("Gray must be an RGB tuple with integer values in [0,255]")
    result = image.convert("RGB")
    token_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    pixel_mask = np.asarray(token_mask.resize(result.size, Image.Resampling.NEAREST)) > 0
    pixels = np.asarray(result).copy()
    pixels[pixel_mask] = gray
    return Image.fromarray(pixels, mode="RGB")


def _artifact(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes its run directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_measurements(root, samples):
    import numpy as np

    per_sample = []
    steps = None
    for sample in samples:
        with np.load(_artifact(root, sample["trajectory"]), allow_pickle=False) as archive:
            required = {"entropy", "argmax", "unresolved", "token_mask", "commitment_step"}
            if not required.issubset(archive.files):
                raise ValueError(f"Missing trajectory arrays for {sample['key']}: {required - set(archive.files)}")
            entropy = archive["entropy"].astype(np.float64)
            predictions = archive["argmax"]
            unresolved = archive["unresolved"].astype(bool)
            mask = archive["token_mask"].astype(bool)
            commitment = archive["commitment_step"]
        if entropy.ndim != 3 or predictions.shape != entropy.shape or unresolved.shape != entropy.shape:
            raise ValueError(f"Expected matching [steps,height,width] trajectories for {sample['key']}")
        if mask.shape != entropy.shape[1:] or commitment.shape != mask.shape:
            raise ValueError(f"Token mask/commitment shape mismatch for {sample['key']}")
        if not mask.any() or mask.all():
            raise ValueError(f"Both edit and context tokens are required for {sample['key']}")
        if not np.isfinite(entropy).all() or (entropy < -1e-5).any():
            raise ValueError(f"Invalid predictive entropy for {sample['key']}")
        if steps is None:
            steps = len(entropy)
        elif steps != len(entropy):
            raise ValueError("All aggregated trajectories must use the same number of decoding steps")
        if (commitment < 1).any() or (commitment > steps).any():
            raise ValueError(f"Commitment steps must be in [1,{steps}] for {sample['key']}")
        different = predictions != predictions[-1]
        values = {"key": sample["key"], "edit_tokens": int(mask.sum()),
                  "context_tokens": int((~mask).sum())}
        for region, selected in (("edited", mask), ("context", ~mask)):
            values[region] = {
                "entropy_nats": entropy[:, selected].mean(1),
                "argmax_disagreement": different[:, selected].mean(1),
                "unresolved_fraction": unresolved[:, selected].mean(1),
            }
        per_sample.append(values)

    curves = {}
    for region in ("edited", "context"):
        curves[region] = {}
        for metric in ("entropy_nats", "argmax_disagreement", "unresolved_fraction"):
            data = np.stack([row[region][metric] for row in per_sample])
            sem = data.std(0, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else np.zeros(steps)
            curves[region][metric] = {"mean": data.mean(0).tolist(), "sem": sem.tolist()}
    serializable = [dict(row, **{
        region: {name: value.tolist() for name, value in row[region].items()}
        for region in ("edited", "context")}) for row in per_sample]
    return steps, curves, serializable


def _display_steps(total, available, requested):
    if requested is not None:
        selected = list(requested)
        if not selected or len(set(selected)) != len(selected) or selected != sorted(selected):
            raise ValueError("Selected steps must be unique, nonempty and increasing")
    else:
        selected = sorted(set(range(6, total + 1, 6)) | {total})
    if any(type(step) is not int or not 1 <= step <= total for step in selected):
        raise ValueError(f"Selected steps must be integers in [1,{total}]")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Missing saved token-state frames for steps {missing}; available steps: "
                         f"{sorted(available)}. Collect snapshots at every sixth step, or specify "
                         "--display-steps using saved frames.")
    return selected


def render_figure(run_dir, output_prefix, sample_key=None, selected_steps=None, dpi=300):
    """Render source plus committed-token snapshots, with unresolved cells gray.

    Returns PNG/PDF/SVG, numeric data and a caption. The primary figure shows
    one example at every sixth decoding step and the same magnified ROI below.
    Numeric regional statistics still aggregate all examples in the run.
    """
    import os
    import tempfile

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "editmgt-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    from PIL import Image

    root, prefix = Path(run_dir).resolve(), Path(output_prefix).resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    samples = metadata.get("samples", [])
    if not samples:
        raise ValueError("No recorded samples to render")
    if len({sample["key"] for sample in samples}) != len(samples):
        raise ValueError("Recorded sample keys must be unique")
    chosen_key = sample_key or metadata.get("default_visual_sample") or samples[0]["key"]
    sample = next((sample for sample in samples if sample["key"] == chosen_key), None)
    if sample is None:
        raise ValueError(f"Unknown visual sample {chosen_key!r}")
    total, curves, sample_data = _load_measurements(root, samples)
    if total < 2:
        raise ValueError("The decoding motivation figure requires at least two steps")
    frames = {int(step): value for step, value in sample["frames"].items()}
    if not frames:
        raise ValueError(f"No saved intermediate frames for {chosen_key}")
    requested = selected_steps
    if requested is None:
        requested = metadata.get("protocol", {}).get("captured_image_steps")
    shown = _display_steps(total, frames, requested)
    with Image.open(_artifact(root, sample["source"])) as image:
        source = np.asarray(image.convert("RGB"))
    with Image.open(_artifact(root, sample["mask"])) as image:
        mask_image = image.convert("L").resize((source.shape[1], source.shape[0]), Image.Resampling.NEAREST)
        mask_values = np.asarray(mask_image)
        pixel_mask = mask_values >= (1 if mask_values.max() <= 1 else 128)
    if not pixel_mask.any():
        raise ValueError(f"The visual example {chosen_key} has an empty mask")
    with np.load(_artifact(root, sample["trajectory"]), allow_pickle=False) as trajectory:
        unresolved = trajectory["unresolved"].astype(bool)
        token_mask = trajectory["token_mask"].astype(bool)
    images = [source]
    for step in shown:
        with Image.open(_artifact(root, frames[step])) as image:
            image = image.convert("RGB").resize((source.shape[1], source.shape[0]), Image.Resampling.NEAREST)
            images.append(np.asarray(apply_unresolved_gray(image, unresolved[step - 1])))
    snapshot_data = [{"step": step,
                      "committed_fraction": float((~unresolved[step - 1]).mean()),
                      "edited_committed_fraction": float((~unresolved[step - 1])[token_mask].mean()),
                      "context_committed_fraction": float((~unresolved[step - 1])[~token_mask].mean())}
                     for step in shown]

    yy, xx = np.nonzero(pixel_mask)
    side = max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1)
    side = min(max(round(side * 1.35), round(min(source.shape[:2]) * 0.20)), min(source.shape[:2]))
    crop_width = min(round(side * 2.5), source.shape[1])
    left = int(np.clip(round((xx.min() + xx.max() - crop_width) / 2), 0, source.shape[1] - crop_width))
    top = int(np.clip(round((yy.min() + yy.max() - side) / 2), 0, source.shape[0] - side))

    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
             "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
             "axes.edgecolor": "#9DA6B1", "axes.linewidth": 0.65,
             "savefig.facecolor": "white", "figure.facecolor": "white"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13.0, 4.5), layout=None)
        grid = fig.add_gridspec(2, 1, left=0.032, right=0.986, bottom=0.165,
                               top=0.79, height_ratios=[1.6, 0.75], hspace=0.33)
        upper = grid[0].subgridspec(1, len(images), wspace=0.055)
        detail = grid[1].subgridspec(1, len(images), wspace=0.055)
        labels = ["Source + edit mask"] + [f"Step {step}" for step in shown]
        for column, (image, label) in enumerate(zip(images, labels)):
            ax = fig.add_subplot(upper[0, column])
            ax.imshow(image, interpolation="nearest")
            ax.set_title(label, pad=5, fontsize=9, fontweight="medium")
            if column == 0:
                overlay = np.zeros((*pixel_mask.shape, 4))
                overlay[pixel_mask] = (*matplotlib.colors.to_rgb(EDIT_COLOR), 0.22)
                ax.imshow(overlay)
                padded = np.pad(pixel_mask, 1)
                ax.contour(np.arange(-1, pixel_mask.shape[1] + 1),
                           np.arange(-1, pixel_mask.shape[0] + 1), padded,
                           levels=[0.5], colors=[EDIT_COLOR], linewidths=0.85)
                ax.add_patch(Rectangle((left, top), crop_width, side, fill=False,
                                       edgecolor="white", linewidth=1.1, linestyle=(0, (3, 2))))
                ax.set_xlim(-0.5, source.shape[1] - 0.5)
                ax.set_ylim(source.shape[0] - 0.5, -0.5)
            ax.set_axis_off()
            if column:
                values = snapshot_data[column - 1]
                ax.text(0.5, -0.055, f'{values["committed_fraction"]:.0%} tokens committed',
                        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#56606D")
            crop_ax = fig.add_subplot(detail[0, column])
            crop_ax.imshow(image[top:top + side, left:left + crop_width], interpolation="nearest", aspect="equal")
            crop_ax.set_xticks([]); crop_ax.set_yticks([])
            for spine in crop_ax.spines.values():
                spine.set_visible(True); spine.set_color(EDIT_COLOR); spine.set_linewidth(0.8)
            if column == 0:
                crop_ax.text(-0.03, 0.5, "Edit-region\ndetail", transform=crop_ax.transAxes,
                             ha="right", va="center", rotation=90, fontsize=7.5, color="#56606D")

        protocol = metadata.get("protocol", {})
        model_label = protocol.get("model_label", "Dense EditMGT")
        dataset = protocol.get("dataset_label")
        heading = f"{model_label}  ·  {total} decoding steps"
        if dataset:
            heading += f"  ·  {dataset}"
        fig.text(0.032, 0.975, heading,
                 ha="left", va="top", fontsize=10, fontweight="medium", color="#18212B")
        instruction = textwrap.fill(textwrap.shorten(
            f'Instruction: “{sample["instruction"]}”', width=260, placeholder="…"), width=145)
        fig.text(0.032, 0.925, instruction, ha="left", va="top", fontsize=9.5, color="#303A47")
        fig.add_artist(Rectangle((0.032, 0.073), 0.012, 0.025, transform=fig.transFigure,
                                 facecolor=np.array(UNRESOLVED_GRAY) / 255, edgecolor="#8A8A8A", linewidth=0.5))
        fig.text(0.050, 0.084, "Uncommitted token", fontsize=8, color="#596574", va="center")
        fig.add_artist(Rectangle((0.235, 0.073), 0.012, 0.025, transform=fig.transFigure,
                                 fill=False, edgecolor=EDIT_COLOR, linewidth=1.0))
        fig.text(0.253, 0.084, "User-specified edit region", fontsize=8, color="#596574", va="center")
        fig.text(0.032, 0.030,
                 "Each snapshot shows tokens committed after that decoding step. The mask annotates the edit region only.",
                 fontsize=7.8, color="#596574", va="bottom")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paths = {}
        for suffix in ("png", "pdf", "svg"):
            path = Path(f"{prefix}.{suffix}")
            fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=None)
            paths[suffix] = str(path)
        plt.close(fig)

    data_path = Path(f"{prefix}.figure_data.json")
    data = {"sample_count": len(samples), "visual_sample": chosen_key,
            "visual_instruction": sample["instruction"], "displayed_steps": shown,
            "snapshots": snapshot_data, "unresolved_gray_rgb": list(UNRESOLVED_GRAY),
            "frame_definition": "Actual post-step sampler state: committed token cells are decoded; currently masked cells are opaque gray",
            "steps": list(range(1, total + 1)), "aggregation": "Equal weight per image; mean over each image's region tokens, then image mean; SEM across images",
            "entropy_definition": "-sum_v p_t(v) log p_t(v), in nats; p_t is float32 softmax of full-grid post-CFG logits before scheduler freezing",
            "stability_definition": "mean_i 1[argmax_v p_t(v|i) != argmax_v p_T(v|i)] within each fixed mask partition; final-step reference, not ground truth",
            "commitment_definition": "First decoding step after which the sampler leaves a token unmasked; a sampling decision, not an independent uncertainty measurement",
            "curves": curves, "per_sample": sample_data, "protocol": metadata.get("protocol", {})}
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    caption_path = Path(f"{prefix}.caption.txt")
    caption = (f"Token prediction order during {total}-step dense masked-generative image editing. "
               "The source image and user-specified edit mask appear on the left; snapshots after decoding steps "
               + ", ".join(map(str, shown)) + " appear from left to right. "
               "Each snapshot shows the sampler state after confidence-based re-masking: committed tokens are "
               "visualized by VQ decoding, while token cells that remain masked are covered by solid gray. "
               "The narrow row shows the same magnified edit-region crop in every column. Percentages report "
               "the fraction of all target tokens committed by that step. The annotation mask is used for analysis "
               "and visualization only; it is not passed to the generative model or sampler. "
               "All target tokens are initially masked, so no source-token locking forces the observed order.\n")
    caption_path.write_text(caption, encoding="utf-8")
    paths.update(data=str(data_path), caption=str(caption_path))
    return paths
