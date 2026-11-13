#!/usr/bin/env python3
"""Export a publication figure from captured vanilla EditMGT decoding traces."""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


CONTEXT = "#0072B2"
EDIT = "#D55E00"
INK = "#17212B"
MUTED = "#687482"


def load_curves(root):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    samples = metadata["samples"]
    selected = metadata["selection"]["selected_keys"]
    if len(samples) != len(selected) or len(samples) != 32:
        raise ValueError("Expected all 32 fixed-selection Inter-Edit traces")
    protocol = metadata["protocol"]
    if protocol["model_protocol"] != "native_editmgt_mask_free_no_cache":
        raise ValueError("Expected vanilla mask-free EditMGT traces")
    steps = protocol["num_inference_steps"]
    edit, context, delays = [], [], []
    for sample in samples:
        with np.load(root / sample["trajectory"], allow_pickle=False) as archive:
            mask = archive["token_mask"].astype(bool)
            committed = archive["commitment_step"]
            unresolved = archive["unresolved"].astype(bool)
        if (committed.shape != mask.shape or unresolved.shape != (steps, *mask.shape)
                or not mask.any() or mask.all() or unresolved[-1].any()
                or not np.array_equal(committed, np.argmax(~unresolved, axis=0) + 1)):
            raise ValueError(f"Invalid trace for {sample['key']}")
        edit.append(np.array([(committed[mask] <= step).mean() for step in range(1, steps + 1)]))
        context.append(np.array([(committed[~mask] <= step).mean() for step in range(1, steps + 1)]))
        delays.append(float(committed[mask].mean() - committed[~mask].mean()))
    return metadata, np.stack(edit), np.stack(context), np.asarray(delays)


def bootstrap_curve(values, indices):
    draws = values[indices].mean(axis=1)
    return values.mean(axis=0), np.quantile(draws, [0.025, 0.975], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("outputs/vanilla_decode_order_quant_32"))
    parser.add_argument("--output-prefix", type=Path,
                        default=Path("docs/figures/vanilla_decode_order_quantitative"))
    args = parser.parse_args()
    root = args.input_dir.resolve()
    prefix = args.output_prefix.resolve()
    metadata, edit, context, delays = load_curves(root)
    summary = json.loads((root / "decode_order_summary.json").read_text(encoding="utf-8"))
    if (summary["sample_count"] != len(delays)
            or not np.isclose(summary["statistics"]["delta_steps"]["mean"], delays.mean())):
        raise ValueError("The saved statistical summary does not match these trajectories")
    steps = edit.shape[1]
    if steps != 36:
        raise ValueError("Expected the released 36-step decoding protocol")
    non_tied = delays[delays != 0]
    positive = int((non_tied > 0).sum())
    sign_p = sum(comb(len(non_tied), k) for k in range(positive, len(non_tied) + 1)) / 2 ** len(non_tied)
    rng = np.random.default_rng(2026)
    bootstrap_indices = rng.integers(0, len(delays), size=(10000, len(delays)))
    edit_mean, edit_ci = bootstrap_curve(edit, bootstrap_indices)
    context_mean, context_ci = bootstrap_curve(context, bootstrap_indices)
    delay_ci = np.asarray(summary["statistics"]["delta_steps"]["ci95"])
    gap30 = (context[:, 29] - edit[:, 29]) * 100
    gap30_ci = np.quantile(gap30[bootstrap_indices].mean(axis=1), [0.025, 0.975])
    data = {
        "sample_count": len(delays), "selection_seed": metadata["selection"]["selection_seed"],
        "steps": list(range(1, steps + 1)),
        "edit_committed_mean": edit_mean.tolist(), "edit_committed_ci95": edit_ci.tolist(),
        "context_committed_mean": context_mean.tolist(), "context_committed_ci95": context_ci.tolist(),
        "per_sample_delay_steps": dict(zip([s["key"] for s in metadata["samples"]],
                                           delays.tolist())),
        "mean_delay_steps": float(delays.mean()), "mean_delay_ci95": delay_ci.tolist(),
        "positive_image_count": positive, "non_tied_image_count": len(non_tied),
        "one_sided_sign_p": sign_p,
        "step30_gap_percentage_points": float(gap30.mean()),
        "step30_gap_ci95": gap30_ci.tolist(),
        "aggregation": "Equal image weight; 10,000 image bootstrap resamples, seed 2026",
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.titlesize": 9, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#7E8994", "axes.linewidth": 0.7,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.1, 2.9),
                                     gridspec_kw={"width_ratios": [1.34, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.89, wspace=0.27)
    x = np.arange(1, steps + 1)
    left.fill_between(x, context_ci[0] * 100, context_ci[1] * 100,
                      color=CONTEXT, alpha=0.14, linewidth=0)
    left.fill_between(x, edit_ci[0] * 100, edit_ci[1] * 100,
                      color=EDIT, alpha=0.16, linewidth=0)
    left.plot(x, context_mean * 100, color=CONTEXT, lw=2.0, label="Non-edit region")
    left.plot(x, edit_mean * 100, color=EDIT, lw=2.0, ls="--", label="Edit region")
    left.axvline(30, color=MUTED, lw=0.75, ls=":", alpha=0.85)
    left.scatter([30, 30], [context_mean[29] * 100, edit_mean[29] * 100],
                 s=23, color=[CONTEXT, EDIT], edgecolor="white", linewidth=0.5, zorder=4)
    left.annotate("", xy=(31.1, context_mean[29] * 100),
                  xytext=(31.1, edit_mean[29] * 100),
                  arrowprops={"arrowstyle": "<->", "lw": 0.95, "color": INK})
    left.text(32.1, 64.0, f"{gap30.mean():.1f} pp", va="center", ha="left",
              color=INK, fontweight="semibold", fontsize=8)
    left.set(xlim=(1, 36), ylim=(0, 101), xticks=[1, 6, 12, 18, 24, 30, 36],
             yticks=[0, 20, 40, 60, 80, 100], xlabel="Decoding step",
             ylabel="Tokens committed (%)")
    left.grid(axis="y", color="#D9E0E5", lw=0.5, alpha=0.7)
    left.set_axisbelow(True)
    left.legend(loc="upper left", frameon=False, fontsize=7.6, handlelength=2.2)
    left.set_title("a  Non-edit tokens commit earlier", loc="left", color=INK,
                   fontweight="semibold", pad=9)

    order = np.argsort(delays)
    sorted_delays = delays[order]
    yy = np.arange(1, len(delays) + 1)
    colors = np.where(sorted_delays > 0, EDIT, "#9BA6B1")
    right.axvline(0, color="#7E8994", lw=0.9, zorder=0)
    for y, value, color in zip(yy, sorted_delays, colors):
        right.plot([0, value], [y, y], color=color, alpha=0.53, lw=0.85, zorder=1)
    right.scatter(sorted_delays, yy, c=colors, s=13, linewidth=0, zorder=2)
    summary_y = 34.2
    right.plot([delay_ci[0], delay_ci[1]], [summary_y, summary_y],
               color=INK, lw=2.4, solid_capstyle="round")
    right.scatter([delays.mean()], [summary_y], marker="D", s=27, color=INK, zorder=3)
    right.text(-4.9, summary_y, f"Mean +{delays.mean():.2f}",
               ha="left", va="center", fontsize=7.5,
               color=INK, fontweight="semibold")
    right.set(xlim=(-5.2, 13.2), ylim=(0, 37.2), xticks=[-4, 0, 4, 8, 12],
              yticks=[], xlabel="Edit − non-edit commitment (steps)")
    right.grid(axis="x", color="#D9E0E5", lw=0.5, alpha=0.7)
    right.set_axisbelow(True)
    right.set_title("b  Per-image delay", loc="left", color=INK,
                    fontweight="semibold", pad=9)
    fig.text(0.075, 0.026,
             "Annotation mask used for analysis only. Bands: 95% image-bootstrap intervals.",
             fontsize=7.2, color=MUTED, ha="left", va="bottom")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(prefix.with_suffix(f".{suffix}"), dpi=450)
    plt.close(fig)
    print(f"Saved {prefix}.pdf, .svg, .png, and .json")


if __name__ == "__main__":
    main()
