#!/usr/bin/env python3
"""Summarize saved vanilla EditMGT token commitment order on Inter-Edit.

The annotation mask is used only to partition tokens after generation. Each
image, rather than each token, is one observation for uncertainty estimates.
"""

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np


def mean_ci(values, rng, repeats):
    values = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(repeats, len(values)))
    boot = values[draws].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])]}


def auc_from_commitments(steps, edit_mask, total_steps):
    """P(edit committed later than context) + half the tied pairs."""
    edit = np.bincount(steps[edit_mask], minlength=total_steps + 1)[1:]
    context = np.bincount(steps[~edit_mask], minlength=total_steps + 1)[1:]
    context_before = np.cumsum(context) - context
    favorable = np.sum(edit * (context_before + 0.5 * context))
    return float(favorable / (edit.sum() * context.sum()))


def load_sample(root, sample, expected_steps):
    path = root / sample["trajectory"]
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        mask = archive["token_mask"].astype(bool)
        commitment = archive["commitment_step"].astype(np.int32)
        unresolved = archive["unresolved"].astype(bool)
        entropy = archive["entropy"].astype(np.float64)
    if (mask.shape != commitment.shape or unresolved.shape != (expected_steps, *mask.shape)
            or entropy.shape != unresolved.shape or not mask.any() or mask.all()):
        raise ValueError(f"Invalid trajectory shapes or edit mask: {sample['key']}")
    if (commitment < 1).any() or (commitment > expected_steps).any():
        raise ValueError(f"Invalid commitment steps: {sample['key']}")
    if unresolved[-1].any() or (unresolved[1:] & ~unresolved[:-1]).any():
        raise ValueError(f"Incomplete or nonmonotonic trajectory: {sample['key']}")
    if not np.array_equal(commitment, np.argmax(~unresolved, axis=0) + 1):
        raise ValueError(f"Commitment and sampler state disagree: {sample['key']}")
    edited, context = commitment[mask], commitment[~mask]
    row = {
        "key": sample["key"], "edit_type": sample.get("edit_type"),
        "edit_tokens": int(mask.sum()), "context_tokens": int((~mask).sum()),
        "edit_fraction": float(mask.mean()),
        "mean_edit_step": float(edited.mean()),
        "mean_context_step": float(context.mean()),
        "delta_steps": float(edited.mean() - context.mean()),
        "late_auc": auc_from_commitments(commitment, mask, expected_steps),
        "first_step_edit_entropy": float(entropy[0, mask].mean()),
        "first_step_context_entropy": float(entropy[0, ~mask].mean()),
        "first_step_entropy_delta": float(entropy[0, mask].mean() - entropy[0, ~mask].mean()),
    }
    for step in (6, 12, 18, 24, 30, 36):
        if step <= expected_steps:
            row[f"edit_by_{step}"] = float((edited <= step).mean())
            row[f"context_by_{step}"] = float((context <= step).mean())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("outputs/vanilla_decode_order_quant_32"))
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("--bootstrap-repeats must be at least 100")
    root = args.input_dir.resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    samples = metadata["samples"]
    protocol = metadata["protocol"]
    if protocol["model_protocol"] != "native_editmgt_mask_free_no_cache":
        raise ValueError("Expected the native mask-free vanilla EditMGT protocol")
    if metadata["selection"].get("benchmark") != "Inter-Edit":
        raise ValueError("Expected Inter-Edit examples")
    selected_keys = metadata["selection"]["selected_keys"]
    if len(samples) + len(metadata.get("skipped", [])) != len(selected_keys):
        raise ValueError("Capture is incomplete; wait for all selected samples")
    steps = int(protocol["num_inference_steps"])
    rows = [load_sample(root, sample, steps) for sample in samples]
    rng = np.random.default_rng(args.bootstrap_seed)
    fields = ("edit_fraction", "mean_edit_step", "mean_context_step", "delta_steps",
              "late_auc", "first_step_edit_entropy", "first_step_context_entropy",
              "first_step_entropy_delta")
    stats = {field: mean_ci([row[field] for row in rows], rng, args.bootstrap_repeats)
             for field in fields}
    for step in (6, 12, 18, 24, 30, 36):
        if step <= steps:
            for region in ("edit", "context"):
                field = f"{region}_by_{step}"
                stats[field] = mean_ci([row[field] for row in rows], rng,
                                       args.bootstrap_repeats)
    non_tied = [row["delta_steps"] for row in rows if row["delta_steps"] != 0]
    positive = sum(value > 0 for value in non_tied)
    sign_p = sum(comb(len(non_tied), k) for k in range(positive, len(non_tied) + 1)) / 2 ** len(non_tied)
    subgroups = {}
    for label, group in (("text_editing", [row for row in rows if row["edit_type"] == "Text Editing"]),
                         ("other_edits", [row for row in rows if row["edit_type"] != "Text Editing"])):
        if group:
            subgroups[label] = {
                "count": len(group),
                "mean_delta_steps": float(np.mean([row["delta_steps"] for row in group])),
                "mean_late_auc": float(np.mean([row["late_auc"] for row in group])),
                "positive_delta_count": sum(row["delta_steps"] > 0 for row in group),
            }
    report = {
        "sample_count": len(rows), "selected_count": len(selected_keys),
        "skipped": metadata.get("skipped", []), "selection": metadata["selection"],
        "protocol": {key: protocol[key] for key in
                     ("model_protocol", "resolution", "num_inference_steps", "guidance_scale",
                      "reference_strength", "mask_role", "source_token_lock", "sparse_target",
                      "reference_kv_cache", "condition_kv_cache")},
        "definitions": {
            "commitment_step": "First 1-based sampler step after which a token remains unmasked",
            "late_auc": "P(edit token committed later than context token) + 0.5 * P(tie), within image",
            "aggregation": "Equal weight per image; percentile bootstrap 95% CI over images",
            "mask": "Inter-Edit annotation projected by nearest neighbor to 64x64 token grid; analysis only",
            "sign_test": "One-sided exact image-level sign test for delta_steps > 0; exploratory",
        },
        "statistics": stats,
        "positive_delta_samples": positive,
        "non_tied_samples": len(non_tied),
        "one_sided_sign_p": sign_p,
        "subgroups": subgroups,
        "sample_keys": [row["key"] for row in rows],
    }
    (root / "decode_order_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (root / "decode_order_per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def cell(field, percent=False):
        item = stats[field]
        scale = 100 if percent else 1
        return f"{item['mean'] * scale:.2f} [{item['ci95'][0] * scale:.2f}, {item['ci95'][1] * scale:.2f}]"
    lines = [
        "# Vanilla EditMGT decoding order on Inter-Edit test/en",
        "",
        f"Fixed hash selection of **{len(selected_keys)}** images ({len(rows)} usable), "
        f"selection seed {metadata['selection']['selection_seed']}; "
        f"{steps} decoding steps, 1024px, CFG {protocol['guidance_scale']}, "
        f"reference strength {protocol['reference_strength']}. The edit mask was used only for analysis.",
        "",
        "| Image-level metric | Mean [95% bootstrap CI] |",
        "|---|---:|",
        f"| Edit-region mean commitment step | {cell('mean_edit_step')} |",
        f"| Context mean commitment step | {cell('mean_context_step')} |",
        f"| Paired difference (edit − context), steps | {cell('delta_steps')} |",
        f"| Late-order AUC (random-order reference: 0.5) | {cell('late_auc')} |",
        f"| First-step entropy difference (edit − context), nats | {cell('first_step_entropy_delta')} |",
        f"| Edit mask share of tokens | {cell('edit_fraction', True)}% |",
        "",
        "| After step | Edit tokens committed, % [CI] | Context tokens committed, % [CI] |",
        "|---:|---:|---:|",
    ]
    for step in (6, 12, 18, 24, 30, 36):
        if step <= steps:
            lines.append(f"| {step} | {cell(f'edit_by_{step}', True)} | "
                         f"{cell(f'context_by_{step}', True)} |")
    lines += [
        "",
        f"Edit tokens committed later on {positive}/{len(non_tied)} non-tied images "
        f"(one-sided exact sign-test p={sign_p:.4g}).",
        "",
        "| Exploratory subgroup | Images | Mean step difference | Mean late-order AUC | Positive images |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, group in subgroups.items():
        lines.append(f"| {label.replace('_', ' ')} | {group['count']} | "
                     f"{group['mean_delta_steps']:.2f} | {group['mean_late_auc']:.2f} | "
                     f"{group['positive_delta_count']}/{group['count']} |")
    lines += [
        "",
        "Commitment time is a sampler decision. The association with the annotated edit region "
        "does not establish why the model chooses this order. This small sample is exploratory; "
        "the 95% intervals resample images rather than treating tokens as independent observations.",
        "",
        "Raw inputs: `metadata.json` and each `samples/*/trajectory.npz`. "
        "Machine-readable results: `decode_order_summary.json`, `decode_order_per_sample.csv`.",
        "",
    ]
    (root / "decode_order_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
