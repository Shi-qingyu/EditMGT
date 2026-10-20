import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from tests.ablate_cache_steps import (
    aggregate, build_cache_plan, check_cache_info, image_metrics, main, parse_args,
)
from src.mask_schedule import scheduled_mask_ratio
from test_infer_inter_edit_train import train_data


def test_exact_reuse_counts_exclude_prefill_and_keep_trained_phase():
    plans, maximum = build_cache_plan(18, .3, allow_early_cache=False)
    assert maximum == 2
    assert [p["reuse_count"] for p in plans] == [0, 1, 2]
    assert plans[1]["capture_step"] == 16 and plans[1]["reuse_steps"] == [17]
    assert plans[2]["capture_step"] == 15 and plans[2]["reuse_steps"] == [16, 17]
    for plan in plans[1:]:
        capture = next(i for i in range(17) if scheduled_mask_ratio(i, 18, "cosine") <= plan["capture_threshold"])
        assert capture == plan["capture_step"]
        assert plan["capture_threshold"] <= .3
    with pytest.raises(ValueError, match="maximum native KV reuse count is 2"):
        build_cache_plan(18, .3, [0, 4], allow_early_cache=False)
    with pytest.raises(ValueError, match="unique"):
        build_cache_plan(18, .3, [0, 1, 1])
    assert build_cache_plan(1, .3)[0][0]["reuse_steps"] == []
    plans, maximum = build_cache_plan(36, .3, [2, 4, 5])
    assert maximum == 5 and plans[-1]["reuse_count"] == 0
    assert next(p for p in plans if p["reuse_count"] == 5)["reuse_steps"] == [31, 32, 33, 34, 35]


def test_default_sweep_matches_17_8_3_and_zero_without_changing_the_phase():
    plans, native_maximum = build_cache_plan(18, .3)
    assert native_maximum == 2
    assert [p["reuse_count"] for p in plans] == [17, 8, 3, 0]
    assert [p["capture_step"] for p in plans] == [0, 9, 14, None]
    assert [p["fresh_forwards"] for p in plans] == [1, 10, 15, 18]
    assert [p["approximate_early_snapshot"] for p in plans] == [True, True, True, False]
    assert plans[0]["capture_threshold"] == 1.0
    for plan in plans:
        assert len(plan["reuse_steps"]) == plan["reuse_count"]
    with pytest.raises(ValueError, match="integers in"):
        build_cache_plan(18, .3, [18])


def test_actual_cache_mismatch_fails_instead_of_reporting_a_false_ablation():
    plans, _ = build_cache_plan(18, .3, [0, 1, 2])
    pipe = SimpleNamespace(last_condition_cache_info={"enabled": True, "capture_step": 15, "reuse_steps": [16, 17]})
    assert check_cache_info(pipe, plans[2])["reuse_steps"] == [16, 17]
    with pytest.raises(RuntimeError, match="differs from the plan"):
        check_cache_info(pipe, plans[1])


def test_image_metrics_respect_roi_and_measure_changes_from_no_cache():
    source = Image.new("RGB", (2, 2), "black")
    target = Image.new("RGB", (2, 2), "white")
    generated = source.copy()
    generated.putpixel((0, 0), (255, 255, 255))
    mask = Image.new("L", (2, 2), 0)
    mask.putpixel((0, 0), 255)
    metrics = image_metrics(generated, source, target, source, mask)
    assert metrics["mae_gt"] == .75
    assert metrics["mae_edit_gt"] == metrics["mae_outside_source"] == 0
    assert metrics["mae_no_cache"] == metrics["changed_pixel_fraction"] == .25
    assert metrics["mae_edit_no_cache"] == 1


def test_speedup_pairs_the_same_cases_and_uses_median_case_times():
    plans, _ = build_cache_plan(18, .3, [0, 2])
    metrics = {k: 0.0 for k in ("mae_gt", "mae_edit_gt", "mae_no_cache", "mae_edit_no_cache", "changed_pixel_fraction")}
    metrics.update(mse_gt=.25, mse_edit_gt=.25, mae_outside_source=0.0)
    records = [{"variant": p["key"], "reuse_count": p["reuse_count"], "median_seconds": time,
                "metrics": metrics} for p, times in zip(plans, ((10, 20), (5, 10))) for time in times]
    summary = aggregate(records, plans)
    assert summary[0]["speedup"] == 1 and summary[1]["speedup"] == 2
    assert summary[1]["mean_seconds"] == 7.5
    assert summary[1]["psnr_edit_gt"] == pytest.approx(6.020599913)


def arguments(train_data, *extra):
    root, checkpoint, output, cache = train_data
    (checkpoint / "transformer/config.json").write_text(json.dumps({
        "cache_conditioning": True, "condition_cache_threshold": .3,
    }))
    return parse_args([
        "--checkpoint", str(checkpoint), "--data-dir", str(root.parent),
        "--output-dir", str(output), "--tar-cache-dir", str(cache), "--resolution", "32",
        "--num-samples", "2", "--device", "cpu", "--tile-size", "32", *extra,
    ])


def test_plan_only_reads_no_images_or_model_weights(train_data):
    args = arguments(train_data, "--plan-only")
    with patch("src.inter_edit_dataset.InterEditReader", side_effect=AssertionError("No images")), \
            patch("src.editmgt.init_edit_mgt", side_effect=AssertionError("No model")):
        main(args)
    plan = json.loads((args.output_dir / "plan.json").read_text())
    assert plan["metadata"]["maximum_reuse_steps"] == 17
    assert plan["metadata"]["maximum_native_reuse_steps"] == 9
    assert plan["metadata"]["condition_cache_threshold"] == .8
    assert plan["metadata"]["checkpoint_condition_cache_threshold"] == .3
    assert not (args.output_dir / "samples.json").exists()


def test_fake_ablation_holds_inputs_seeds_policy_fixed_and_writes_reports(train_data):
    class Pipeline:
        _execution_device = torch.device("cpu")

        def __init__(self):
            self.transformer = torch.nn.Linear(1, 1)
            self.transformer.config = SimpleNamespace(sparse_target=True, cache_conditioning=True,
                                                      condition_cache_threshold=.8)
            self.vqvae = self.text_encoder = self.text_encoder_t5 = None
            self.calls = []

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert "target" not in kwargs and "output" not in kwargs
            assert self.transformer.config.condition_cache_threshold == .8
            self.calls.append(kwargs)
            enabled = kwargs["use_condition_kv_cache"]
            reuse_count = kwargs["condition_kv_cache_reuse_steps"]
            capture = 18 - reuse_count - 1 if enabled else None
            approximate = bool(enabled and scheduled_mask_ratio(capture, 18, "cosine") > .8)
            assert kwargs["allow_early_condition_cache"] == approximate
            self.last_condition_cache_info = {
                "enabled": enabled, "capture_step": capture,
                "reuse_steps": list(range(capture + 1, 18)) if capture is not None else [],
                "approximate_early_snapshot": approximate,
            }
            pixels = np.asarray(kwargs["reference_image"][0]).copy()
            pixels[8:16, 8:16] = (100 + (capture or 0), 120, 140)
            return SimpleNamespace(images=[Image.fromarray(pixels)])

    args = arguments(train_data, "--repeats", "2", "--warmup", "1")
    pipe = Pipeline()
    # This test exercises orchestration and image reporting, with no model inference.
    with patch("src.editmgt.init_edit_mgt", return_value=pipe) as initialize, \
            patch("ablate_cache_steps.plot_results") as plot:
        main(args)
    initialize.assert_called_once()
    assert len(pipe.calls) == 4 + 2 * 4 * 2  # four warmups, two cases, four variants, two repeats
    for call in pipe.calls:
        source_id = call["reference_image"][0].getpixel((0, 0))[0]
        assert call["generator"].initial_seed() == 142 + source_id
        assert call["mask_image"].getbbox() is not None
        assert not call["preserve_unmasked_pixels"] and call["sparse_target_halo"] == 0
    plot.assert_called_once()
    records = json.loads((args.output_dir / "records.json").read_text())
    assert len(records) == 8
    for record in records:
        assert len(record["seconds"]) == 2
        assert len(record["cache"]["reuse_steps"]) == record["reuse_count"]
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["metadata"]["status"] == "complete"
    assert next(v for v in summary["variants"] if v["reuse_count"] == 0)["speedup"] == 1
    assert [v["reuse_count"] for v in summary["variants"]] == [17, 8, 3, 0]
    assert (args.output_dir / "summary.csv").is_file()
    assert (args.output_dir / "contact_sheet.png").is_file()
    for sample_id in (102, 103):
        directory = args.output_dir / f"sample_{sample_id:05d}"
        for image in ("comparison", "roi_comparison", "differences", "reuse_17", "reuse_08", "reuse_03", "reuse_00"):
            assert (directory / f"{image}.png").is_file()
    page = (args.output_dir / "index.html").read_text()
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert "phase threshold 0.3 held fixed" in page
    with pytest.raises(FileExistsError, match="new --output-dir"):
        main(args)
