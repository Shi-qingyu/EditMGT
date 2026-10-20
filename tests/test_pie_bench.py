"""PIE annotation/metric regression cases; no model weights or CUDA required."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.benchmark_data import BenchmarkImages
from src.benchmark_metrics import parse_metrics
from src.pie_bench_data import decode_pie_mask, select_pie_cases
from src.pie_bench_metrics import DEFAULT_PIE_METRICS, PIEBenchMetrics, pie_metric_summary


def test_zero_based_rle_uses_row_major_runs_and_official_boundary_fix():
    mask = np.asarray(decode_pie_mask([5, 2, 10, 1], (4, 4), fix_boundary=False))
    assert np.argwhere(mask).tolist() == [[1, 1], [1, 2], [2, 2]]
    corrected = np.asarray(decode_pie_mask([5, 2, 10, 1], (4, 4)))
    assert corrected[0].all() and corrected[-1].all()
    assert corrected[:, 0].all() and corrected[:, -1].all()
    assert corrected[2, 1] == 0
    with pytest.raises(ValueError, match="pairs"):
        decode_pie_mask([3], (4, 4))
    with pytest.raises(ValueError, match="Invalid"):
        decode_pie_mask([-1, 2], (4, 4))
    images = BenchmarkImages()
    try:
        restored = images.edit_mask({"kind": "pie_rle", "runs": [5, 2, 10, 1], "shape": [4, 4]}, "pie_bench")
        np.testing.assert_array_equal(restored, corrected)
    finally:
        images.close()


def test_selection_keeps_ids_and_uses_instructions_separately_from_metric_captions(tmp_path):
    payload = {
        "000000000001": {"image_path": "6_change_color/one.jpg", "original_prompt": "a [red] car",
                         "editing_prompt": "a [blue] car", "editing_instruction": "Make the car blue.",
                         "editing_type_id": "6", "mask": [513, 4]},
        "000000000002": {"image_path": "2_add_object/two.jpg", "original_prompt": "a field",
                         "editing_prompt": "a field with a [tree]", "editing_instruction": "Add a tree.",
                         "editing_type_id": "2", "mask": [1024, 10]},
    }
    (tmp_path / "mapping_file.json").write_text(json.dumps(payload))
    args = SimpleNamespace(data_dir=tmp_path, annotation_file=None, images_dir=None,
                           edit_category_list=["6"], sample_ids=None, limit=None,
                           prompt_field="editing_instruction")
    selection, dataset = select_pie_cases(args)
    assert len(selection) == 1 and selection[0]["sample_id"] == "000000000001"
    assert selection[0]["instruction"] == "Make the car blue."
    assert selection[0]["caption"] == "a blue car" and selection[0]["source_caption"] == "a red car"
    assert "target" not in selection[0]  # The original image must never be labeled edited GT.
    assert dataset["total_samples"] == 2 and not dataset["full_benchmark"]
    args.sample_ids = ["000000000002"]
    with pytest.raises(ValueError, match="outside"):
        select_pie_cases(args)
    args.sample_ids, args.prompt_field = None, "editing_prompt"
    assert select_pie_cases(args)[0][0]["instruction"] == "a blue car"


def test_pie_pixel_metrics_zero_excluded_pixels_without_region_area_renormalization():
    source = Image.new("RGB", (4, 4), "black")
    prediction = np.zeros((4, 4, 3), dtype=np.uint8)
    prediction[1:3, 1:3] = 255
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    suite = PIEBenchMetrics(["mse", "mse_edit_part", "mse_unedit_part", "psnr", "psnr_unedit_part"],
                            device="cpu", metric_resolution=4)
    scores = suite.score(Image.fromarray(prediction), source, Image.fromarray(mask))
    assert scores["mse"] == scores["mse_edit_part"] == 0.25
    assert scores["mse_unedit_part"] == 0
    assert scores["psnr"] == pytest.approx(-10 * math.log10(0.25))
    assert scores["psnr_unedit_part"] == "inf"
    all_edit = suite.score(Image.fromarray(prediction), source, Image.new("L", (4, 4), 255))
    assert all_edit["mse_unedit_part"] is None and all_edit["psnr_unedit_part"] is None


def test_pie_summary_preserves_infinite_psnr_and_counts_undefined_regions():
    summary = pie_metric_summary([
        {"metrics": {"psnr_unedit_part": "inf", "mse_unedit_part": 0}},
        {"metrics": {"psnr_unedit_part": 20.0, "mse_unedit_part": 0.1}},
        {"metrics": {"psnr_unedit_part": None, "mse_unedit_part": None}},
    ])
    assert summary["psnr_unedit_part"] == {"mean": "inf", "std": None, "count": 2, "undefined_count": 1}
    assert summary["mse_unedit_part"]["mean"] == 0.05
    json.dumps(summary, allow_nan=False)
    assert parse_metrics(",".join(DEFAULT_PIE_METRICS), "pie_bench") == DEFAULT_PIE_METRICS
    with pytest.raises(ValueError):
        parse_metrics("alpha-in", "pie_bench")
