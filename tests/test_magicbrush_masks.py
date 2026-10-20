"""MagicBrush alpha decoding through training, generation, and compositing."""
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from eval.benchmark_magicbrush import parse_args
from src import benchmark_utils
from src.benchmark_data import BenchmarkImages, path_spec
from src.benchmark_utils import BenchmarkRun, InferenceRunner, generate_benchmark, generation_config
from src.magicbrush_dataset import MagicBrushDataset
from src.mask_utils import MAGICBRUSH_MASK_DECODING, decode_magicbrush_edit_mask, normalize_edit_mask_image


def _png(image):
    handle = io.BytesIO()
    image.save(handle, format="PNG")
    return handle.getvalue()


def _cutout():
    # Deliberately bright RGB outside the edit and dark RGB inside. A luma
    # conversion selects exactly the opposite of the intended transparent ROI.
    mask = Image.new("RGBA", (32, 32), (240, 220, 200, 255))
    mask.paste((0, 0, 0, 0), (8, 8, 24, 24))
    return mask


def test_magicbrush_cutout_uses_alpha_while_generic_masks_keep_their_convention():
    cutout = _cutout()
    decoded = decode_magicbrush_edit_mask(cutout)
    assert decoded.mode == "L" and decoded.getbbox() == (8, 8, 24, 24)
    assert np.count_nonzero(decoded) == 16 * 16
    assert decoded.getpixel((16, 16)) == 255 and decoded.getpixel((0, 0)) == 0
    generic = normalize_edit_mask_image(cutout)
    assert generic.getpixel((16, 16)) == 0 and generic.getpixel((0, 0)) == 255


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize("positive", [1, 255])
def test_binary_magicbrush_exports_keep_white_as_edit(mode, positive):
    binary = Image.new("L", (8, 8), 0)
    binary.paste(positive, (2, 2, 6, 6))
    decoded = decode_magicbrush_edit_mask(binary.convert(mode))
    assert decoded.getbbox() == (2, 2, 6, 6)
    assert set(np.unique(np.asarray(decoded))) == {0, 255}


def test_opaque_cutout_is_empty_and_transparent_cutout_is_full():
    assert decode_magicbrush_edit_mask(Image.new("RGBA", (8, 8), (255, 255, 255, 255))).getbbox() is None
    assert decode_magicbrush_edit_mask(Image.new("RGBA", (8, 8), (0, 0, 0, 0))).getbbox() == (0, 0, 8, 8)


@pytest.fixture
def magicbrush_cutout_data(tmp_path):
    root = tmp_path / "magicbrush"
    (root / "data").mkdir(parents=True)
    source = Image.new("RGB", (32, 32), (30, 80, 130))
    target = Image.new("RGB", (32, 32), (200, 20, 30))
    path = root / "data/train-00000.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "img_id": "7730", "turn_index": 1, "instruction": "Make the suitcase red.",
        "source_img": {"bytes": _png(source)}, "target_img": {"bytes": _png(target)},
        "mask_img": {"bytes": _png(_cutout())},
    }]), path)
    mask_path = root / "mask.png"
    _cutout().save(mask_path)

    def spec(column):
        return {"kind": "parquet", "root": str(root), "file": str(path), "row": 0, "column": column}

    return root, path, mask_path, source, spec


def test_benchmark_file_and_parquet_mask_readers_preserve_alpha_until_decoding(magicbrush_cutout_data):
    _, _, mask_path, source, spec = magicbrush_cutout_data
    images = BenchmarkImages()
    try:
        for locator in (path_spec(mask_path), spec("mask_img")):
            assert images.image(locator).mode == "RGBA"
            decoded = images.edit_mask(locator, "magicbrush")
            assert decoded.mode == "L" and decoded.getbbox() == (8, 8, 24, 24)
            assert np.count_nonzero(decoded) == 256
        assert images.image(spec("source_img"), "RGB").tobytes() == source.tobytes()
    finally:
        images.close()


def test_training_receives_the_same_white_edit_region(magicbrush_cutout_data):
    root, _, _, _, _ = magicbrush_cutout_data
    dataset = MagicBrushDataset(root, size=32, tokenizer=None, shuffle=False, repeat=False)
    samples = list(dataset)
    assert len(samples) == 1
    mask = samples[0]["edit_mask"][0]
    expected = torch.zeros(32, 32)
    expected[8:24, 8:24] = 1
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("preserve_rgb", [False, True])
def test_decoded_mask_reaches_pipeline_and_controls_native_pixel_compositing(
    magicbrush_cutout_data, tmp_path, monkeypatch, preserve_rgb,
):
    _, _, _, source, spec = magicbrush_cutout_data
    monkeypatch.setattr(sys, "argv", [
        "benchmark_magicbrush.py", "--vanilla", "--stage", "generate", "--device", "cpu",
        "--resolution", "64", "--warmup", "0", "--output-dir", str(tmp_path / "output"),
        *(["--preserve-unmasked-pixels"] if preserve_rgb else []),
    ])
    args = parse_args()
    calls = []

    class CapturePipeline:
        _execution_device = torch.device("cpu")
        transformer = SimpleNamespace(config=SimpleNamespace(
            sparse_target=True, sparse_target_halo=0, supervise_target_halo=False,
        ))

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (64, 64), (255, 0, 0))])

    class Runner(InferenceRunner):
        def __init__(self, worker_args):
            self.args = worker_args
            self.pipe = CapturePipeline()
            self.metadata = {"device_name": "CPU test"}
            self.warmed_up = True

        def close(self):
            pass

    monkeypatch.setattr(benchmark_utils, "InferenceRunner", Runner)
    task = {
        "key": "7730__1", "seed_key": "magicbrush:7730:1", "session_id": "7730",
        "instruction": "Make the suitcase red.", "prediction": "images/7730/7730_1.png",
        "source": spec("source_img"), "target": spec("target_img"), "mask": spec("mask_img"),
        "groups": ["single_turn_all"],
    }
    run = BenchmarkRun(args, "magicbrush", [task], {"split": "dev"})
    images = BenchmarkImages()
    try:
        generate_benchmark(args, run, images)
        assert len(calls) == 1
        pipeline_mask = calls[0]["mask_image"]
        assert pipeline_mask.mode == "L" and pipeline_mask.size == (64, 64)
        assert pipeline_mask.getbbox() == (16, 16, 48, 48)
        assert calls[0]["edit_only_diffusion"] and calls[0]["sparse_target"]
        assert calls[0]["preserve_unmasked_pixels"] == preserve_rgb
        with Image.open(run.image_path(task)) as generated:
            pixels = np.asarray(generated)
        roi = np.asarray(decode_magicbrush_edit_mask(_cutout())) > 0
        if preserve_rgb:
            assert np.array_equal(pixels[~roi], np.asarray(source)[~roi])
        else:
            assert np.all(pixels[~roi] == (255, 0, 0))
        assert np.all(pixels[roi] == (255, 0, 0))
        record = run.record(task["key"])
        assert record["edit_mask_decoding"] == MAGICBRUSH_MASK_DECODING
        assert record["input_mask"]["edit_fraction"] == 0.25
        assert record["input_mask"]["native_edit_bbox"] == [8, 8, 24, 24]
        assert record["preserve_unmasked_pixels"] == preserve_rgb
        assert record["output_compositing"] == (
            "original_pixel_mask_at_generation_and_native_resolution" if preserve_rgb else "none"
        )
        assert run.info["settings"]["generation"]["edit_mask_decoding"] == MAGICBRUSH_MASK_DECODING
    finally:
        images.close()
        run.close()

    # A compositing-policy change must not silently reuse earlier predictions.
    args.resume = True
    args.preserve_unmasked_pixels = not preserve_rgb
    with pytest.raises(ValueError, match="Output already exists"):
        BenchmarkRun(args, "magicbrush", [task], {"split": "dev"})
    args.preserve_unmasked_pixels = preserve_rgb

    # Old masked outputs cannot be silently reused after the decoder changes.
    saved = json.loads((args.output_dir / "run.json").read_text())
    del saved["settings"]["generation"]["edit_mask_decoding"]
    (args.output_dir / "run.json").write_text(json.dumps(saved))
    args.resume = True
    with pytest.raises(ValueError, match="Output already exists"):
        BenchmarkRun(args, "magicbrush", [task], {"split": "dev"})
    # Mask-free baselines remain compatible with their original settings.
    args.mask_mode = "none"
    assert "edit_mask_decoding" not in generation_config(args, "magicbrush")
