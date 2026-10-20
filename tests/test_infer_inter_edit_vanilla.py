import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from infer_vanilla import load_cases, main, parse_args
from test_infer_inter_edit_train import arguments as train_arguments, train_data


class FakeVanillaPipeline:
    _execution_device = torch.device("cpu")

    def __init__(self, **config):
        self.transformer = torch.nn.Linear(1, 1)
        self.transformer.config = SimpleNamespace(**{
            "sparse_target": False, "token_type_rope": False,
            "cache_conditioning": False, "time_conditioning": "legacy", **config,
        })
        self.vqvae = self.text_encoder = self.text_encoder_t5 = None
        self.scheduler = SimpleNamespace(inference_shift=1.0)
        self.calls = []

    def __call__(self, **kwargs):
        assert not torch.is_grad_enabled()
        assert "mask_image" not in kwargs and "target" not in kwargs
        self.calls.append(kwargs)
        return SimpleNamespace(images=[kwargs["reference_image"][0].copy()])


def arguments(train_data, *extra):
    root, _, output, cache = train_data
    return parse_args([
        "--data-dir", str(root.parent), "--output-dir", str(output),
        "--tar-cache-dir", str(cache), "--resolution", "32", *extra,
    ])


def test_vanilla_uses_train_selection_and_reports_without_mask_conditioning(train_data):
    args = arguments(train_data, "--num-samples", "2", "--seed", "7", "--bf16")
    pipe = FakeVanillaPipeline()
    with patch("src.editmgt.init_edit_mgt", return_value=pipe) as initialize:
        main(args)
    assert initialize.call_args.kwargs == {
        "device": "cuda:0", "enable_bf16": True,
        "base_model_path": args.model_path, "transformer_path": None,
    }
    assert not pipe.transformer.training and not pipe.transformer.weight.requires_grad
    assert [call["generator"].initial_seed() for call in pipe.calls] == [109, 110]
    for call in pipe.calls:
        assert call["num_inference_steps"] == 36 and call["reference_strength"] == 1.1
        for flag in ("edit_only_diffusion", "sparse_target", "use_reference_kv_cache",
                     "use_condition_kv_cache", "preserve_unmasked_pixels"):
            assert call[flag] is False
        assert call["reference_image"][0].getpixel((0, 0))[0] != 200
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    assert metadata["split"] == "train" and not metadata["mask_input"]
    assert [case["train_index"] for case in metadata["cases"]] == [2, 3]
    assert metadata["output_compositing"] == "none"
    page = (args.output_dir / "index.html").read_text()
    assert "Inter-Edit train" in page and "display only" in page
    assert "&lt;script&gt;" in page and "<script>" not in page
    for case in metadata["cases"]:
        for name in ("generated", "comparison"):
            assert (args.output_dir / case["output_dir"] / f"{name}.png").is_file()
        assert (args.output_dir / case["input_dir"] / "mask_overlay.png").is_file()
    with Image.open(args.output_dir / "contact_sheet.png") as sheet:
        assert sheet.width == 4 * 32


def test_vanilla_reuses_trained_manifest_indices_order_prompts_and_seeds(train_data):
    from infer import prepare_cases

    trained_args = train_arguments(train_data, "--sample-indices", "4", "0", "--language", "all")
    prepare_cases(trained_args)
    args = arguments(train_data, "--samples-manifest", str(trained_args.output_dir / "samples.json"),
                     "--output-dir", str(trained_args.output_dir / "vanilla"), "--seed", "42")
    pipe = FakeVanillaPipeline()
    with patch("src.inter_edit_dataset.InterEditReader", side_effect=AssertionError("Reuse exported PNGs")), \
            patch("src.editmgt.init_edit_mgt", return_value=pipe):
        main(args)
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    assert [case["train_index"] for case in metadata["cases"]] == [4, 0]
    assert [call["generator"].initial_seed() for call in pipe.calls] == [146, 142]
    assert pipe.calls[1]["prompt"] == ["添加物体"]
    assert "添加物体" in (args.output_dir / "index.html").read_text()
    assert metadata["split"] == "train" and metadata["selection"] == "manifest"


def test_legacy_test_manifest_without_masks_still_loads(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("source", "target"):
        Image.new("RGB", (8, 8), (20, 30, 40)).save(inputs / f"{name}.png")
    manifest = tmp_path / "samples.json"
    manifest.write_text(json.dumps([{
        "sample_id": 7, "test_index": 2, "instruction": "添加物体", "instruction_en": "Add an object.",
        "edit_type": "Add", "input_dir": "inputs",
    }]))
    cases = load_cases(manifest, 32)
    assert cases[0]["split"] == "test" and cases[0]["test_index"] == 2
    assert cases[0]["instruction"] == "Add an object."
    assert "mask" not in cases[0]
    args = parse_args(["--samples-manifest", str(manifest), "--resolution", "32",
                       "--output-dir", str(tmp_path / "results")])
    with patch("src.editmgt.init_edit_mgt", return_value=FakeVanillaPipeline()):
        main(args)
    assert "Inter-Edit test" in (args.output_dir / "index.html").read_text()


def test_prepare_only_exports_train_inputs_without_loading_weights(train_data):
    args = arguments(train_data, "--prepare-only", "--sample-indices", "5", "2")
    with patch("src.editmgt.init_edit_mgt", side_effect=AssertionError("Do not load model weights")):
        main(args)
    assert (args.output_dir / "train_inputs.png").is_file()
    assert not (args.output_dir / "metadata.json").exists()


@pytest.mark.parametrize("config", [
    {"sparse_target": True}, {"token_type_rope": True},
    {"cache_conditioning": True}, {"time_conditioning": "mask_ratio"},
])
def test_vanilla_rejects_finetuned_protocols(train_data, config):
    args = arguments(train_data, "--num-samples", "1", "--checkpoint", str(train_data[1]))
    pipe = FakeVanillaPipeline(**config)
    with patch("src.editmgt.init_edit_mgt", return_value=pipe) as initialize:
        with pytest.raises(ValueError, match="original EditMGT"):
            main(args)
    assert initialize.call_args.kwargs["transformer_path"] == str(train_data[1] / "transformer")
    assert not pipe.calls
