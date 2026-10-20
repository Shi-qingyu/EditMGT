import io
import json
import random
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.train_validation import (
    prepare_inter_edit_validation_cases, run_inter_edit_validation,
    select_validation_indices, synchronized_main_process_call, validation_model_state,
)


@pytest.fixture
def validation_data(tmp_path):
    root = tmp_path / "Inter-Edit" / "test"
    root.mkdir(parents=True)
    rows, members = [], {"source_images.tar": {}, "target_images.tar": {}, "masks.tar": {}}
    for index, category in enumerate(("Local", "Add", "Local", "Remove", "Texture", "Text Editing")):
        row = {
            "index": index, "source_path": f"source_images/{index:05d}.png",
            "gt_path": f"target_images/{index:05d}.png", "mask_path": f"masks/{index:05d}.png",
            "instruction": f"中文指令 {index} <script>test</script>", "edit_type": category, "language": "中文",
        }
        rows.append(row)
        mask = Image.new("L", (8, 4), 0)
        mask.paste(255, (1, 1, 3, 3))
        for archive, name, image in (
            ("source_images.tar", row["source_path"], Image.new("RGB", (16, 8), (20 + index, 40, 60))),
            ("target_images.tar", row["gt_path"], Image.new("RGB", (12, 6), (100 + index, 120, 140))),
            ("masks.tar", row["mask_path"], mask),
        ):
            payload = io.BytesIO()
            image.save(payload, "PNG")
            members[archive][name] = payload.getvalue()
    (root / "metadata.json").write_text(json.dumps(rows, ensure_ascii=False))
    for archive, images in members.items():
        with tarfile.open(root / archive, "w") as tar:
            for name, payload in images.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return root.parent, tmp_path / "validation", tmp_path / "tar_cache", rows


def test_fixed_validation_selection_covers_edit_types_and_honors_explicit_indices(validation_data):
    _, _, _, rows = validation_data
    assert select_validation_indices(rows) == [0, 1, 3, 4, 5]
    assert select_validation_indices(rows, sample_indices=[5, 0]) == [5, 0]
    assert select_validation_indices(rows, num_images=6) == [0, 1, 3, 4, 5, 2]
    with pytest.raises(ValueError):
        select_validation_indices(rows, sample_indices=[0, 0])
    with pytest.raises(IndexError):
        select_validation_indices(rows, sample_indices=[6])


def test_validation_inputs_are_cached_with_same_geometry_and_original_metadata(validation_data):
    root, output, cache, _ = validation_data
    cases = prepare_inter_edit_validation_cases(root, output, 16, sample_indices=[1, 4], cache_dir=cache)
    manifest = json.loads((output / "samples.json").read_text())
    assert [case["sample_id"] for case in cases] == [1, 4]
    assert manifest[0]["source_original_size"] == [16, 8]
    assert manifest[0]["mask_original_size"] == [8, 4]
    assert (output / "test_inputs.png").is_file()
    for case in cases:
        assert case["source"].size == case["target"].size == case["mask"].size == (16, 16)
        assert set(np.asarray(case["mask"]).flatten()) == {0, 255}
        assert case["overlay"].getpixel((0, 0)) == case["source"].getpixel((0, 0))
        assert case["overlay"].getpixel((3, 5)) != case["source"].getpixel((3, 5))
        for name in ("source", "target", "mask", "mask_overlay"):
            assert (output / case["input_dir"] / f"{name}.png").is_file()


def test_validation_can_explicitly_include_translated_chinese_test_cases(validation_data, tmp_path):
    from test_inter_edit_dataset import write_official_test_translations

    root, output, cache, original = validation_data
    directory = write_official_test_translations(root, tmp_path / "translated-test")
    cases = prepare_inter_edit_validation_cases(root, output, 16, cache_dir=cache,
                                                language="en", translations_dir=directory)
    assert [case["sample_id"] for case in cases] == [0, 1, 3, 4, 5]
    assert all(case["instruction"].startswith("Add translated object ") for case in cases)
    assert cases[0]["instruction_original"] == original[0]["instruction"]
    selection = json.loads((output / "selection.json").read_text())
    assert selection["eligible_samples"] == len(original)
    assert selection["instruction_variant"] == "translated_english"


class FakePipeline:
    _execution_device = torch.device("cpu")

    def __init__(self):
        self.calls = []
        self.scheduler = SimpleNamespace(config=SimpleNamespace(masking_schedule="cosine"), inference_shift=1.0)

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        color = tuple(torch.randint(0, 256, (3,), generator=kwargs["generator"]).tolist())
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), color)])


def test_validation_reports_use_fixed_seeds_sparse_flags_and_no_target_conditioning(validation_data):
    root, output, cache, _ = validation_data
    cases = prepare_inter_edit_validation_cases(root, output, 16, num_images=2, cache_dir=cache)
    pipe = FakePipeline()
    run_inter_edit_validation(pipe, cases, output, 0, seed=100)
    report = run_inter_edit_validation(pipe, cases, output, 200, seed=100)
    assert report["contact_sheet"].width == 64
    for case in cases:
        name = f"sample_{case['sample_id']:05d}"
        with Image.open(output / "step_000000" / name / "generated.png") as initial:
            with Image.open(output / "step_000200" / name / "generated.png") as later:
                assert initial.tobytes() == later.tobytes()
        assert (output / "step_000200" / name / "comparison.png").is_file()
    for call in pipe.calls:
        assert call["sparse_target"] and call["edit_only_diffusion"]
        assert call["lora_part_enable"] and call["sparse_target_halo"] == 0
        assert not call["preserve_unmasked_pixels"]
        assert "target" not in call and "output" not in call
    metadata = json.loads((report["step_dir"] / "metadata.json").read_text())
    assert metadata["split"] == "test"
    assert not metadata["preserve_unmasked_pixels"] and metadata["output_compositing"] == "none"
    assert [case["seed"] for case in metadata["cases"]] == [100, 101]
    assert "step_000000" in (output / "index.html").read_text()
    page = (report["step_dir"] / "index.html").read_text()
    assert "中文指令" in page and "&lt;script&gt;" in page and "<script>" not in page


@pytest.mark.parametrize("raise_error", [False, True])
def test_eval_ema_and_random_states_are_restored_even_after_generation_failure(raise_error):
    model = torch.nn.Linear(3, 2).train()
    original_strength = torch.tensor([0.9])
    model.reference_strength_factor = original_strength
    weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    class EMA:
        def store(self, parameters):
            self.original = [parameter.detach().clone() for parameter in parameters]

        def copy_to(self, parameters):
            for parameter in parameters:
                parameter.data.fill_(100)

        def restore(self, parameters):
            for parameter, original in zip(parameters, self.original):
                parameter.data.copy_(original)

    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        with validation_model_state(model, EMA()):
            assert not model.training and not torch.is_grad_enabled()
            assert model.weight.eq(100).all()
            model.reference_strength_factor = torch.tensor([1.5])
            random.random()
            np.random.rand()
            torch.rand(2)
            if raise_error:
                raise ValueError("generation failed")
    except ValueError:
        assert raise_error
    assert model.training
    assert model.reference_strength_factor is original_strength
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
        assert parameter.grad is None
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_distributed_validation_pauses_ranks_and_propagates_main_process_failure():
    class Accelerator:
        is_main_process = True
        device = torch.device("cpu")

        def __init__(self):
            self.waits = 0

        def wait_for_everyone(self):
            self.waits += 1

    accelerator = Accelerator()
    with patch("torch.distributed.is_initialized", return_value=False):
        assert synchronized_main_process_call(accelerator, lambda: 123) == 123
    assert accelerator.waits == 2
    accelerator.is_main_process = False
    with patch("torch.distributed.is_initialized", return_value=True), patch(
        "torch.distributed.broadcast", side_effect=lambda flag, src: flag.fill_(1)
    ):
        with pytest.raises(RuntimeError, match="failed on the main process"):
            synchronized_main_process_call(accelerator, lambda: pytest.fail("Non-main rank must not generate"))


def test_visualization_runs_real_sparse_pipeline_and_decodes_images(validation_data):
    from diffusers import VQModel
    from src.pipeline import Pipeline
    from src.scheduler import Scheduler
    from src.transformer import Transformer2DModel

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root, output, cache, _ = validation_data
        cases = prepare_inter_edit_validation_cases(root, output, 16, num_images=1, cache_dir=cache)
        model = Transformer2DModel(
            num_layers=1, num_single_layers=1, attention_head_dim=12, num_attention_heads=2,
            joint_attention_dim=24, pooled_projection_dim=8, axes_dims_rope=(4, 4, 4),
            vocab_size=32, codebook_size=31, downsample=True, upsample=True, sparse_target=True,
        ).train()
        vq = VQModel(block_out_channels=(16,), latent_channels=4, vq_embed_dim=4,
                     num_vq_embeddings=31, norm_num_groups=4, lookup_from_codebook=True).eval()
        pipe = Pipeline(vq, None, None, model, Scheduler(mask_token_id=31), None, None)

        class EmbeddedPipeline:
            _execution_device = torch.device("cpu")

            def __getattr__(self, name):
                return getattr(pipe, name)

            def set_progress_bar_config(self, **kwargs):
                pipe.set_progress_bar_config(**kwargs)

            def __call__(self, **kwargs):
                kwargs.pop("prompt")
                return pipe(prompt_embeds=torch.zeros(1, 8), encoder_hidden_states=torch.zeros(1, 3, 24),
                            negative_prompt_embeds=torch.zeros(1, 8), negative_encoder_hidden_states=torch.zeros(1, 3, 24),
                            **kwargs)

        with validation_model_state(model):
            report = run_inter_edit_validation(EmbeddedPipeline(), cases, output, 200, num_inference_steps=2)
        assert model.training
        with Image.open(report["step_dir"] / "sample_00000" / "generated.png") as generated:
            assert generated.size == (16, 16) and generated.mode == "RGB"
        metadata = json.loads((report["step_dir"] / "metadata.json").read_text())
        assert not metadata["preserve_unmasked_pixels"] and metadata["output_compositing"] == "none"
        assert metadata["cases"][0]["edit_region"]["output_compositing"] == "none"
    finally:
        torch.set_num_threads(previous_threads)
