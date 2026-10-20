import gzip
import io
import json
import random
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from infer import iter_train_rows, main, parse_args, prepare_cases, resolve_checkpoint
from src.inter_edit_dataset import InterEditReader


@pytest.fixture
def train_data(tmp_path):
    root = tmp_path / "Inter-Edit" / "train"
    (root / "metadata").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "num_samples": 6, "samples_per_shard": 3, "num_sample_shards": 2,
    }))
    # Deliberately omit test/, so every test also checks train-only access.
    for shard in range(2):
        rows = []
        with tarfile.open(root / f"images-{shard}.tar", "w") as archive:
            for index in range(shard * 3, shard * 3 + 3):
                row = {
                    "sample_id": 100 + index, "edit_type": "Add", "source_archive": f"images-{shard}.tar",
                    "asset_archive": f"images-{shard}.tar", "source_file": f"source_{index}.png",
                    "target_file": f"target_{index}.png", "mask_file": f"mask_{index}.png",
                    "instruction": "添加物体" if index == 0 else f"Add object {index} <script>x</script>",
                }
                rows.append(row)
                mask = Image.new("L", (8, 4), 0)
                if index != 1:  # First English example has an empty mask.
                    mask.paste(1, (2, 1, 4, 3))
                for field, image in (
                    ("source_file", Image.new("RGB", (16, 8), (index, 20, 30))),
                    ("target_file", Image.new("RGB", (16, 8), (200, index, 100))),
                    ("mask_file", mask),
                ):
                    payload = io.BytesIO()
                    image.save(payload, "PNG")
                    entry = tarfile.TarInfo(row[field])
                    entry.size = len(payload.getvalue())
                    archive.addfile(entry, io.BytesIO(payload.getvalue()))
        with gzip.open(root / "metadata" / f"train-{shard:05d}.jsonl.gz", "wt") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)
    checkpoint = tmp_path / "checkpoint-200" / "transformer"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    return root, checkpoint.parent, tmp_path / "outputs", tmp_path / "cache"


def arguments(train_data, *extra):
    root, checkpoint, output, cache = train_data
    return parse_args([
        "--checkpoint", str(checkpoint), "--data-dir", str(root.parent),
        "--output-dir", str(output), "--tar-cache-dir", str(cache), "--resolution", "32", *extra,
    ])


def test_explicit_global_indices_preserve_order_and_read_only_needed_shards(train_data):
    root, _, _, cache = train_data
    reader = InterEditReader(root, "train", cache)
    with patch.object(reader, "read_metadata", wraps=reader.read_metadata) as read:
        rows = list(iter_train_rows(reader, [5, 3, 4]))
    assert [row[0] for row in rows] == [5, 3, 4]
    assert [row[3]["sample_id"] for row in rows] == [105, 103, 104]
    assert read.call_count == 1
    for indices, error in (([0, 0], ValueError), ([-1], IndexError), ([6], IndexError)):
        with pytest.raises(error):
            list(iter_train_rows(reader, indices))


def test_random_selection_is_reproducible_and_keeps_original_indices(train_data):
    root, _, _, cache = train_data
    reader = InterEditReader(root, "train", cache)
    state = random.getstate()
    first = list(iter_train_rows(reader, selection="random", selection_seed=17))
    second = list(iter_train_rows(reader, selection="random", selection_seed=17))
    assert first == second
    assert sorted(row[0] for row in first) == list(range(6))
    assert all(row[3]["sample_id"] == 100 + row[0] for row in first)
    assert random.getstate() == state


def test_selection_skips_empty_masks_and_exports_binary_aligned_train_inputs(train_data):
    args = arguments(train_data, "--num-samples", "2")
    cases = prepare_cases(args)
    assert [case["train_index"] for case in cases] == [2, 3]
    for case in cases:
        assert case["source"].size == case["mask"].size == case["target"].size == (32, 32)
        assert set(case["mask"].tobytes()) == {0, 255}
        assert case["source"].getpixel((0, 0))[0] == case["train_index"]
        assert case["target"].getpixel((0, 0))[0] == 200
        for name in ("source", "target", "mask", "mask_overlay"):
            assert (args.output_dir / case["input_dir"] / f"{name}.png").is_file()
    selection = json.loads((args.output_dir / "selection.json").read_text())
    assert selection["split"] == "train" and selection["skipped_empty_indices"] == [1]
    assert (args.output_dir / "train_inputs.png").is_file()
    with pytest.raises(ValueError, match="does not match"):
        prepare_cases(arguments(train_data, "--sample-indices", "0"))
    with pytest.raises(ValueError, match="empty edit mask"):
        prepare_cases(arguments(train_data, "--sample-indices", "1"))


def test_checkpoint_root_and_transformer_directory_resolve_same_weights(train_data):
    _, checkpoint, _, _ = train_data
    expected = (checkpoint / "transformer", "checkpoint-200", 200)
    assert resolve_checkpoint(checkpoint) == resolve_checkpoint(checkpoint / "transformer") == expected
    with pytest.raises(FileNotFoundError, match="Transformer config"):
        resolve_checkpoint(checkpoint / "missing")


def test_prepare_only_does_not_load_model_weights(train_data):
    args = arguments(train_data, "--prepare-only", "--sample-indices", "4", "0", "--language", "all")
    with patch("src.editmgt.init_edit_mgt", side_effect=AssertionError("Must not load weights")):
        main(args)
    manifest = json.loads((args.output_dir / "samples.json").read_text())
    assert [row["train_index"] for row in manifest] == [4, 0]
    assert not list(args.output_dir.glob("step_*"))


@pytest.mark.parametrize("preserve_rgb", [False, True])
def test_main_loads_requested_checkpoint_and_generates_train_reports_without_gt_input(train_data, preserve_rgb):
    class Pipeline:
        _execution_device = torch.device("cpu")

        def __init__(self):
            self.transformer = torch.nn.Linear(1, 1)
            self.transformer.config = SimpleNamespace(sparse_target=True)
            self.vqvae = self.text_encoder = self.text_encoder_t5 = None
            self.scheduler = SimpleNamespace(config=SimpleNamespace(masking_schedule="cosine"), inference_shift=1.0)
            self.calls = []

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            self.calls.append(kwargs)
            self.last_generation_mask = torch.ones(1, 2, 2, dtype=torch.bool)
            return SimpleNamespace(images=[kwargs["reference_image"][0].copy()])

    args = arguments(train_data, "--sample-indices", "5", "2", "--seed", "7", "--bf16",
                     *(["--preserve-unmasked-pixels"] if preserve_rgb else []))
    pipe = Pipeline()
    with patch("src.editmgt.init_edit_mgt", return_value=pipe) as initialize:
        main(args)
    assert initialize.call_args.kwargs["transformer_path"] == str(args.checkpoint / "transformer")
    assert initialize.call_args.kwargs["enable_bf16"]
    assert not pipe.transformer.training and not pipe.transformer.weight.requires_grad
    assert [call["generator"].initial_seed() for call in pipe.calls] == [112, 109]
    for call in pipe.calls:
        assert "target" not in call and "output" not in call
        assert call["sparse_target"] and call["edit_only_diffusion"]
        assert call["sparse_target_halo"] == 0
        assert call["preserve_unmasked_pixels"] == preserve_rgb
        assert call["reference_image"][0].getpixel((0, 0))[0] != 200
    step_dir = args.output_dir / "step_000200"
    metadata = json.loads((step_dir / "metadata.json").read_text())
    assert metadata["split"] == "train" and metadata["checkpoint"] == str(args.checkpoint)
    assert metadata["preserve_unmasked_pixels"] == preserve_rgb
    assert metadata["output_compositing"] == ("original_pixel_mask" if preserve_rgb else "none")
    assert [row["train_index"] for row in metadata["cases"]] == [5, 2]
    page = (step_dir / "index.html").read_text()
    assert "Inter-Edit train ·" in page and "Inter-Edit test" not in page
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert "Inter-Edit train inference" in (args.output_dir / "index.html").read_text()
    for sample_id in (105, 102):
        for image in ("generated", "comparison", "generation_mask", "generation_mask_overlay"):
            assert (step_dir / f"sample_{sample_id:05d}" / f"{image}.png").is_file()
    assert (step_dir / "contact_sheet.png").is_file()


def test_train_entry_runs_real_sparse_sampler_and_vq_decoder(train_data):
    from diffusers import VQModel

    from src.pipeline import Pipeline
    from src.scheduler import Scheduler
    from src.transformer import Transformer2DModel

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        transformer = Transformer2DModel(
            num_layers=1, num_single_layers=1, attention_head_dim=12, num_attention_heads=2,
            joint_attention_dim=24, pooled_projection_dim=8, axes_dims_rope=(4, 4, 4),
            vocab_size=32, codebook_size=31, downsample=True, upsample=True, sparse_target=True,
        )
        vq = VQModel(block_out_channels=(16,), latent_channels=4, vq_embed_dim=4,
                     num_vq_embeddings=31, norm_num_groups=4, lookup_from_codebook=True)
        pipe = Pipeline(vq, None, None, transformer, Scheduler(mask_token_id=31), None, None)

        class EmbeddedPipeline:
            def __getattr__(self, name):
                return getattr(pipe, name)

            def __call__(self, **kwargs):
                kwargs.pop("prompt")
                return pipe(
                    prompt_embeds=torch.zeros(1, 8), encoder_hidden_states=torch.zeros(1, 3, 24),
                    negative_prompt_embeds=torch.zeros(1, 8), negative_encoder_hidden_states=torch.zeros(1, 3, 24),
                    **kwargs,
                )

        args = arguments(train_data, "--sample-indices", "4", "--num-inference-steps", "2", "--device", "cpu")
        with patch("src.editmgt.init_edit_mgt", return_value=EmbeddedPipeline()):
            main(args)
        step_dir = args.output_dir / "step_000200"
        with Image.open(step_dir / "sample_00104/generated.png") as generated:
            assert generated.mode == "RGB" and generated.size == (32, 32)
        assert not pipe.last_edit_region_info["preserve_unmasked_pixels"]
        assert pipe.last_edit_region_info["output_compositing"] == "none"
        assert (step_dir / "sample_00104/generation_mask.png").is_file()
        assert json.loads((step_dir / "metadata.json").read_text())["split"] == "train"
    finally:
        torch.set_num_threads(previous_threads)
