"""Real image/parquet batches across translation overlays and three dataset streams."""
from collections import Counter
import argparse
import ast
import io
import itertools
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from src.maskedit_dataset import MASKEDIT_CONFIGS, MaskEditDataset
from src.training_data import build_inter_edit_training_data, edit_data_probabilities, write_data_mixture_info
from test_inter_edit_dataset import inter_edit, write_test_translations


def encoded(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return {"bytes": buffer.getvalue(), "path": "embedded.png"}


@pytest.fixture
def maskedit_data(tmp_path):
    root = tmp_path / "MaskEdit-10K"
    for config in MASKEDIT_CONFIGS:
        directory = root / "data" / config
        directory.mkdir(parents=True)
        for shard in range(8):
            mask = Image.new("L", (8, 8), 0)
            mask.paste(1 if shard % 2 else 255, (2, 2, 6, 6))
            if config != "scene":
                mask = mask.convert("RGBA")  # Published infographic masks have opaque alpha.
            rows = []
            for index in range(2):
                rows.append({"id": f"{config}_{shard}_{index}", "source": encoded(Image.new("RGB", (8, 8), (10, 20, 30))),
                             "target": encoded(Image.new("RGB", (8, 8), (40, 50, 60))), "mask": encoded(mask),
                             "edit_instruction": "Add a red ball on the left.",
                             "prompt": "Make [MASK_AREA] red.", "content_language": "none" if config == "scene" else "en"})
            pq.write_table(pa.Table.from_pylist(rows), directory / f"train-{shard:05d}.parquet")
        # If test shards are accidentally included, schema validation must fail.
        pq.write_table(pa.Table.from_pylist([{"id": "DO_NOT_READ_TEST"}]), directory / "test-00000.parquet")
    # This entire Chinese configuration must be ignored, even before opening it.
    chinese = root / "data/infographics_cn"
    chinese.mkdir()
    pq.write_table(pa.Table.from_pylist([{"id": "DO_NOT_READ_CHINESE"}]), chinese / "train-00000.parquet")
    return root


def test_maskedit_preserves_white_mask_alignment_and_excludes_chinese_config(maskedit_data):
    samples = list(MaskEditDataset(maskedit_data, size=8, shuffle=False, repeat=False))
    assert len(samples) == 32
    assert {sample["sample_id"] for sample in samples} == set(range(32))
    expected_mask = torch.zeros(1, 8, 8)
    expected_mask[:, 2:6, 2:6] = 1
    for sample in samples:
        assert torch.equal(sample["edit_mask"], expected_mask)
        torch.testing.assert_close(sample["image"][0], torch.full((8, 8), 40 / 255))
        torch.testing.assert_close(sample["reference_image"][0], torch.full((8, 8), 10 / 255))
        assert "[MASK_AREA]" not in sample["instruction"]
    assert all(sample["instruction"] == "Add a red ball on the left." for sample in samples)


@pytest.mark.parametrize("prompt_field", ["edit_instruction", "prompt"])
def test_maskedit_filters_chinese_and_mixed_text_before_decoding_images(tmp_path, prompt_field):
    directory = tmp_path / "data/scene"
    directory.mkdir(parents=True)
    image = encoded(Image.new("RGB", (8, 8), (10, 20, 30)))
    mask = encoded(Image.new("L", (8, 8), 255))
    rows = []
    for index, (instruction, language) in enumerate([
        ("Add an object.", "none"), ("Make it blue.", "en"),
        ("添加一个物体。", "en"), ("Replace the 木勺 with a mug.", "en"),
        ('Add the text "你好".', "en"), ("Keep this English instruction out.", "zh"),
        ("Keep this English instruction out too.", "zh-CN"), ("123", "none"),
    ]):
        # Invalid image payloads ensure rejected language rows are never decoded.
        rows.append({"id": str(index), prompt_field: instruction, "content_language": language,
                     "source": image if index < 2 else None, "target": image if index < 2 else None,
                     "mask": mask if index < 2 else None})
    pq.write_table(pa.Table.from_pylist(rows), directory / "train-00000.parquet")
    dataset = MaskEditDataset(tmp_path, configs=("scene",), prompt_field=prompt_field,
                              size=8, shuffle=False, repeat=False, cache_dir=tmp_path / "cache")
    samples = list(dataset)
    assert len(dataset) == len(samples) == 2
    assert [sample["sample_id"] for sample in samples] == [0, 1]
    assert not (tmp_path / "cache/empty_masks").exists()


def test_maskedit_eight_ranks_two_workers_have_disjoint_complete_coverage(maskedit_data):
    all_ids = []
    for rank in range(8):
        dataset = MaskEditDataset(maskedit_data, size=8, rank=rank, world_size=8, shuffle=False, repeat=False)
        ids = [int(value) for batch in DataLoader(dataset, batch_size=2, num_workers=2) for value in batch["sample_id"]]
        assert len(ids) == len(set(ids)) == len(dataset) == 4
        all_ids.extend(ids)
    assert sorted(all_ids) == list(range(32))


def test_maskedit_mask_placeholder_is_resolved_when_prompt_variant_is_selected(maskedit_data):
    sample = next(iter(MaskEditDataset(maskedit_data, size=8, prompt_field="prompt", shuffle=False)))
    assert sample["instruction"] == "Make the masked area red."


def test_maskedit_empty_or_missing_masks_are_excluded_and_logged(tmp_path):
    directory = tmp_path / "data/scene"
    directory.mkdir(parents=True)
    source = encoded(Image.new("RGB", (8, 8), (10, 20, 30)))
    rows = [{"id": str(index), "source": source, "target": source, "edit_instruction": "Add an object.",
             "mask": value} for index, value in enumerate((None, encoded(Image.new("L", (8, 8), 0)),
                                                           encoded(Image.new("L", (8, 8), 255))))]
    pq.write_table(pa.Table.from_pylist(rows), directory / "train-00000.parquet")
    cache = tmp_path / "cache"
    samples = list(MaskEditDataset(tmp_path, configs=("scene",), cache_dir=cache, size=8, shuffle=False, repeat=False))
    assert [sample["sample_id"] for sample in samples] == [2]
    logs = [json.loads(line) for path in (cache / "empty_masks/train").glob("*.jsonl") for line in path.read_text().splitlines()]
    assert {item["reason"] for item in logs} == {"missing_edit_mask", "empty_edit_mask"}


@pytest.mark.parametrize("magic,mask,expected", [(True, True, [0.7, 0.2, 0.1]),
                                                (True, False, [0.8, 0.2]), (False, True, [0.9, 0.1]),
                                                (False, False, [1.0])])
def test_probabilities_reserve_weights_only_for_enabled_sources(magic, mask, expected):
    assert list(edit_data_probabilities(magic, 0.2, mask, 0.1).values()) == pytest.approx(expected)


@pytest.mark.parametrize("magic,mask", [(0.9, 0.1), (0.9, 0.2), (float("nan"), 0.1), (0.2, 0), (0.2, -1)])
def test_invalid_mixture_probabilities_are_rejected(magic, mask):
    with pytest.raises(ValueError):
        edit_data_probabilities(True, magic, True, mask)


def test_training_cli_accepts_resume_data_options_and_rejects_invalid_total(monkeypatch):
    # Exercise the real argument parser without importing CUDA/model dependencies.
    from src.mask_schedule import validate_inference_shift
    from src.timestep_sampling import MaskRatioSampler
    path = Path(__file__).resolve().parents[1] / "train/train.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("parse_args", "make_mask_ratio_sampler")]
    namespace = dict(argparse=argparse, os=os, MASKEDIT_CONFIGS=MASKEDIT_CONFIGS,
                     edit_data_probabilities=edit_data_probabilities, MaskRatioSampler=MaskRatioSampler,
                     validate_inference_shift=validate_inference_shift, is_wandb_available=lambda: False)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    command = ["train.py", "--pretrained_model_name_or_path", "checkpoint/EditMGT", "--report_to", "none",
               "--instance_dataset", "InterEditDataset", "--train_edit_model", "True", "--dataset_language", "en",
               "--inter_edit_translations_dir", "/translations", "--magicbrush_data_dir", "/magicbrush",
               "--maskedit_data_dir", "/maskedit", "--require_resume", "--resume_from_checkpoint", "latest"]
    monkeypatch.setattr(sys, "argv", command)
    args = namespace["parse_args"]()
    assert args.inter_edit_translations_dir == "/translations"
    assert args.maskedit_configs == list(MASKEDIT_CONFIGS)
    assert args.maskedit_prompt_field == "edit_instruction"
    assert args.require_resume and args.resume_from_checkpoint == "latest"
    assert args.allow_condition_cache_threshold_change_on_resume
    assert args.condition_cache_threshold == 0.8
    assert args.validation_reference_kv_cache_threshold == 0.8
    assert list(edit_data_probabilities(True, args.magicbrush_probability, True, args.maskedit_probability).values()) == [0.7, 0.2, 0.1]
    monkeypatch.setattr(sys, "argv", command + ["--allow_condition_cache_threshold_change_on_resume",
                                             "--condition_cache_threshold", "0.8"])
    migrated = namespace["parse_args"]()
    assert migrated.allow_condition_cache_threshold_change_on_resume and migrated.condition_cache_threshold == 0.8
    monkeypatch.setattr(sys, "argv", command + ["--maskedit_probability", "0.9"])
    with pytest.raises(SystemExit) as exc:
        namespace["parse_args"]()
    assert exc.value.code == 2


def test_three_source_factory_tokenizes_translations_and_collates_real_mixed_batches(inter_edit, maskedit_data, tmp_path):
    root, cache = inter_edit
    translations = write_test_translations(root, tmp_path / "translations")
    magic_root = tmp_path / "magicbrush/data"
    magic_root.mkdir(parents=True)
    mask = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
    mask.paste((0, 0, 0, 0), (2, 2, 6, 6))
    for shard in range(2):
        pq.write_table(pa.Table.from_pylist([{"img_id": str(shard), "turn_index": 1, "instruction": "Add a blue star.",
                                            "source_img": encoded(Image.new("RGB", (8, 8), (10, 20, 30))),
                                            "target_img": encoded(Image.new("RGB", (8, 8), (40, 50, 60))),
                                            "mask_img": encoded(mask)}]), magic_root / f"train-{shard:05d}.parquet")
    args = SimpleNamespace(instance_data_dir=root, dataset_split="train", dataset_language="en", dataset_cache_dir=cache,
                           inter_edit_translations_dir=translations, magicbrush_data_dir=magic_root.parent,
                           magicbrush_probability=0.2, magicbrush_read_batch_size=2, maskedit_data_dir=maskedit_data,
                           maskedit_probability=0.1, maskedit_configs=MASKEDIT_CONFIGS, maskedit_prompt_field="edit_instruction",
                           maskedit_read_batch_size=2, resolution=8, text_encoder_architecture="CLIP", seed=42,
                           resume_from_checkpoint="latest")

    class Tokenizer:
        def __call__(self, instruction, **kwargs):
            return SimpleNamespace(input_ids=torch.full((1, 77), len(instruction), dtype=torch.long))

    dataset, info = build_inter_edit_training_data(args, Tokenizer(), 0, 1)
    assert [item["sample_count"] for item in info["sources"]] == [8, 2, 32]
    assert [item["probability"] for item in info["sources"]] == pytest.approx([0.7, 0.2, 0.1])
    assert info["sources"][0]["translated_prompt_count_global"] == 8
    assert info["sources"][2]["configs"] == ["scene", "infographics_en"]
    assert info["sources"][2]["prompt_language_filter"] == "en_without_han"
    assert info["sources"][2]["content_language_filter"] == "exclude_chinese"
    observed = Counter()
    for batch in itertools.islice(DataLoader(dataset, batch_size=8, num_workers=2), 20):
        observed.update(batch["dataset_name"])
        assert batch["sample_id"].dtype == torch.int64
        assert batch["edit_mask"].shape == (8, 1, 8, 8)
        assert batch["prompt_input_ids"].shape == (8, 77)
        assert all(name + ":" in key for name, key in zip(batch["dataset_name"], batch["sample_key"]))
        for name, instruction in zip(batch["dataset_name"], batch["instruction"]):
            if name == "Inter-Edit":
                assert instruction.startswith("Add object ")
    assert set(observed) == {"Inter-Edit", "MagicBrush", "MaskEdit-10K"}
    output = tmp_path / "run"
    write_data_mixture_info(output, {**info, "sources": [{"name": "old-data"}]})
    write_data_mixture_info(output, info)
    assert json.loads((output / "data_mixture/rank_0000.json").read_text()) == info
    assert len(list((output / "data_mixture/history").glob("*.json"))) == 1
