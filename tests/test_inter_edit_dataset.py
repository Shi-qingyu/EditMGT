import gzip
import io
import json
import tarfile
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset_utils import process_edit_sample
from src.inter_edit_dataset import InterEditDataset, InterEditReader, TarAssetReader


def png(color, size=(8, 8), mask=False):
    image = Image.new("L" if mask else "RGB", size, 0 if mask else color)
    if mask:
        image.paste(255, (2, 2, 4, 4))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def write_tar(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in members.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))


@pytest.fixture
def inter_edit(tmp_path):
    root = tmp_path / "Inter-Edit"
    train = root / "train"
    (train / "metadata").mkdir(parents=True)
    (train / "manifest.json").write_text(json.dumps({
        "num_samples": 8, "samples_per_shard": 2, "num_sample_shards": 4,
    }))
    for shard in range(4):
        rows, sources, assets = [], {}, {}
        for sample_id in range(shard * 2, shard * 2 + 2):
            row = {
                "sample_id": sample_id, "source_id": sample_id,
                "instruction": f"添加物体 {sample_id}", "edit_type": "Add", "better_data": False,
                "source_archive": f"source_shards/source-{shard:05d}.tar",
                "source_file": f"sources/source_{sample_id:07d}.png",
                "asset_archive": f"asset_shards/asset-{shard:05d}.tar",
                "target_file": f"targets/target_{sample_id:07d}.png",
                "mask_file": f"masks/mask_{sample_id:07d}.png",
            }
            rows.append(row)
            sources[row["source_file"]] = png((10 + sample_id, 20, 30))
            assets[row["target_file"]] = png((50 + sample_id, 60, 70))
            assets[row["mask_file"]] = png(None, mask=True)
        write_tar(train / rows[0]["source_archive"], sources)
        write_tar(train / rows[0]["asset_archive"], assets)
        with gzip.open(train / "metadata" / f"train-{shard:05d}-of-00004.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    test = root / "test"
    test.mkdir()
    test_row = {
        "index": 0, "source_path": "source_images/00000_source.png",
        "gt_path": "target_images/00000_gt.png", "mask_path": "masks/00000_mask.png",
        "instruction": "Add an object.", "edit_type": "Add", "language": "English",
    }
    (test / "metadata.json").write_text(json.dumps([test_row]))
    write_tar(test / "source_images.tar", {test_row["source_path"]: png((11, 22, 33))})
    write_tar(test / "target_images.tar", {test_row["gt_path"]: png((44, 55, 66))})
    write_tar(test / "masks.tar", {test_row["mask_path"]: png(None, mask=True)})
    return root, tmp_path / "cache"


def test_official_train_and_test_splits_and_cross_shard_sample_range(inter_edit):
    root, cache = inter_edit
    reader = InterEditReader(root, "train", cache)
    rows = list(reader.iter_samples(start=3, count=3))
    assert [row[0] for row in rows] == [3, 4, 5]
    assert [row[3]["sample_id"] for row in rows] == [3, 4, 5]
    assert rows[0][3]["input"].getpixel((0, 0)) == (13, 20, 30)
    assert rows[0][3]["output"].getpixel((0, 0)) == (53, 60, 70)
    assert rows[0][3]["edit_mask"].getpixel((2, 2)) == 255
    test_reader = InterEditReader(root / "test", "test", cache)
    test_sample = next(test_reader.iter_samples(count=1))[3]
    assert test_sample["sample_id"] == 0
    assert test_sample["input"].getpixel((0, 0)) == (11, 22, 33)
    assert reader.num_samples == 8 and test_reader.num_samples == 1
    with pytest.raises(ValueError):
        list(reader.iter_samples(start=7, count=2))


def test_tar_index_is_reused_after_reader_closes_and_archive_handles_are_bounded(inter_edit):
    root, cache = inter_edit
    with TarAssetReader(root / "train", cache, max_open_archives=1) as reader:
        reader.image("source_shards/source-00000.tar", "sources/source_0000000.png")
        first_handle = next(iter(reader._archives.values()))[0]
        reader.image("asset_shards/asset-00000.tar", "targets/target_0000000.png")
        assert first_handle.closed and len(reader._archives) == 1
    with patch("src.inter_edit_dataset.tarfile.open", side_effect=AssertionError("Index should be cached")):
        with TarAssetReader(root / "train", cache) as reader:
            assert reader.image("source_shards/source-00000.tar", "sources/source_0000000.png").size == (8, 8)
    assert not list((root / "train").glob("sources/*"))


def test_rank_and_worker_sharding_cover_all_samples_without_duplicates(inter_edit):
    root, cache = inter_edit
    all_ids = []
    for rank in range(2):
        dataset = InterEditDataset(root, rank=rank, world_size=2, cache_dir=cache,
                                  shuffle=False, repeat=False, size=8)
        assert len(dataset) == 4
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        rank_ids = [int(sample_id) for batch in loader for sample_id in batch["sample_id"]]
        assert len(rank_ids) == len(set(rank_ids)) == 4
        all_ids.extend(rank_ids)
    assert sorted(all_ids) == list(range(8))


def test_dataset_exposes_training_fields_and_keeps_source_target_mask_aligned(inter_edit):
    root, cache = inter_edit

    class Tokenizer:
        def __call__(self, instruction, **kwargs):
            class Tokens:
                input_ids = torch.ones(1, 77, dtype=torch.long)
            return Tokens()

    dataset = InterEditDataset(root, tokenizer=Tokenizer(), cache_dir=cache, shuffle=False, repeat=False, size=8)
    sample = next(iter(dataset))
    assert sample["image"].shape == sample["reference_image"].shape == (3, 8, 8)
    assert sample["edit_mask"].shape == (1, 8, 8)
    assert sample["edit_mask"].sum().item() == 4
    assert sample["micro_conds"].tolist() == [8, 8, 0, 0, 6]
    assert sample["prompt_input_ids"].shape == (77,)
    assert sample["instruction"] == "添加物体 0"
    torch.testing.assert_close(sample["image"][0], torch.full((8, 8), 50 / 255))
    torch.testing.assert_close(sample["reference_image"][0], torch.full((8, 8), 10 / 255))


def test_inter_edit_crop_keeps_localized_edit_when_random_crop_misses():
    source = Image.new("RGB", (32, 8), (20, 30, 40))
    mask = Image.new("L", (32, 8), 0)
    mask.paste(255, (28, 3, 30, 5))
    with patch("src.dataset_utils.transforms.RandomCrop.get_params", return_value=(0, 0, 8, 8)):
        sample = process_edit_sample(source, source, mask, 8, ensure_edit_in_crop=True)
    assert sample["edit_mask"].any()
    assert torch.equal(sample["image"], sample["reference_image"])
    assert sample["micro_conds"][3] >= 21


def test_analysis_sample_adapter_uses_official_test_split_without_parquet_manifest(inter_edit):
    from scripts.test_zero_shot import decode_image, iter_test_samples
    root, _ = inter_edit
    with patch("src.inter_edit_dataset.DEFAULT_INTER_EDIT_CACHE", root.parent / "script_cache"):
        samples = list(iter_test_samples(root, None, 0, 1))
    assert samples[0][1] == "metadata.json"
    assert samples[0][3]["instruction"] == "Add an object."
    assert decode_image(samples[0][3]["input"]).getpixel((0, 0)) == (11, 22, 33)


def test_tar_index_supports_pax_long_names_and_payload_offsets(tmp_path):
    long_name = "sources/" + "long_name_" * 20 + "图像.png"
    write_tar(tmp_path / "archive.tar", {long_name: png((1, 2, 3)), "masks/next.png": png(None, mask=True)})
    with TarAssetReader(tmp_path, tmp_path / "cache") as reader:
        assert reader.image("archive.tar", long_name).getpixel((0, 0)) == (1, 2, 3)
        assert reader.image("archive.tar", "masks/next.png").getpixel((2, 2)) == 255


def write_test_translations(root, directory):
    """Create the translation script's actual sidecar format for a small release."""
    from src.inter_edit_translations import HAN
    reader = InterEditReader(root)
    (directory / "shards").mkdir(parents=True)
    count, inputs = 0, []
    for path in reader.metadata_files:
        stat = path.stat()
        inputs.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        records = []
        for index, row in enumerate(reader.read_metadata(path)):
            if HAN.search(row["instruction"]):
                records.append({"metadata_file": path.name, "row_index": index, "sample_id": row["sample_id"],
                                "instruction_zh": row["instruction"], "instruction_en": f"Add object {row['sample_id']}."})
        count += len(records)
        (directory / "shards" / path.name.removesuffix(".gz")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({"source_train_dir": str(reader.root.resolve()),
                                                         "inputs": inputs, "limit": None, "model": "Qwen/Qwen3.5-9B"}))
    (directory / "summary.json").write_text(json.dumps({"selected": count, "translated": count, "pending": 0, "complete": True}))
    return directory


def test_translations_enter_english_training_before_filtering_and_tokenization(inter_edit, tmp_path):
    root, cache = inter_edit
    original = InterEditReader(root, cache_dir=cache)
    path = original.metadata_files[0]
    assert original.count_metadata_languages(path).get("en", 0) == 0
    directory = write_test_translations(root, tmp_path / "translated")
    # English instructions may still contain Chinese text that must be rendered.
    shard = directory / "shards" / path.name.removesuffix(".gz")
    records = [json.loads(line) for line in shard.read_text().splitlines()]
    records[0]["instruction_en"] = 'Add the text "你好" to the sign.'
    shard.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    tokenized = []

    class Tokenizer:
        def __call__(self, instruction, **kwargs):
            tokenized.append(instruction)
            class Tokens:
                input_ids = torch.ones(1, 77, dtype=torch.long)
            return Tokens()

    dataset = InterEditDataset(root, cache_dir=cache, translations_dir=directory, language="en",
                              tokenizer=Tokenizer(), shuffle=False, repeat=False, size=8)
    assert len(dataset) == 8
    assert dataset.reader.count_metadata_languages(path)["en"] == 2
    samples = list(dataset)
    assert len(samples) == 8 and tokenized == [row["instruction"] for row in samples]
    assert samples[0]["instruction"] == 'Add the text "你好" to the sign.'
    assert samples[0]["edit_mask"].sum().item() == 4
    assert original.read_metadata(path)[0]["instruction"] == "添加物体 0"
    # Official test prompts remain untouched.
    assert next(InterEditReader(root, "test").iter_samples(count=1))[3]["instruction"] == "Add an object."


@pytest.mark.parametrize("failure", ["missing", "wrong_id", "wrong_original", "duplicate"])
def test_translation_mismatches_fail_instead_of_silently_dropping_rows(inter_edit, tmp_path, failure):
    root, _ = inter_edit
    directory = write_test_translations(root, tmp_path / "translated")
    reader = InterEditReader(root, translations_dir=directory)
    path = reader.metadata_files[0]
    sidecar = directory / "shards" / path.name.removesuffix(".gz")
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    if failure == "missing":
        records.pop()
    elif failure == "wrong_id":
        records[0]["sample_id"] = 999
    elif failure == "wrong_original":
        records[0]["instruction_zh"] = "wrong source"
    else:
        records.append(records[0])
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="translation"):
        reader.read_metadata(path)


@pytest.mark.parametrize("failure", ["limited", "incomplete", "changed_source", "test_split"])
def test_translation_overlay_rejects_incompatible_releases(inter_edit, tmp_path, failure):
    root, _ = inter_edit
    directory = write_test_translations(root, tmp_path / "translated")
    split = "train"
    if failure == "limited":
        config = json.loads((directory / "run_config.json").read_text())
        config["limit"] = 8
        (directory / "run_config.json").write_text(json.dumps(config))
    elif failure == "incomplete":
        (directory / "summary.json").write_text(json.dumps({"complete": False, "pending": 1}))
    elif failure == "changed_source":
        metadata = next((root / "train/metadata").glob("*.gz"))
        with metadata.open("ab") as handle:
            handle.write(b"changed")
    else:
        split = "test"
    with pytest.raises(ValueError):
        InterEditReader(root, split=split, translations_dir=directory)


def write_official_test_translations(root, directory):
    from scripts import translate_inter_edit as translation

    source_root, paths = translation.discover_inputs(root, "test")
    directory.mkdir(parents=True)
    args = translation.parse_args(["--split", "test", "--output-dir", str(directory)])
    config = translation.run_config(args, source_root, paths)
    records = [{**row, "instruction_en": f"Add translated object {row['sample_id']}."}
               for row in translation.selected_rows(paths[0], split="test")]
    (directory / "run_config.json").write_text(json.dumps(config))
    (directory / "translations.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    (directory / "summary.json").write_text(json.dumps({"complete": True, "pending": 0,
                                                        "selected": len(records), "translated": len(records)}))
    return directory


@pytest.mark.parametrize("metadata_format", ["json", "jsonl"])
def test_official_test_translation_keeps_indices_assets_and_english_literal_text(inter_edit, tmp_path, metadata_format):
    root, cache = inter_edit
    metadata = root / "test/metadata.json"
    english = {**json.loads(metadata.read_text())[0], "instruction": 'Write "你好" on the sign.'}
    chinese = {**english, "index": 7, "language": "中文", "instruction": "在牌子上添加一个红色物体"}
    rows = [english, chinese]
    if metadata_format == "json":
        metadata.write_text(json.dumps(rows, ensure_ascii=False))
    else:
        metadata.unlink()
        metadata = root / "test/metadata.jsonl"
        metadata.write_text("\n\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    original_bytes = metadata.read_bytes()
    directory = write_official_test_translations(root, tmp_path / "translated-test")
    reader = InterEditReader(root, "test", cache, translations_dir=directory)
    assert reader.test_rows[0] == english
    assert reader.test_rows[1]["instruction"] == "Add translated object 7."
    assert reader.test_rows[1]["instruction_original"] == chinese["instruction"]
    assert reader.test_rows[1]["language_original"] == "中文"
    dataset = InterEditDataset(root, split="test", cache_dir=cache, translations_dir=directory,
                              language="en", size=8, shuffle=False, repeat=False)
    assert len(dataset) == 2
    samples = list(dataset)
    assert [sample["sample_id"] for sample in samples] == [0, 7]
    assert all(sample["edit_mask"].sum() == 4 for sample in samples)
    assert metadata.read_bytes() == original_bytes
    assert InterEditReader(root, "test").test_rows == rows

    from types import SimpleNamespace
    from eval.benchmark_inter_edit import select_cases

    selection, info = select_cases(SimpleNamespace(data_dir=root, tar_cache_dir=cache, translations_dir=directory,
                                                  language="en", sample_indices=None, limit=None))
    assert [case["test_index"] for case in selection] == [0, 7]
    assert selection[1]["instruction_original"] == chinese["instruction"]
    assert info["eligible_samples"] == 2 and info["instruction_variant"] == "translated_english"


@pytest.mark.parametrize("failure", ["missing", "wrong_id", "wrong_original"])
def test_official_test_translation_rejects_incomplete_or_mismatched_rows(inter_edit, tmp_path, failure):
    root, cache = inter_edit
    metadata = root / "test/metadata.json"
    rows = json.loads(metadata.read_text())
    rows[0].update(instruction="添加一个物体", language="中文")
    metadata.write_text(json.dumps(rows, ensure_ascii=False))
    directory = write_official_test_translations(root, tmp_path / "translated-test")
    path = directory / "translations.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "missing":
        records.clear()
    elif failure == "wrong_id":
        records[0]["sample_id"] = 999
    else:
        records[0]["instruction_zh"] = "不同的原文"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="translation"):
        InterEditReader(root, "test", cache, translations_dir=directory)
