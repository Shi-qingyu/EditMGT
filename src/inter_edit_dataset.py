"""Read the local Inter-Edit releases without extracting their tar archives."""
import fcntl
import gzip
import hashlib
import io
import json
import logging
import os
import random
import tarfile
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from src.dataset_utils import process_edit_sample, tokenize_prompt


DEFAULT_INTER_EDIT_ROOT = "/mnt/geminihzceph/public/data/Inter-Edit"
DEFAULT_INTER_EDIT_CACHE = Path(__file__).resolve().parents[1] / "data" / "inter_edit_tar_index"
logger = logging.getLogger(__name__)


class TarAssetReader:
    """Cache tar member offsets on disk and a bounded number of open archives.

    Index construction visits headers and seeks over payloads. Subsequent reads
    seek straight to the requested image; no archive is unpacked or modified.
    File locks and atomic replacement allow DDP workers to share the index.
    """

    def __init__(self, root, cache_dir=None, max_open_archives=16):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_INTER_EDIT_CACHE
        if max_open_archives < 1:
            raise ValueError("max_open_archives must be positive")
        self.max_open_archives = max_open_archives
        self._archives = OrderedDict()

    def _load_index(self, archive):
        stat = archive.stat()
        identity = f"{archive.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.cache_dir / f"{archive.stem}-{digest}.json"
        with (index_path.with_suffix(".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if index_path.is_file():
                return json.loads(index_path.read_text())
            index = {}
            with tarfile.open(archive, "r:") as tar:
                member = tar.next()
                while member is not None:
                    if member.isfile():
                        if member.offset_data + member.size > stat.st_size:
                            raise OSError(f"Truncated Inter-Edit archive: {archive}")
                        index[member.name] = [member.offset_data, member.size]
                    # TarFile.next also reads the preceding payload's last byte.
                    # On shared storage that adds a remote read per image; seek
                    # straight to the next header and use tarfile's PAX/GNU parser.
                    tar.fileobj.seek(tar.offset)
                    try:
                        member = tar.tarinfo.fromtarfile(tar)
                    except tarfile.EOFHeaderError:
                        break
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile("w", dir=self.cache_dir, suffix=".tmp", delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    json.dump(index, temporary, separators=(",", ":"))
                os.replace(temporary_path, index_path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
            return index

    def image(self, archive_name, member_name):
        # Also accept an already extracted official test release.
        extracted = self.root / member_name
        if extracted.is_file():
            with Image.open(extracted) as image:
                return image.copy()
        archive = self.root / archive_name
        if archive not in self._archives:
            index = self._load_index(archive)
            self._archives[archive] = (archive.open("rb"), index)
            if len(self._archives) > self.max_open_archives:
                _, (old_handle, _) = self._archives.popitem(last=False)
                old_handle.close()
        self._archives.move_to_end(archive)
        handle, index = self._archives[archive]
        if member_name not in index:
            raise KeyError(f"Missing Inter-Edit asset {member_name!r} in {archive}")
        offset, size = index[member_name]
        handle.seek(offset)
        payload = handle.read(size)
        if len(payload) != size:
            raise OSError(f"Truncated Inter-Edit asset {member_name!r} in {archive}")
        with Image.open(io.BytesIO(payload)) as image:
            return image.copy()

    def close(self):
        for handle, _ in self._archives.values():
            handle.close()
        self._archives.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class InterEditReader:
    """Expose official metadata and source/target/mask PIL images for both splits."""

    def __init__(self, root_dir=DEFAULT_INTER_EDIT_ROOT, split="train", cache_dir=None):
        if split not in ("train", "test"):
            raise ValueError("Inter-Edit split must be 'train' or 'test'")
        root = Path(root_dir)
        self.root = root / split if (root / split).is_dir() else root
        self.split = split
        self.cache_dir = cache_dir
        self.test_rows = None
        if split == "train":
            with (self.root / "manifest.json").open() as handle:
                manifest = json.load(handle)
            self.num_samples = int(manifest["num_samples"])
            self.samples_per_shard = int(manifest["samples_per_shard"])
            self.metadata_files = sorted((self.root / "metadata").glob("train-*.jsonl.gz"))
            if len(self.metadata_files) != int(manifest["num_sample_shards"]):
                raise ValueError(f"Incomplete Inter-Edit training metadata under {self.root}")
        else:
            with (self.root / "metadata.json").open(encoding="utf-8") as handle:
                self.test_rows = json.load(handle)
            self.num_samples = len(self.test_rows)
            self.metadata_files = [self.root / "metadata.json"]

    def read_metadata(self, path):
        if self.split == "test":
            return self.test_rows
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def read_sample(self, row, assets):
        if self.split == "train":
            source = assets.image(row["source_archive"], row["source_file"])
            target = assets.image(row["asset_archive"], row["target_file"])
            mask = assets.image(row["asset_archive"], row["mask_file"])
            sample_id = row["sample_id"]
        else:
            source = assets.image("source_images.tar", row["source_path"])
            target = assets.image("target_images.tar", row["gt_path"])
            mask = assets.image("masks.tar", row["mask_path"])
            sample_id = row["index"]
        return {
            **row, "sample_id": sample_id,
            "input": source, "output": target, "edit_mask": mask,
        }

    def iter_samples(self, start=0, count=None):
        if start < 0 or start >= self.num_samples:
            raise IndexError(f"Inter-Edit sample index {start} is outside [0,{self.num_samples})")
        if count is not None and (count < 1 or start + count > self.num_samples):
            raise ValueError("Inter-Edit sample count must be positive and fit within the selected split")
        current, yielded = 0, 0
        with TarAssetReader(self.root, self.cache_dir) as assets:
            for path in self.metadata_files:
                if self.split == "train" and current + self.samples_per_shard <= start:
                    current += self.samples_per_shard
                    continue
                rows = self.read_metadata(path)
                for row_index, row in enumerate(rows):
                    sample_index = current + row_index
                    if sample_index < start:
                        continue
                    yield sample_index, path.name, row_index, self.read_sample(row, assets)
                    yielded += 1
                    if count is not None and yielded == count:
                        return
                current += len(rows)


class InterEditDataset(IterableDataset):
    """Stream train shards with disjoint DDP-rank and DataLoader-worker ownership.

    The training entry point leaves this loader out of Accelerator.prepare:
    this class already distributes data, and all tensors are moved explicitly.
    Empty-mask annotations are filtered during iteration. __len__ reports the
    nominal metadata count before filtering; repeated training stops by steps.
    """

    def __init__(
        self, root_dir=DEFAULT_INTER_EDIT_ROOT, split="train", tokenizer=None,
        size=1024, text_encoder_architecture="CLIP", norm=False,
        rank=0, world_size=1, shuffle=True, repeat=True, cache_dir=None, seed=42,
    ):
        super().__init__()
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Inter-Edit rank must be in [0,world_size)")
        self.reader = InterEditReader(root_dir, split, cache_dir)
        self.tokenizer, self.size = tokenizer, size
        self.text_encoder_architecture, self.norm = text_encoder_architecture, norm
        self.rank, self.world_size = rank, world_size
        self.shuffle, self.repeat, self.seed = shuffle, repeat, seed
        if split == "train":
            self.metadata_files = self.reader.metadata_files[rank::world_size]
            if not self.metadata_files:
                raise ValueError("Inter-Edit has fewer metadata shards than distributed ranks")
            shard_ids = range(rank, len(self.reader.metadata_files), world_size)
            self.num_samples = sum(min(self.reader.samples_per_shard,
                                       self.reader.num_samples - index * self.reader.samples_per_shard)
                                   for index in shard_ids)
        else:
            self.metadata_files = self.reader.metadata_files
            self.num_samples = len(self.reader.test_rows[rank::world_size])

    def __len__(self):
        return self.num_samples

    def _process_sample(self, raw):
        result = process_edit_sample(
            raw["output"], raw["input"], raw["edit_mask"], self.size, self.norm,
            ensure_edit_in_crop=True,
        )
        if not result["edit_mask"].any():
            return None
        result.update({"sample_id": raw["sample_id"], "instruction": raw["instruction"], "edit_type": raw["edit_type"]})
        if self.tokenizer is not None:
            tokens = tokenize_prompt(self.tokenizer, raw["instruction"], self.text_encoder_architecture)
            result["prompt_input_ids"] = [value[0] for value in tokens] if isinstance(self.tokenizer, list) else tokens[0]
        return result

    def _record_empty_mask(self, raw, worker_id):
        log_root = (Path(self.reader.cache_dir) if self.reader.cache_dir is not None
                    else DEFAULT_INTER_EDIT_CACHE) / "empty_masks" / self.reader.split
        log_root.mkdir(parents=True, exist_ok=True)
        # Workers use separate journals, so DDP does not need a shared-file lock.
        log_path = log_root / f"rank_{self.rank:04d}_worker_{worker_id:02d}_pid_{os.getpid()}.jsonl"
        record = {
            "sample_id": raw["sample_id"], "rank": self.rank, "worker_id": worker_id,
            "reason": "empty_edit_mask", "data_root": str(self.reader.root),
            "edit_type": raw.get("edit_type"),
            "mask_archive": raw.get("asset_archive", "masks.tar"),
            "mask_file": raw.get("mask_file", raw.get("mask_path")),
            "original_mask_mode": raw["edit_mask"].mode,
            "original_mask_extrema": raw["edit_mask"].getextrema(),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return log_path

    def __iter__(self):
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        files = self.metadata_files[worker_id::worker_count] if self.reader.split == "train" else self.metadata_files
        if not files:
            raise ValueError("Inter-Edit has fewer rank-local metadata shards than DataLoader workers")
        rng = random.Random(self.seed + self.rank * worker_count + worker_id)
        empty_sample_ids = set()
        with TarAssetReader(self.reader.root, self.reader.cache_dir) as assets:
            while True:
                if self.shuffle:
                    rng.shuffle(files)
                yielded = 0
                for path in files:
                    rows = self.reader.read_metadata(path)
                    if self.reader.split == "test":
                        rows = rows[self.rank::self.world_size][worker_id::worker_count]
                    else:
                        rows = list(rows)
                    if self.shuffle:
                        rng.shuffle(rows)
                    for row in rows:
                        sample_id = row["sample_id"] if self.reader.split == "train" else row["index"]
                        if sample_id in empty_sample_ids:
                            continue
                        raw = self.reader.read_sample(row, assets)
                        sample = self._process_sample(raw)
                        if sample is None:
                            empty_sample_ids.add(sample_id)
                            log_path = self._record_empty_mask(raw, worker_id)
                            # Every rejected sample is journaled; bound console
                            # warnings while keeping cumulative counts visible.
                            if len(empty_sample_ids) <= 5 or len(empty_sample_ids) % 1000 == 0:
                                logger.warning(
                                    "Skipping Inter-Edit sample %s with empty edit mask "
                                    "(rank=%s, worker=%s, skipped=%s); journal=%s",
                                    sample_id, self.rank, worker_id, len(empty_sample_ids), log_path,
                                )
                            continue
                        yielded += 1
                        yield sample
                if not self.repeat:
                    return
                if not yielded:
                    raise ValueError(
                        f"No valid Inter-Edit samples remain for rank={self.rank}, worker={worker_id}; "
                        f"{len(empty_sample_ids)} empty-mask samples were excluded. "
                        "Check the assigned annotation shards instead of repeating an empty stream."
                    )
