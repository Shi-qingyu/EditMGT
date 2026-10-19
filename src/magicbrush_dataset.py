"""Stream the local MagicBrush parquet release using its provided edit masks."""
import io
import json
import logging
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from src.dataset_utils import process_edit_sample, tokenize_prompt
from src.inter_edit_dataset import INTER_EDIT_LANGUAGES, matches_inter_edit_language
from src.mask_utils import decode_magicbrush_edit_mask


logger = logging.getLogger(__name__)
DEFAULT_MAGICBRUSH_ROOT = Path(__file__).resolve().parents[1] / "data/magicbrush"
DEFAULT_MAGICBRUSH_CACHE = Path(__file__).resolve().parents[1] / "data/magicbrush_cache"
MAGICBRUSH_COLUMNS = ("img_id", "turn_index", "instruction", "source_img", "mask_img", "target_img")


class MagicBrushDataset(IterableDataset):
    """Each row is one annotated turn; source_img already contains its history.

    Shards are owned by one rank/worker per pass. Bounded Arrow batches avoid
    converting a whole image-containing parquet shard into Python objects.
    Only the requested split is opened; training never consumes dev/test.
    """

    def __init__(
        self, root_dir=DEFAULT_MAGICBRUSH_ROOT, split="train", tokenizer=None,
        size=1024, text_encoder_architecture="CLIP", norm=False,
        rank=0, world_size=1, shuffle=True, repeat=True, seed=42,
        language="en", cache_dir=None, read_batch_size=4,
    ):
        super().__init__()
        if split not in ("train", "dev", "test"):
            raise ValueError("MagicBrush split must be train, dev or test")
        if world_size < 1 or not 0 <= rank < world_size or read_batch_size < 1:
            raise ValueError("Invalid MagicBrush rank/world size/read batch size")
        if language not in INTER_EDIT_LANGUAGES:
            raise ValueError(f"Unsupported MagicBrush language filter: {language!r}")
        self.root = Path(root_dir)
        self.data_dir = self.root / "data" if (self.root / "data").is_dir() else self.root
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_MAGICBRUSH_CACHE
        self.split, self.tokenizer, self.size = split, tokenizer, size
        self.text_encoder_architecture, self.norm = text_encoder_architecture, norm
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.shuffle, self.repeat, self.language = shuffle, repeat, language
        self.read_batch_size = read_batch_size
        all_files = sorted(self.data_dir.glob(f"{split}-*.parquet"))
        if not all_files:
            raise FileNotFoundError(f"No MagicBrush {split}-*.parquet under {self.data_dir}")
        self.files, self.file_offsets, self.num_samples = [], {}, 0
        offset = 0
        for file_index, path in enumerate(all_files):
            parquet = pq.ParquetFile(path)
            try:
                self.file_offsets[path] = offset
                offset += parquet.metadata.num_rows
                if file_index % world_size != rank:
                    continue
                missing = set(MAGICBRUSH_COLUMNS) - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f"MagicBrush shard {path} is missing fields: {sorted(missing)}")
                if language == "all":
                    eligible = parquet.metadata.num_rows
                else:
                    # Read only the small instruction column for filtered lengths.
                    # Images are neither read nor decoded during this count pass.
                    eligible = sum(
                        matches_inter_edit_language(row, language)
                        for batch in parquet.iter_batches(batch_size=1024, columns=["instruction"], use_threads=False)
                        for row in batch.to_pylist()
                    )
                if eligible:
                    self.files.append(path)
                    self.num_samples += eligible
            finally:
                parquet.close()
        if not self.files:
            raise ValueError(f"No MagicBrush {split} samples match language={language!r} on rank={rank}")

    def __len__(self):
        # As with Inter-Edit, this is the count before empty-mask exclusions.
        return self.num_samples

    def _decode_image(self, value, shard):
        if isinstance(value, Image.Image):
            return value.copy()
        if isinstance(value, dict):
            value = value.get("bytes") if value.get("bytes") is not None else value.get("path")
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(value)) as image:
                return image.copy()
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                candidates = (self.root / path, shard.parent / path)
                path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            with Image.open(path) as image:
                return image.copy()
        raise TypeError(f"Unsupported MagicBrush image value in {shard}: {type(value)!r}")

    def _record_exclusion(self, row, sample_id, path, row_index, worker_id, reason):
        log_root = self.cache_dir / "empty_masks" / self.split
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"rank_{self.rank:04d}_worker_{worker_id:02d}_pid_{os.getpid()}.jsonl"
        record = {
            "dataset": "MagicBrush", "split": self.split, "sample_id": sample_id,
            "img_id": row["img_id"], "turn_index": row["turn_index"],
            "shard": str(path), "row_index": row_index,
            "rank": self.rank, "worker_id": worker_id, "reason": reason,
            "language_filter": self.language,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _process_sample(self, row, sample_id, path):
        result = process_edit_sample(
            self._decode_image(row["target_img"], path),
            self._decode_image(row["source_img"], path),
            decode_magicbrush_edit_mask(self._decode_image(row["mask_img"], path)),
            self.size, self.norm, ensure_edit_in_crop=True,
        )
        if not result["edit_mask"].any():
            return None
        result.update({"sample_id": sample_id, "instruction": row["instruction"], "edit_type": "MagicBrush"})
        if self.tokenizer is not None:
            tokens = tokenize_prompt(self.tokenizer, row["instruction"], self.text_encoder_architecture)
            result["prompt_input_ids"] = [value[0] for value in tokens] if isinstance(self.tokenizer, list) else tokens[0]
        return result

    def __iter__(self):
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        files = list(self.files[worker_id::worker_count])
        if not files:
            raise ValueError("MagicBrush has fewer rank-local train shards than DataLoader workers")
        rng = random.Random(self.seed + 65537 + 1000003 * self.rank + worker_id)
        excluded_ids = set()
        while True:
            if self.shuffle:
                rng.shuffle(files)
            yielded = 0
            for path in files:
                parquet = pq.ParquetFile(path)
                try:
                    row_offset = 0
                    for batch in parquet.iter_batches(
                        batch_size=self.read_batch_size, columns=list(MAGICBRUSH_COLUMNS), use_threads=False,
                    ):
                        indexed_rows = list(enumerate(batch.to_pylist(), row_offset))
                        row_offset += batch.num_rows
                        if self.shuffle:
                            rng.shuffle(indexed_rows)
                        for row_index, row in indexed_rows:
                            sample_id = self.file_offsets[path] + row_index
                            if sample_id in excluded_ids or not matches_inter_edit_language(row, self.language):
                                continue
                            mask_value = row["mask_img"]
                            if mask_value is None or (isinstance(mask_value, dict) and
                                                      mask_value.get("bytes") is None and not mask_value.get("path")):
                                sample, reason = None, "missing_edit_mask"
                            else:
                                sample = self._process_sample(row, sample_id, path)
                                reason = "empty_edit_mask"
                            if sample is None:
                                excluded_ids.add(sample_id)
                                self._record_exclusion(row, sample_id, path, row_index, worker_id, reason)
                                if len(excluded_ids) <= 5 or len(excluded_ids) % 1000 == 0:
                                    logger.warning("Skipping MagicBrush sample %s: %s (rank=%s, worker=%s, skipped=%s)",
                                                   sample_id, reason, self.rank, worker_id, len(excluded_ids))
                                continue
                            yielded += 1
                            yield sample
                finally:
                    parquet.close()
            if not self.repeat:
                return
            if not yielded:
                raise ValueError(f"No valid MagicBrush samples remain for rank={self.rank}, worker={worker_id}")
