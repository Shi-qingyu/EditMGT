"""Stream MaskEdit-10K train parquet with aligned source/target/white edit masks."""
import io
import json
import logging
import os
from pathlib import Path
import random

from PIL import Image
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from src.dataset_utils import process_edit_sample, tokenize_prompt
from src.inter_edit_dataset import matches_inter_edit_language


DEFAULT_MASKEDIT_ROOT = Path(__file__).resolve().parents[1] / "data/MaskEdit-10K"
MASKEDIT_CONFIGS = ("scene", "infographics_en")
logger = logging.getLogger(__name__)


class MaskEditDataset(IterableDataset):
    """Disjoint rank/worker file ownership and bounded image-containing batches.

    Exclude the Chinese-infographic configuration and Chinese content labels.
    Only English instructions without Han characters enter training, including
    rejection of mixed-language instructions and quoted Chinese image text.
    """

    def __init__(self, root_dir=DEFAULT_MASKEDIT_ROOT, tokenizer=None, size=1024,
                 text_encoder_architecture="CLIP", norm=False, rank=0, world_size=1,
                 shuffle=True, repeat=True, seed=42, configs=MASKEDIT_CONFIGS,
                 prompt_field="edit_instruction", read_batch_size=4, cache_dir=None):
        super().__init__()
        if world_size < 1 or not 0 <= rank < world_size or read_batch_size < 1:
            raise ValueError("Invalid MaskEdit rank/world size/read batch size")
        if not configs or len(set(configs)) != len(configs) or set(configs) - set(MASKEDIT_CONFIGS):
            raise ValueError(f"MaskEdit configs must be a nonempty subset of {MASKEDIT_CONFIGS}")
        if prompt_field not in ("edit_instruction", "prompt"):
            raise ValueError("MaskEdit prompt field must be edit_instruction or prompt")
        self.root = Path(root_dir)
        self.data_dir = self.root / "data" if (self.root / "data").is_dir() else self.root
        self.configs, self.prompt_field = tuple(configs), prompt_field
        self.tokenizer, self.size, self.norm = tokenizer, size, norm
        self.text_encoder_architecture = text_encoder_architecture
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.shuffle, self.repeat, self.read_batch_size = shuffle, repeat, read_batch_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_MASKEDIT_ROOT.parent / "maskedit_cache"
        self.files, self.file_offsets, self.file_configs, self.file_columns = [], {}, {}, {}
        self.num_samples, offset = 0, 0
        columns = {"id", "source", "target", "mask", prompt_field}
        all_files = []
        for config in self.configs:
            files = sorted((self.data_dir / config).glob("train-*.parquet"))
            if not files:
                raise FileNotFoundError(f"No MaskEdit train shards for config={config} under {self.data_dir}")
            all_files.extend((config, path) for path in files)
        for file_index, (config, path) in enumerate(all_files):
            parquet = pq.ParquetFile(path)
            try:
                missing = columns - set(parquet.schema_arrow.names)
                if missing:
                    raise ValueError(f"MaskEdit shard {path} is missing fields: {sorted(missing)}")
                self.file_offsets[path], self.file_configs[path] = offset, config
                language_columns = [prompt_field]
                if "content_language" in parquet.schema_arrow.names:
                    language_columns.append("content_language")
                self.file_columns[path] = sorted(columns | set(language_columns))
                count = parquet.metadata.num_rows
                offset += count
                if file_index % world_size != rank:
                    continue
                # Count only eligible text rows; image columns are not read here.
                eligible = sum(
                    self._is_english(row)
                    for batch in parquet.iter_batches(batch_size=1024, columns=language_columns, use_threads=False)
                    for row in batch.to_pylist()
                )
                if eligible:
                    self.files.append(path)
                    self.num_samples += eligible
            finally:
                parquet.close()
        if not self.files:
            raise ValueError(f"No English non-Chinese MaskEdit samples assigned to rank {rank}")

    def __len__(self):
        return self.num_samples

    def _is_english(self, row):
        content_language = str(row.get("content_language", "")).strip().lower()
        if content_language in ("zh", "chinese", "中文", "cn") or content_language.startswith(("zh-", "zh_")):
            return False
        # content_language describes image text, not instruction language.
        return matches_inter_edit_language({"instruction": row[self.prompt_field]}, "en")

    def _image(self, value, shard):
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
                path = next((item for item in candidates if item.is_file()), candidates[0])
            with Image.open(path) as image:
                return image.copy()
        raise TypeError(f"Unsupported MaskEdit image value in {shard}: {type(value)!r}")

    def _process_sample(self, row, sample_id, path):
        instruction = row[self.prompt_field]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Missing MaskEdit instruction for {row['id']} in {path}")
        instruction = instruction.replace("[MASK_AREA]", "the masked area")
        result = process_edit_sample(self._image(row["target"], path), self._image(row["source"], path),
                                     self._image(row["mask"], path), self.size, self.norm,
                                     ensure_edit_in_crop=True)
        if not result["edit_mask"].any():
            return None
        # Use numeric IDs like Inter-Edit/MagicBrush so mixed batches collate.
        result.update(sample_id=sample_id, instruction=instruction, edit_type="MaskEdit-10K")
        if self.tokenizer is not None:
            tokens = tokenize_prompt(self.tokenizer, instruction, self.text_encoder_architecture)
            result["prompt_input_ids"] = [value[0] for value in tokens] if isinstance(self.tokenizer, list) else tokens[0]
        return result

    def _record_exclusion(self, path, row, sample_id, worker_id, reason):
        root = self.cache_dir / "empty_masks/train"
        root.mkdir(parents=True, exist_ok=True)
        record = {"dataset": "MaskEdit-10K", "sample_id": sample_id, "original_id": row["id"],
                  "config": self.file_configs[path], "shard": str(path), "reason": reason,
                  "rank": self.rank, "worker_id": worker_id}
        with (root / f"rank_{self.rank:04d}_worker_{worker_id:02d}_pid_{os.getpid()}.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __iter__(self):
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        files = list(self.files[worker_id::worker_count])
        if not files:
            raise ValueError("MaskEdit has fewer rank-local train shards than DataLoader workers")
        rng = random.Random(self.seed + 131071 + 1000003 * self.rank + worker_id)
        excluded = set()
        while True:
            if self.shuffle:
                rng.shuffle(files)
            yielded = 0
            for path in files:
                parquet = pq.ParquetFile(path)
                try:
                    offset = 0
                    for batch in parquet.iter_batches(batch_size=self.read_batch_size, columns=self.file_columns[path], use_threads=False):
                        rows = list(enumerate(batch.to_pylist(), offset))
                        offset += batch.num_rows
                        if self.shuffle:
                            rng.shuffle(rows)
                        for row_index, row in rows:
                            sample_id = self.file_offsets[path] + row_index
                            if sample_id in excluded or not self._is_english(row):
                                continue
                            mask = row["mask"]
                            missing_mask = mask is None or (isinstance(mask, dict) and
                                            mask.get("bytes") is None and not mask.get("path"))
                            result = None if missing_mask else self._process_sample(row, sample_id, path)
                            if result is None:
                                excluded.add(sample_id)
                                reason = "missing_edit_mask" if missing_mask else "empty_edit_mask"
                                self._record_exclusion(path, row, sample_id, worker_id, reason)
                                if len(excluded) <= 5 or len(excluded) % 1000 == 0:
                                    logger.warning("Skipping MaskEdit sample %s: %s", row["id"], reason)
                                continue
                            yielded += 1
                            yield result
                finally:
                    parquet.close()
            if not self.repeat:
                return
            if not yielded:
                raise ValueError(f"No valid MaskEdit samples remain for rank={self.rank}, worker={worker_id}")
