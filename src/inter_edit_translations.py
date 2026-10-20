"""Validate and overlay a completed Chinese-to-English Inter-Edit release."""
import json
from pathlib import Path
import re


HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")


def needs_translation(row, split="train"):
    """Respect official test labels, including English prompts quoting Chinese."""
    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return False
    if split == "test":
        language = str(row.get("language", "")).strip().lower()
        if language in ("en", "english", "英文"):
            return False
        if language in ("cn", "zh", "chinese", "中文") or language.startswith(("zh-", "zh_")):
            return True
    return bool(HAN.search(instruction))


class InterEditTranslations:
    """Load one small translation shard at a time, before filtering/tokenization."""

    def __init__(self, directory, source_root, metadata_files, split="train"):
        self.split = split
        self.root = Path(directory).resolve()
        self.summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.config = json.loads((self.root / "run_config.json").read_text(encoding="utf-8"))
        if (self.summary.get("complete") is not True or self.summary.get("pending") != 0
                or self.summary.get("selected") != self.summary.get("translated")
                or self.config.get("limit") is not None):
            raise ValueError(f"Loading requires complete, unlimited Inter-Edit translations: {self.root}")
        if self.config.get("split", "train") != split:
            raise ValueError("Inter-Edit translation split does not match the requested data")
        if Path(self.config[f"source_{split}_dir"]).resolve() != Path(source_root).resolve():
            raise ValueError("Inter-Edit translation source directory does not match the requested data")
        inputs = {item["name"]: item for item in self.config["inputs"]}
        if set(inputs) != {path.name for path in metadata_files}:
            raise ValueError("Inter-Edit translation metadata shard set does not match the requested data")
        for path in metadata_files:
            stat = path.stat()
            expected = inputs[path.name]
            if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError(f"Inter-Edit source metadata changed since translation: {path}")

    def shard_path(self, source):
        if self.split == "test":
            return self.root / "translations.jsonl"
        return self.root / "shards" / Path(source).name.removesuffix(".gz")

    def cache_identity(self, source):
        path = self.shard_path(source)
        if not path.is_file():
            return f"translation:{path}:missing"
        stat = path.stat()
        return f"translation:{path}:{stat.st_size}:{stat.st_mtime_ns}"

    def apply(self, source, indexed_rows):
        path = self.shard_path(source)
        translated = {}
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    index = item["row_index"]
                    if index in translated or item["metadata_file"] != Path(source).name:
                        raise ValueError(f"Duplicate or mismatched translation record in {path}")
                    translated[index] = item
        result = []
        for index, row in indexed_rows:
            sample_id = row["index"] if self.split == "test" else row["sample_id"]
            instruction = row.get("instruction", "")
            item = translated.pop(index, None)
            if item is not None:
                if not needs_translation(row, self.split):
                    raise ValueError(f"Unexpected Inter-Edit translation for original English sample {sample_id} in {path}")
                english = item.get("instruction_en")
                if (item["sample_id"] != sample_id or item["instruction_zh"] != instruction
                        or not isinstance(english, str) or not english.strip()
                        or not re.search("[A-Za-z]", english)):
                    raise ValueError(f"Invalid Inter-Edit translation for sample {sample_id} in {path}")
                row = {**row, "instruction_original": instruction, "instruction": english.strip(), "language": "English",
                       **({"language_original": row["language"]} if "language" in row else {})}
            elif needs_translation(row, self.split):
                raise ValueError(f"Missing Inter-Edit translation for sample {sample_id} in {path}")
            result.append(row)
        if translated:
            raise ValueError(f"Translation records do not belong to source rows: {path}")
        return result
