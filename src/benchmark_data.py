"""Benchmark-only image access and MagicBrush session metadata.

No training crops, repetition, language heuristics or empty-mask exclusions are
applied here. Missing/invalid examples remain visible in evaluation counts.
"""
import io
import json
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

from PIL import Image

from src.benchmark_utils import file_identity, safe_name
from src.inter_edit_dataset import TarAssetReader
from src.mask_utils import decode_magicbrush_edit_mask, normalize_edit_mask_image


def path_spec(path):
    return {"kind": "file", "path": str(Path(path).resolve())}


class BenchmarkImages:
    """Bounded tar handles and streamed parquet rows, shared by both stages."""
    def __init__(self, tar_cache_dir=None):
        self.tar_cache_dir = tar_cache_dir
        self.tars = {}
        self.parquet_streams = OrderedDict()
        self.row_cache = OrderedDict()

    def _parquet_row(self, path, index, column):
        import pyarrow.parquet as pq

        key = (path, index, column)
        if key in self.row_cache:
            self.row_cache.move_to_end(key)
            return self.row_cache[key]
        stream_key = (path, column)
        state = self.parquet_streams.get(stream_key)
        if state is None or index < state["next"]:
            if state is not None:
                state["file"].close()
            parquet = pq.ParquetFile(path)
            if column not in parquet.schema_arrow.names:
                parquet.close()
                raise ValueError(f"Missing MagicBrush parquet image column: {column}")
            # Reading source must not also load GT/mask bytes into memory. In
            # particular, native generation has no dependency on mask_img.
            state = {"file": parquet, "iterator": parquet.iter_batches(
                batch_size=1, columns=[column], use_threads=False,
            ), "next": 0}
            self.parquet_streams[stream_key] = state
        self.parquet_streams.move_to_end(stream_key)
        # Two files with up to three image columns each. Keeping all columns of
        # the current file avoids repeatedly rescanning it with --save-inputs.
        while len(self.parquet_streams) > 6:
            _, previous = self.parquet_streams.popitem(last=False)
            previous["file"].close()
        row = None
        while state["next"] <= index:
            row = next(state["iterator"]).to_pylist()[0]
            state["next"] += 1
        self.row_cache[key] = row
        while len(self.row_cache) > 8:
            self.row_cache.popitem(last=False)
        return row

    @staticmethod
    def _convert_image(image, mode):
        # A raw detached copy preserves alpha for dataset-specific mask decoding.
        return image.copy() if mode is None else image.convert(mode)

    def image(self, spec, mode=None):
        if not spec:
            raise ValueError("Missing image specification")
        if spec["kind"] == "tar":
            root = spec["root"]
            if root not in self.tars:
                self.tars[root] = TarAssetReader(root, self.tar_cache_dir)
            return self._convert_image(self.tars[root].image(spec["archive"], spec["member"]), mode)
        if spec["kind"] == "file":
            with Image.open(spec["path"]) as image:
                return self._convert_image(image, mode)
        if spec["kind"] != "parquet":
            raise ValueError(f"Unknown image specification: {spec['kind']}")
        value = self._parquet_row(spec["file"], spec["row"], spec["column"])[spec["column"]]
        if isinstance(value, dict):
            value = value.get("bytes") if value.get("bytes") is not None else value.get("path")
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(value)) as image:
                return self._convert_image(image, mode)
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                candidates = [Path(spec["root"]) / path, Path(spec["file"]).parent / path]
                path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            with Image.open(path) as image:
                return self._convert_image(image, mode)
        raise ValueError(f"Missing/invalid {spec['column']} for parquet row {spec['row']}")

    def edit_mask(self, spec, benchmark):
        image = self.image(spec)
        if benchmark == "magicbrush":
            return decode_magicbrush_edit_mask(image)
        return normalize_edit_mask_image(image)

    def close(self):
        for reader in self.tars.values():
            reader.close()
        for state in self.parquet_streams.values():
            state["file"].close()
        self.tars.clear()
        self.parquet_streams.clear()
        self.row_cache.clear()


def _resolve_image(root, session_id, name):
    path = Path(name)
    if path.is_absolute():
        return path_spec(path)
    candidates = [root / path, root / "images" / path, root / "images" / session_id / path,
                  root / session_id / path]
    return path_spec(next((candidate for candidate in candidates if candidate.is_file()), candidates[0]))


def _session_rows(payload):
    if isinstance(payload, dict) and "sessions" in payload:
        payload = payload["sessions"]
    if isinstance(payload, dict):
        for session_id, turns in payload.items():
            if not isinstance(turns, list):
                raise ValueError("Expected official {session_id: [edit turns]} JSON or a flat list of turns")
            for number, turn in enumerate(turns, 1):
                yield str(session_id), number, turn
    elif isinstance(payload, list):
        for turn in payload:
            image_name = turn.get("input", turn.get("source_img", ""))
            session_id = turn.get("img_id", turn.get("session_id"))
            if session_id is None:
                session_id = Path(str(image_name)).name.split("-")[0]
            if not session_id or "turn_index" not in turn:
                raise ValueError("Flat manifests require img_id/session_id (or an input filename) and turn_index")
            yield str(session_id), int(turn["turn_index"]), turn
    else:
        raise ValueError("Unsupported MagicBrush manifest format")


def magicbrush_sessions(data_dir, split="dev", manifest=None):
    root = Path(data_dir).resolve()
    if root.name in ("train", "dev", "test") and root.name != split:
        raise ValueError(f"Requested split {split!r}, but data directory is named {root.name!r}")
    if (root / split).is_dir():
        root = root / split
    if manifest is None:
        candidates = [root / "edit_sessions.json", root / f"{split}.json", root / f"{split}.jsonl"]
        manifest = next((path for path in candidates if path.is_file()), None)
    sessions = OrderedDict()
    identities = []
    if manifest is not None:
        manifest = Path(manifest).resolve()
        identities.append(file_identity(manifest))
        if manifest.suffix == ".jsonl":
            payload = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        for session_id, number, row in _session_rows(payload):
            source = row.get("input", row.get("source_img"))
            if not isinstance(source, str):
                raise ValueError("JSON source images must be paths; use parquet input for embedded image bytes")
            target = row.get("output", row.get("target_img", f"{session_id}-output{number}.png"))
            mask = row.get("mask", row.get("mask_img", f"{session_id}-mask{number}.png"))
            caption = row.get("target_caption", row.get("global_description"))
            sessions.setdefault(session_id, []).append({
                "session_id": session_id, "turn": number, "instruction": row["instruction"],
                "source": _resolve_image(root, session_id, source),
                "target": _resolve_image(root, session_id, target),
                "mask": _resolve_image(root, session_id, mask) if mask else None,
                "target_name": Path(target).name, "caption": caption,
            })
        format_name = "official edit_sessions / explicit JSON manifest"
    else:
        import pyarrow.parquet as pq

        parquet_root = root / "data" if (root / "data").is_dir() else root
        files = sorted(parquet_root.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No {split} edit_sessions.json or {split}-*.parquet under {root}. "
                "The local Hugging Face release may contain only train/dev; supply the official test release."
            )
        for path in files:
            identities.append(file_identity(path))
            with closing(pq.ParquetFile(path)) as parquet:
                fields = set(parquet.schema_arrow.names)
                required = {"img_id", "turn_index", "instruction", "source_img", "target_img"}
                if not required <= fields:
                    raise ValueError(f"Missing MagicBrush parquet fields: {sorted(required - fields)}")
                columns = [name for name in ("img_id", "turn_index", "instruction", "target_caption", "global_description") if name in fields]
                row_index = 0
                for batch in parquet.iter_batches(batch_size=1024, columns=columns, use_threads=False):
                    for row in batch.to_pylist():
                        session_id, number = str(row["img_id"]), int(row["turn_index"])
                        def spec(column):
                            return {"kind": "parquet", "root": str(root), "file": str(path), "row": row_index, "column": column}
                        sessions.setdefault(session_id, []).append({
                            "session_id": session_id, "turn": number, "instruction": row["instruction"],
                            "source": spec("source_img"), "target": spec("target_img"),
                            "mask": spec("mask_img") if "mask_img" in fields else None,
                            "target_name": f"{session_id}-output{number}.png",
                            "caption": row.get("target_caption", row.get("global_description")),
                        })
                        row_index += 1
        format_name = "local parquet, metadata-only selection with streamed image rows"
    # Some converted releases use zero-based turn indices; normalize whole sessions.
    for session_id, turns in sessions.items():
        if min(turn["turn"] for turn in turns) == 0:
            for turn in turns:
                turn["turn"] += 1
                if turn["source"]["kind"] == "parquet":
                    turn["target_name"] = f"{session_id}-output{turn['turn']}.png"
        turns.sort(key=lambda turn: turn["turn"])
        if len({turn["turn"] for turn in turns}) != len(turns):
            raise ValueError(f"Duplicate turns in session {session_id}")
        if not all(isinstance(turn["instruction"], str) and turn["instruction"].strip() for turn in turns):
            raise ValueError(f"Empty instruction in session {session_id}")
    return sessions, {"root": str(root), "split": split, "format": format_name,
                      "metadata_files": identities, "total_sessions": len(sessions),
                      "total_turns": sum(len(turns) for turns in sessions.values())}


def magicbrush_tasks(sessions, mode="both", limit=None):
    selection = []
    chosen = list(sessions.items())[:limit]
    for session_id, turns in chosen:
        if [turn["turn"] for turn in turns] != list(range(1, len(turns) + 1)):
            raise ValueError(f"Benchmark evaluation requires complete consecutive turns in {session_id}")
        slug = safe_name(session_id)
        previous_key = None
        for index, turn in enumerate(turns):
            number, final = turn["turn"], index == len(turns) - 1
            variants = ["first"] if index == 0 else (["single", "multi"] if mode == "both" else [mode])
            current_multi_key = None
            for variant in variants:
                groups = []
                if mode in ("single", "both") and variant in ("first", "single"):
                    groups.append("single_turn_all")
                if mode in ("multi", "both") and variant in ("first", "multi"):
                    groups.append("multi_turn_all_diagnostic")
                    if final:
                        groups.append("multi_turn_final")
                suffix = "1" if variant == "first" else ("inde_" if variant == "single" else "iter_") + str(number)
                key = f"{slug}__{suffix}"
                task = {**turn, "key": key, "variant": variant, "groups": groups, "language": "en",
                        "seed_key": f"magicbrush:{session_id}:{number}",
                        "prediction": f"images/{slug}/{slug}_{suffix}.png"}
                if variant == "multi":
                    task["input_prediction_key"] = previous_key
                if variant in ("first", "multi"):
                    current_multi_key = key
                selection.append(task)
            if mode in ("multi", "both"):
                previous_key = current_multi_key
    return selection, len(chosen)


def attach_magicbrush_captions(selection, caption_file):
    if caption_file is None:
        return
    captions = json.loads(Path(caption_file).read_text(encoding="utf-8"))
    for task in selection:
        nested = captions.get(task["session_id"], {})
        value = nested.get(task["target_name"]) if isinstance(nested, dict) else None
        if value is None:
            value = captions.get(task["target_name"])
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid target caption for {task['key']}")
            task["caption"] = value
