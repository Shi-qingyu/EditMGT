"""Preview and export separate-file qualitative results from Inter-Edit evals."""

import argparse
import hashlib
import io
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS = (
    ROOT
    / "outputs/dimo_step4000_sft8000_comparison/ema1/inter_edit/metrics/per_sample.jsonl"
)
DEFAULT_PREDICTIONS = (
    ROOT / "outputs/dimo_step4000_sft8000_comparison/ema1/inter_edit"
)
DEFAULT_CACHE = ROOT / "data/inter_edit_tar_index"
EDIT_TYPES = ("Add", "Local", "Remove", "Texture")
HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def load_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quality_score(row):
    metrics = row["metrics"]
    return (
        metrics["S_global_LPIPS"]
        + 2.0 * (1.0 - metrics["S_in_AlphaCLIP"])
        + 2.0 * (1.0 - metrics["S_out_AlphaCLIP"])
    )


class AssetStore:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.open_archives = {}
        self.indices = {}

    def _archive(self, spec):
        path = Path(spec["root"]) / spec["archive"]
        if path not in self.open_archives:
            stat = path.stat()
            identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
            index_path = self.cache_dir / f"{path.stem}-{digest}.json"
            if not index_path.is_file():
                raise FileNotFoundError(f"Missing tar index: {index_path}")
            self.indices[path] = json.loads(index_path.read_text())
            self.open_archives[path] = path.open("rb")
        return path, self.open_archives[path], self.indices[path]

    def image(self, spec):
        extracted = Path(spec["root"]) / spec["member"]
        if extracted.is_file():
            with Image.open(extracted) as image:
                return image.copy()
        path, handle, index = self._archive(spec)
        if spec["member"] not in index:
            raise KeyError(f"Missing {spec['member']!r} in {path}")
        offset, size = index[spec["member"]]
        handle.seek(offset)
        payload = handle.read(size)
        if len(payload) != size:
            raise OSError(f"Truncated read for {spec['member']!r} in {path}")
        with Image.open(io.BytesIO(payload)) as image:
            return image.copy()

    def close(self):
        for handle in self.open_archives.values():
            handle.close()
        self.open_archives.clear()
        self.indices.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def normalize_mask(mask, size=None):
    mask = ImageOps.exif_transpose(mask).convert("L")
    threshold = 1 if mask.getextrema()[1] <= 1 else 128
    mask = mask.point(lambda value: 255 if value >= threshold else 0)
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask


def mask_ratio(mask):
    histogram = mask.histogram()
    return histogram[255] / (mask.width * mask.height)


def overlay(source, mask):
    source = ImageOps.exif_transpose(source).convert("RGB")
    mask = normalize_mask(mask, source.size)
    red = Image.new("RGB", source.size, (255, 35, 35))
    alpha = mask.point(lambda value: 112 if value else 0)
    result = Image.composite(red, source, alpha)

    # A bright outline keeps narrow edit regions legible after downsampling.
    expanded = mask.filter(ImageFilter.MaxFilter(9))
    boundary = Image.frombytes(
        "L",
        mask.size,
        bytes(max(a - b, 0) for a, b in zip(expanded.tobytes(), mask.tobytes())),
    )
    yellow = Image.new("RGB", source.size, (255, 230, 30))
    return Image.composite(yellow, result, boundary)


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def fit(image, size):
    canvas = Image.new("RGB", size, "white")
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    canvas.paste(contained, (left, top))
    return canvas


def shortlist(rows, assets, per_type, min_mask_ratio, max_mask_ratio):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("edit_type") not in EDIT_TYPES or HAN.search(row.get("instruction", "")):
            continue
        grouped[row["edit_type"]].append(row)
    selected = []
    for edit_type in EDIT_TYPES:
        accepted = 0
        for row in sorted(grouped[edit_type], key=quality_score):
            mask = normalize_mask(assets.image(row["mask"]))
            ratio = mask_ratio(mask)
            if not min_mask_ratio <= ratio <= max_mask_ratio:
                continue
            row = dict(row)
            row["mask_ratio"] = ratio
            selected.append(row)
            accepted += 1
            if accepted == per_type:
                break
        if accepted != per_type:
            raise RuntimeError(f"Only found {accepted}/{per_type} candidates for {edit_type}")
    return selected


def prediction_path(row, prediction_root):
    return prediction_root / row["prediction"]


def save_preview(rows, assets, prediction_root, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["edit_type"]].append(row)
    tile = (300, 300)
    gap = 12
    caption_height = 86
    heading_height = 48
    label_font = font(18)
    small_font = font(14)
    for edit_type in EDIT_TYPES:
        group = grouped[edit_type]
        width = gap + 3 * (tile[0] + gap)
        height = heading_height + gap + len(group) * (tile[1] + caption_height + gap)
        sheet = Image.new("RGB", (width, height), "#f3f3f3")
        draw = ImageDraw.Draw(sheet)
        for column, name in enumerate(("Source + mask", "DiMO EMA-4000 (1 step)", "Target")):
            draw.text((gap + column * (tile[0] + gap), 12), name, fill="#111", font=label_font)
        y = heading_height
        for row in group:
            source = assets.image(row["source"])
            mask = assets.image(row["mask"])
            target = assets.image(row["target"])
            with Image.open(prediction_path(row, prediction_root)) as prediction:
                panels = (overlay(source, mask), prediction.copy(), target)
            for column, panel in enumerate(panels):
                x = gap + column * (tile[0] + gap)
                sheet.paste(fit(panel, tile), (x, y))
            metrics = row["metrics"]
            caption = (
                f"{row['key']}  mask={row['mask_ratio']:.1%}  "
                f"LPIPS={metrics['S_global_LPIPS']:.3f}  "
                f"Sin={metrics['S_in_AlphaCLIP']:.3f}  "
                f"Sout={metrics['S_out_AlphaCLIP']:.3f}"
            )
            draw.text((gap, y + tile[1] + 7), caption, fill="#111", font=small_font)
            instruction = row["instruction"]
            if len(instruction) > 115:
                instruction = instruction[:112] + "..."
            draw.text((gap, y + tile[1] + 32), instruction, fill="#333", font=small_font)
            y += tile[1] + caption_height + gap
        sheet.save(output_dir / f"{edit_type.lower()}_candidates.jpg", quality=92)
    (output_dir / "candidates.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def export(rows, keys, assets, prediction_root, output_dir):
    by_key = {row["key"]: row for row in rows}
    unknown = [key for key in keys if key not in by_key]
    if unknown:
        raise KeyError(f"Unknown sample keys: {unknown}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for number, key in enumerate(keys, 1):
        row = by_key[key]
        name = f"{number:02d}_{row['edit_type'].lower()}_{key}"
        folder = output_dir / name
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Output folder is nonempty: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        source = ImageOps.exif_transpose(assets.image(row["source"])).convert("RGB")
        mask = normalize_mask(assets.image(row["mask"]), source.size)
        source.save(folder / "source.png")
        mask.save(folder / "mask.png")
        overlay(source, mask).save(folder / "source_mask_overlay.png")
        with Image.open(prediction_path(row, prediction_root)) as prediction:
            ImageOps.exif_transpose(prediction).convert("RGB").save(folder / "edited.png")
        (folder / "instruction.txt").write_text(row["instruction"] + "\n", encoding="utf-8")
        metrics = row["metrics"]
        manifest.append(
            {
                "order": number,
                "name": name,
                "sample_key": key,
                "edit_type": row["edit_type"],
                "instruction": row["instruction"],
                "source": f"{name}/source.png",
                "mask": f"{name}/mask.png",
                "source_mask_overlay": f"{name}/source_mask_overlay.png",
                "edited": f"{name}/edited.png",
                "instruction_file": f"{name}/instruction.txt",
                "mask_ratio": mask_ratio(mask),
                "metrics": metrics,
                "model": "DiMO EMA-4000, 1 step",
                "benchmark": "Inter-Edit test/en",
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps({"samples": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "export"))
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=8)
    parser.add_argument("--min-mask-ratio", type=float, default=0.01)
    parser.add_argument("--max-mask-ratio", type=float, default=0.35)
    parser.add_argument("--keys", nargs="*")
    args = parser.parse_args()
    if args.mode == "export" and not args.keys:
        parser.error("export requires --keys")
    rows = load_rows(args.records)
    with AssetStore(args.cache_dir) as assets:
        if args.mode == "preview":
            candidates = shortlist(
                rows,
                assets,
                args.per_type,
                args.min_mask_ratio,
                args.max_mask_ratio,
            )
            save_preview(candidates, assets, args.prediction_root, args.output_dir)
        else:
            export(rows, args.keys, assets, args.prediction_root, args.output_dir)


if __name__ == "__main__":
    main()
