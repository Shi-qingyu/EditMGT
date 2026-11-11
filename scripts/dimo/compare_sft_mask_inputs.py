"""Qualitative SFT-8000 comparison: no mask, tight bbox, original mask.

Selects reproducible Inter-Edit test/en cases, excluding text edits. All
source/target/input-mask/output images remain separate files.
"""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.benchmark_data import BenchmarkImages
from src.benchmark_utils import InferenceRunner, atomic_json, safe_name, sample_seed, save_png
from src.dimo.benchmark import parse_args as benchmark_args
from src.inter_edit_dataset import is_inter_edit_text_editing
from src.mask_utils import normalize_edit_mask_image, resize_binary_edit_mask


ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "outputs/dimo_step4000_sft8000_comparison/sft18/inter_edit/selection.json"
DEFAULT_OUTPUT = ROOT / "outputs/sft_mask_input_comparison/inter_edit_test_en"
DEFAULT_CHECKPOINT = ROOT / "runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-8000"
TYPES = ("Add", "Local", "Remove", "Texture")
CONDITIONS = (("no_mask", "无 mask"), ("bbox", "bbox"), ("mask", "原始 mask"))
RESOLUTION = 1024


def _manifest(output):
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def prepare(args):
    manifest_path = args.output_dir / "manifest.json"
    old = _manifest(args.output_dir) if manifest_path.exists() else None
    if old is not None:
        if (old["seed"], old["per_type"]) != (args.seed, args.per_type):
            raise ValueError("Existing selection uses a different seed or per-type count")
        if not args.refresh_selection:
            print(f"Reusing {len(old['samples'])} prepared cases", flush=True)
            return
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    by_type = {name: [] for name in TYPES}
    for task in selection:
        category = task.get("edit_type")
        if (category in by_type and not is_inter_edit_text_editing(task)
                and not re.search(r"[\u4e00-\u9fff]", task["instruction"])):
            digest = hashlib.sha256(f"{args.seed}:{task['key']}".encode()).digest()
            by_type[category].append((digest, task))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolver = BenchmarkImages(ROOT / "data/inter_edit_tar_index")
    samples = []
    try:
        for category in TYPES:
            accepted = 0
            for _, task in sorted(by_type[category], key=lambda item: item[0]):
                if not task.get("mask"):
                    continue
                mask = normalize_edit_mask_image(resolver.edit_mask(task["mask"], "inter_edit"))
                model_mask = resize_binary_edit_mask(mask, (RESOLUTION, RESOLUTION))
                box = model_mask.getbbox()
                if box is None:
                    continue
                source = resolver.image(task["source"], "RGB")
                target = resolver.image(task["target"], "RGB")
                bbox_mask = Image.new("L", model_mask.size, 0)
                ImageDraw.Draw(bbox_mask).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
                folder = args.output_dir / f"{category.lower()}_{safe_name(task['key'])}"
                if folder.exists():
                    if not args.refresh_selection:
                        raise FileExistsError(folder)
                    if not all((folder / filename).is_file() for filename in
                               ("source.png", "target.png", "mask.png", "bbox.png")):
                        raise ValueError(f"Incomplete existing input set: {folder}")
                else:
                    folder.mkdir()
                    save_png(folder / "source.png", source)
                    save_png(folder / "target.png", target)
                    save_png(folder / "mask.png", mask)
                    save_png(folder / "bbox.png", bbox_mask)
                samples.append({
                    "name": folder.name, "key": task["key"], "test_index": task["test_index"],
                    "edit_type": category, "instruction": task["instruction"],
                    "seed": sample_seed(args.seed, task["seed_key"]),
                    "mask_fraction_1024": float(np.asarray(model_mask, dtype=np.uint8).mean() / 255),
                    "bbox_fraction_1024": float(np.asarray(bbox_mask, dtype=np.uint8).mean() / 255),
                    "bbox_xyxy_1024": list(box),
                })
                accepted += 1
                if accepted == args.per_type:
                    break
            if accepted != args.per_type:
                raise RuntimeError(f"Only {accepted}/{args.per_type} nonempty {category} masks")
    finally:
        resolver.close()
    if old is not None and args.refresh_selection:
        selected = {row["name"] for row in samples}
        archive = args.output_dir / "_excluded_initial_selection"
        for previous in old["samples"]:
            if previous["name"] not in selected:
                archive.mkdir(exist_ok=True)
                (args.output_dir / previous["name"]).rename(archive / previous["name"])
    atomic_json(manifest_path, {
        "dataset": "Inter-Edit test/en", "selection": "lowest SHA256(seed:key) per edit type after filtering",
        "filters": ["Inter-Edit text editing detector", "Han characters in inference instruction"],
        "seed": args.seed, "per_type": args.per_type,
        "checkpoint": str(args.checkpoint.resolve()), "inference_steps": 18, "cfg": 6,
        "resolution": RESOLUTION, "condition_cache_threshold": 0.8,
        "preserve_unmasked_pixels": False,
        "bbox_protocol": "tight rectangle of binarized annotation after resize to 1024; oracle region",
        "samples": samples,
    })
    print(f"Prepared {len(samples)} cases in {args.output_dir}", flush=True)


def generate(args):
    manifest = _manifest(args.output_dir)
    if str(args.checkpoint.resolve()) != manifest["checkpoint"]:
        raise ValueError("Checkpoint differs from prepared manifest")
    benchmark, _ = benchmark_args([
        "--benchmark", "inter_edit", "--backend", "teacher",
        "--checkpoint", str(args.checkpoint), "--model-path", str(args.base_model),
        "--guidance-scale", "6", "--num-inference-steps", "18", "--bf16",
        "--condition-cache-threshold", "0.8", "--reference-kv-cache-threshold", "0.8",
        "--seed", str(args.seed), "--warmup", "0", "--device", args.device,
        "--output-dir", str(args.output_dir / "_benchmark_unused"), "--stage", "generate",
    ])
    runner = InferenceRunner(benchmark)
    try:
        for index, row in enumerate(manifest["samples"], 1):
            folder = args.output_dir / row["name"]
            with Image.open(folder / "source.png") as source_image:
                source = source_image.convert("RGB")
            for condition, _ in CONDITIONS:
                output = folder / f"{condition}_output.png"
                record = folder / f"{condition}_output.json"
                if output.is_file() and record.is_file():
                    continue
                benchmark.mask_mode = "none" if condition == "no_mask" else "provided"
                if condition == "no_mask":
                    mask = None
                else:
                    with Image.open(folder / f"{condition}.png") as mask_image:
                        mask = mask_image.copy()
                image, info = runner.generate(source, mask, row["instruction"], row["seed"])
                save_png(output, image)
                atomic_json(record, {"key": row["key"], "condition": condition, "info": info})
            print(f"{index}/{len(manifest['samples'])}: {row['name']}", flush=True)
    finally:
        runner.close()


def _mean(values):
    return sum(values) / len(values) if values else None


def report(args):
    manifest = _manifest(args.output_dir)
    scores = []
    for row in manifest["samples"]:
        folder = args.output_dir / row["name"]
        with Image.open(folder / "source.png") as image:
            source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        with Image.open(folder / "target.png") as image:
            target = np.asarray(image.convert("RGB").resize((source.shape[1], source.shape[0])), dtype=np.float32) / 255
        with Image.open(folder / "mask.png") as image:
            mask = resize_binary_edit_mask(normalize_edit_mask_image(image), source.shape[:2])
            inside = np.asarray(mask, dtype=np.uint8) > 0
        outside = ~inside
        for condition, _ in CONDITIONS:
            path = folder / f"{condition}_output.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                prediction = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            pred_target = np.abs(prediction - target)
            pred_source = np.abs(prediction - source)
            source_target = np.abs(source - target)
            scores.append({
                "key": row["key"], "edit_type": row["edit_type"], "condition": condition,
                "full_l1_to_target": float(pred_target.mean()),
                "inside_l1_to_target": float(pred_target[inside].mean()),
                "inside_l1_gain_vs_source": float(source_target[inside].mean() - pred_target[inside].mean()),
                "outside_l1_to_source": float(pred_source[outside].mean()) if outside.any() else None,
            })
    atomic_json(args.output_dir / "scores.json", scores)
    metric_keys = ("full_l1_to_target", "inside_l1_to_target", "inside_l1_gain_vs_source", "outside_l1_to_source")
    summary = {}
    for category in ("All", *TYPES):
        summary[category] = {}
        for condition, _ in CONDITIONS:
            selected = [s for s in scores if s["condition"] == condition and (category == "All" or s["edit_type"] == category)]
            summary[category][condition] = {key: _mean([s[key] for s in selected if s[key] is not None]) for key in metric_keys}
    atomic_json(args.output_dir / "summary.json", summary)

    cards = []
    for row in manifest["samples"]:
        name = html.escape(row["name"], quote=True)
        tiles = []
        for filename, label in (("source.png", "Source"), ("target.png", "GT"),
                                ("mask.png", "原始 mask"), ("bbox.png", "bbox"),
                                ("no_mask_output.png", "无 mask 输出"),
                                ("bbox_output.png", "bbox 输出"),
                                ("mask_output.png", "原始 mask 输出")):
            href = f"{name}/{filename}"
            tiles.append(f'<a class="tile" href="{href}"><span>{html.escape(label)}</span><img loading="lazy" src="{href}" alt="{html.escape(label)}"></a>')
        cards.append(f'<section><h2>{name} · {html.escape(row["edit_type"])} · #{row["test_index"]}</h2>'
                     f'<p>{html.escape(row["instruction"])}</p><div class="grid">{"".join(tiles)}</div></section>')
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SFT mask 输入对比</title>'
            '<style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#222}section{background:white;padding:16px;margin:20px 0;border-radius:8px}h2{font-size:18px;margin:0}p{margin:8px 0 14px}.grid{display:grid;grid-template-columns:repeat(7,minmax(150px,1fr));gap:10px;overflow:auto}.tile{color:inherit;text-decoration:none}.tile span{display:block;margin-bottom:4px}.tile img{width:100%;aspect-ratio:1;object-fit:contain;background:#eee}</style>'
            '<h1>Inter-Edit test/en · SFT-8000 · mask 输入对比</h1><p>每个图像独立保存；点击图片可打开原文件。bbox 为标注区域的最小外接矩形。固定 18 步、CFG 6、seed 42、1024 分辨率，不做像素合成。</p>'
            + "".join(cards) + '</html>')
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    lines = ["# Inter-Edit test/en：SFT-8000 三种区域输入对比", "",
             f"共 {len(manifest['samples'])} 张图，Add/Local/Remove/Texture 各 {manifest['per_type']} 张；固定样本、checkpoint 和逐样本 seed。",
             "从既有 test/en 选择中进一步排除了实际文本编辑任务及包含汉字的指令。样本按键的 SHA256 固定选取，没有按输出质量挑选。",
             "原始 mask 是数据标注；bbox 是其在 1024 分辨率上的紧致外接矩形，提供了目标位置。无 mask 使用整图生成路径。三个条件均为 18 步、CFG 6、BF16、条件缓存阈值 0.8，输出不做 source 像素合成。",
             "", "[逐例浏览](index.html) · [人工观察](observations.md) · [样本及 instruction](manifest.json) · [逐样本诊断分数](scores.json)", "",
             "| 类别 | 输入 | 全图对 GT L1 ↓ | 编辑区对 GT L1 ↓ | 编辑区相对 source 改善 ↑ | 未编辑区对 source L1 ↓ |",
             "|---|---|---:|---:|---:|---:|"]
    for category in ("All", *TYPES):
        for condition, label in CONDITIONS:
            values = summary[category][condition]
            fmt = lambda key: "N/A" if values[key] is None else f"{values[key]:.4f}"
            lines.append(f"| {category} | {label} | {fmt('full_l1_to_target')} | {fmt('inside_l1_to_target')} | {fmt('inside_l1_gain_vs_source')} | {fmt('outside_l1_to_source')} |")
    lines += ["", f"这些 L1 值仅作所选 {len(manifest['samples'])} 例的像素诊断；若 GT 与 source 尺寸不同，计算前将 GT 缩放到 source 尺寸。它们不能替代感知质量或编辑遵循度，本实验也不是完整测试集跑分。无 mask 路径关闭 sparse target，而 bbox 与原始 mask 路径开启它，因此差异同时包含推理区域机制的改变。", ""]
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report and independent-image browser to {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "report"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, default=ROOT / "checkpoint/EditMGT")
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.per_type < 1:
        parser.error("--per-type must be positive")
    {"prepare": prepare, "generate": generate, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
