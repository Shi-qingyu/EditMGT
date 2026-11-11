"""Rank Inter-Edit train SFT demos for human visual review."""

import argparse
import csv
import json
from pathlib import Path

import lpips
import numpy as np
from PIL import Image
import torch


def rgb(path, size):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255


def lpips_tensor(array):
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).mul(2).sub(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = lpips.LPIPS(net="alex", verbose=False).eval()
    root = args.output_dir
    samples = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["samples"]
    scored = []
    for index, row in enumerate(samples, 1):
        folder = root / row["name"]
        output_path = folder / "sft18.png"
        if not output_path.is_file() or not (folder / "sft18.json").is_file():
            continue
        source = rgb(folder / "source.png", args.size)
        target = rgb(folder / "target.png", args.size)
        edited = rgb(output_path, args.size)
        with Image.open(folder / "mask.png") as image:
            mask = np.asarray(image.convert("L").resize((args.size, args.size), Image.Resampling.NEAREST), dtype=np.uint8) >= 128
        with torch.inference_mode():
            before = float(model(lpips_tensor(source), lpips_tensor(target)).item())
            after = float(model(lpips_tensor(edited), lpips_tensor(target)).item())
        difference = np.abs(edited - source).mean(axis=2)
        outside_mae = float(difference[~mask].mean()) if (~mask).any() else 0.0
        inside_mae = float(difference[mask].mean()) if mask.any() else 0.0
        item = {
            "name": row["name"], "sample_id": row["sample_id"],
            "edit_type": row["edit_type"], "instruction": row["instruction"],
            "mask_ratio": float(mask.mean()),
            "lpips_source_target": before, "lpips_output_target": after,
            "lpips_gain": before - after,
            "outside_mae_source_output": outside_mae,
            "inside_mae_source_output": inside_mae,
            "review_score": before - after - 0.5 * outside_mae,
        }
        scored.append(item)
        if index % 25 == 0:
            print(f"Scored {len(scored)} generated samples", flush=True)
    scored.sort(key=lambda row: row["review_score"], reverse=True)
    (root / "review_scores.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (root / "review_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]) if scored else [])
        if scored:
            writer.writeheader()
            writer.writerows(scored)
    print(f"Scored {len(scored)} of {len(samples)} generated examples", flush=True)


if __name__ == "__main__":
    main()
