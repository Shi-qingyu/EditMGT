"""Fill real-time baseline rows from complete local benchmark summaries."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/real_time_baselines"
MODELS = {
    "chordedit": ("chordedit_corrected", "ChordEdit", r"$1^{\dagger}$"),
    "turboedit": ("turboedit", "TurboEdit", "4"),
    "flux2_klein": ("flux2_klein", "FLUX.2 klein 4B", "4"),
    "swiftedit": ("swiftedit", "SwiftEdit", r"$1^{\dagger}$"),
    "sparse_lavida": ("sparse_lavida", "Sparse-LaViDa", "64"),
    "spotedit_flux2_4b": ("spotedit_flux2_4b", "SpotEdit (FLUX.2 Klein 4B)", "4"),
    "ultraedit": ("ultraedit", "UltraEdit", "50"),
    "lumina_dimoo": ("lumina_dimoo", "Lumina-DiMOO", "64"),
    "mmada_parallel_a": ("mmada_parallel_a", "MMaDA-Parallel", "64"),
    "nep_512": ("nep_512", "NEP", "AR"),
}


def read(model_dir, benchmark):
    path = OUTPUT / model_dir / benchmark / "metrics/summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary["complete"] or summary["failed_predictions"]:
        return None
    if summary["selected_predictions"] != {"magicbrush": 1571, "inter_edit": 3413, "pie_bench": 700}[benchmark]:
        return None
    return summary


def mean(summary, key):
    return summary["overall"][key]["mean"]


def replace_row(tex, label, cells):
    pattern = re.compile(r"(?m)^(\s*" + re.escape(label) + r"~\\cite\{[^}]+\}\s*&)[\s\S]*?\\\\$")
    replacement = lambda match: match.group(1) + " " + " & ".join(cells) + r" \\" 
    updated, count = pattern.subn(replacement, tex)
    if count != 1:
        raise ValueError(f"Expected exactly one {label} row, got {count}")
    return updated


def main():
    magic_path = ROOT / "docs/magicbrush.tex"
    pie_path = ROOT / "docs/pie_inter.tex"
    magic = magic_path.read_text(encoding="utf-8")
    pie = pie_path.read_text(encoding="utf-8")
    for _, (model_dir, label, steps) in MODELS.items():
        mb = read(model_dir, "magicbrush")
        inter = read(model_dir, "inter_edit")
        pb = read(model_dir, "pie_bench")
        if mb:
            one = mb["by_protocol"]["single_turn_all"]["metrics"]
            multi = mb["by_protocol"]["multi_turn_final"]["metrics"]
            cells = [steps, f'{mb["timing"]["mean_seconds"]:.2f}']
            cells += [f'{m[k]["mean"]:.4f}' for m in (one, multi) for k in ("l1", "l2", "clip-i", "dino")]
            magic = replace_row(magic, label, cells)
        if pb and inter:
            cells = [steps, f'{pb["timing"]["mean_seconds"]:.2f}/{inter["timing"]["mean_seconds"]:.2f}']
            cells += [f'{mean(pb, k):.2f}' for k in ("clip_similarity_target_image", "clip_similarity_target_image_edit_part")]
            cells += [f'{mean(pb, "structure_distance"):.4f}', f'{mean(pb, "psnr_unedit_part"):.2f}']
            cells += [f'{mean(pb, k):.4f}' for k in ("lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part")]
            cells += [f'{mean(inter, k):.4f}' for k in ("S_global_LPIPS", "S_in_AlphaCLIP", "S_out_AlphaCLIP")]
            cells += [f'{mean(inter, "BSS_Laplacian"):.2f}']
            pie = replace_row(pie, label, cells)
    magic_path.write_text(magic, encoding="utf-8")
    pie_path.write_text(pie, encoding="utf-8")


if __name__ == "__main__":
    main()
