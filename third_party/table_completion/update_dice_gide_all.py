"""Fill DICE/GIDE table rows from complete local three-benchmark runs."""

import json
from pathlib import Path

from update_realtime_tables import mean, replace_row


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/real_time_baselines"
MODELS = {
    "DICE": ("dice_paella", "32+32"),
    "GIDE": ("gide_pie", "64--192"),
}
EXPECTED = {"magicbrush": (1571, 1571), "inter_edit": (3413, 3411),
            "pie_bench": (700, 700)}


def read_complete(model, benchmark):
    model_dir = model if benchmark == "pie_bench" else model + "_blip_flan"
    path = OUTPUT / model_dir / benchmark / "metrics/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    selected, scored = EXPECTED[benchmark]
    if not (summary["complete"] and summary["selected_predictions"] == selected
            and summary["scored_predictions"] == scored
            and summary["failed_predictions"] == 0):
        raise ValueError(f"Incomplete or failed evaluation: {path}")
    return summary


def main():
    magic_path = ROOT / "docs/magicbrush.tex"
    pie_path = ROOT / "docs/pie_inter.tex"
    magic = magic_path.read_text(encoding="utf-8")
    pie = pie_path.read_text(encoding="utf-8")
    for label, (model, steps) in MODELS.items():
        mb = read_complete(model, "magicbrush")
        inter = read_complete(model, "inter_edit")
        pb = read_complete(model, "pie_bench")
        one = mb["by_protocol"]["single_turn_all"]["metrics"]
        multi = mb["by_protocol"]["multi_turn_final"]["metrics"]
        cells = [steps, f'{mb["timing"]["mean_seconds"]:.2f}']
        cells += [f'{m[k]["mean"]:.4f}' for m in (one, multi)
                  for k in ("l1", "l2", "clip-i", "dino")]
        magic = replace_row(magic, label, cells)

        cells = [steps, f'{pb["timing"]["mean_seconds"]:.2f}/{inter["timing"]["mean_seconds"]:.2f}']
        cells += [f'{mean(pb, k):.2f}' for k in
                  ("clip_similarity_target_image", "clip_similarity_target_image_edit_part")]
        cells += [f'{mean(pb, "structure_distance"):.4f}',
                  f'{mean(pb, "psnr_unedit_part"):.2f}']
        cells += [f'{mean(pb, k):.4f}' for k in
                  ("lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part")]
        cells += [f'{mean(inter, k):.4f}' for k in
                  ("S_global_LPIPS", "S_in_AlphaCLIP", "S_out_AlphaCLIP")]
        cells += [f'{mean(inter, "BSS_Laplacian"):.2f}']
        pie = replace_row(pie, label, cells)
    magic = magic.replace(
        "% DICE/GIDE need paired source/target descriptions (GIDE also subject words);\n"
        "% the local 1571-example MagicBrush test selection has instructions only.",
        "% DICE/GIDE MagicBrush prompts are generated from the supplied source image\n"
        "% and edit instruction; see third_party/table_completion/provenance.md.",
    )
    magic = magic.replace(
        "Other available baseline scores are published.",
        "DICE and GIDE use source-image captions and instruction-derived target prompts. "
        "Other available baseline scores are published.",
    )
    pie = pie.replace(
        "DICE (Paella) and GIDE use local full-split results with source and target captions; GIDE also uses subject annotations.",
        "DICE (Paella) and GIDE use local full-split results. On Inter-Edit, their "
        "source captions and target prompts are derived from the supplied source image "
        "and edit instruction; GIDE also uses instruction-derived subject words.",
    )
    magic_path.write_text(magic, encoding="utf-8")
    pie_path.write_text(pie, encoding="utf-8")
    print("Updated DICE and GIDE rows from complete local MagicBrush, Inter-Edit, and PIE-Bench runs")


if __name__ == "__main__":
    main()
