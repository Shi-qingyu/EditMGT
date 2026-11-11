"""Fill GIDE's PIE-Bench block only after its adapted full split completes."""

import json
from pathlib import Path

from update_realtime_tables import mean, replace_row


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/real_time_baselines/gide_pie/pie_bench/metrics/summary.json"
DICE_SUMMARY = ROOT / "outputs/real_time_baselines/dice_paella/pie_bench/metrics/summary.json"
TABLE = ROOT / "docs/pie_inter.tex"


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if not (summary["complete"] and summary["selected_predictions"] == 700
            and summary["scored_predictions"] == 700 and not summary["failed_predictions"]):
        raise ValueError("GIDE PIE-Bench summary is incomplete")
    cells = ["64--192", f'{summary["timing"]["mean_seconds"]:.2f}/N/R']
    cells += [f'{mean(summary, k):.2f}' for k in
              ("clip_similarity_target_image", "clip_similarity_target_image_edit_part")]
    cells += [f'{mean(summary, "structure_distance"):.4f}',
              f'{mean(summary, "psnr_unedit_part"):.2f}']
    cells += [f'{mean(summary, k):.4f}' for k in
              ("lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part")]
    cells += ["N/R"] * 4
    table = TABLE.read_text(encoding="utf-8")
    table = replace_row(table, "GIDE", cells)
    dice_local = DICE_SUMMARY.is_file() and json.loads(
        DICE_SUMMARY.read_text(encoding="utf-8")
    ).get("complete", False)
    if dice_local:
        comment = "% DICE and GIDE PIE-Bench blocks: local full-split evaluations."
        caption = "DICE (Paella) and GIDE use local full-split results with source and target captions; GIDE also uses subject annotations."
    else:
        comment = "% DICE: user-provided PIE-Bench quality; GIDE: local full-split evaluation."
        caption = "DICE uses source-reported PIE-Bench quality; GIDE uses local full-split results with source and target captions plus subject annotations."
    for old in (
        "% DICE/GIDE: user-provided Table 1, PIE-Bench quality metrics; timings await remeasurement.",
        "% GIDE: user-provided Table 1 PIE-Bench metrics; DICE: local full PIE-Bench evaluation.",
    ):
        table = table.replace(old, comment)
    for old in (
        "DICE and GIDE use source-reported PIE-Bench quality results.",
        "DICE uses local full PIE-Bench results; GIDE uses source-reported quality results.",
        "DICE uses a local full PIE-Bench evaluation with the Paella backbone; GIDE uses source-reported quality results.",
        "DICE (Paella) uses local full-split results with source and target captions; GIDE uses source-reported quality results.",
    ):
        table = table.replace(old, caption)
    TABLE.write_text(table, encoding="utf-8")
    print("GIDE full PIE-Bench block updated from local results")


if __name__ == "__main__":
    main()
