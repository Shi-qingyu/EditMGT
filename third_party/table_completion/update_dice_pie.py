"""Fill DICE's PIE-Bench block only after its local full split completes."""

import json
from pathlib import Path

from update_realtime_tables import mean, replace_row


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/real_time_baselines/dice_paella/pie_bench/metrics/summary.json"
TABLE = ROOT / "docs/pie_inter.tex"


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if not (summary["complete"] and summary["selected_predictions"] == 700
            and summary["scored_predictions"] == 700 and not summary["failed_predictions"]):
        raise ValueError("DICE PIE-Bench summary is incomplete")
    cells = ["32+32", f'{summary["timing"]["mean_seconds"]:.2f}/N/R']
    cells += [f'{mean(summary, k):.2f}' for k in
              ("clip_similarity_target_image", "clip_similarity_target_image_edit_part")]
    cells += [f'{mean(summary, "structure_distance"):.4f}',
              f'{mean(summary, "psnr_unedit_part"):.2f}']
    cells += [f'{mean(summary, k):.4f}' for k in
              ("lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part")]
    cells += ["N/R"] * 4
    table = TABLE.read_text(encoding="utf-8")
    table = replace_row(table, "DICE", cells)
    table = table.replace(
        "% DICE/GIDE: user-provided Table 1, PIE-Bench quality metrics; timings await remeasurement.",
        "% GIDE: user-provided Table 1 PIE-Bench metrics; DICE: local full PIE-Bench evaluation.",
    ).replace(
        "DICE and GIDE use source-reported PIE-Bench quality results.",
        "DICE (Paella) uses local full-split results with source and target captions; GIDE uses source-reported quality results.",
    )
    TABLE.write_text(table, encoding="utf-8")
    print("DICE full PIE-Bench block updated from local results")


if __name__ == "__main__":
    main()
