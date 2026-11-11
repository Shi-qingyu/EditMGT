"""Check table structure and local vanilla scores against completed summaries."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/dimo_step4000_sft8000_comparison/vanilla36"
REALTIME = ROOT / "outputs/real_time_baselines"
REALTIME_ROWS = {
    "DreamLite-mobile": "dreamlite_mobile",
    "ChordEdit": "chordedit_corrected",
    "TurboEdit": "turboedit",
    "SwiftEdit": "swiftedit",
    "Sparse-LaViDa": "sparse_lavida",
    "SpotEdit (FLUX.2 Klein 4B)": "spotedit_flux2_4b",
    "FLUX.2 klein 4B": "flux2_klein",
    "UltraEdit": "ultraedit",
    "Lumina-DiMOO": "lumina_dimoo",
    "MMaDA-Parallel": "mmada_parallel_a",
    "NEP": "nep_512",
}


def table_rows(name: str, width: int) -> tuple[dict[str, list[str]], int]:
    source = (ROOT / "docs" / name).read_text(encoding="utf-8")
    rows = {}
    unavailable = 0
    for match in re.finditer(r"(?m)^\s*((?:[^\n]*?)(?:\n[^\n]*?)?)\\\\", source):
        candidate = " ".join(match.group(1).split())
        if "\\cite{" not in candidate and not candidate.startswith("Ours &"):
            continue
        cells = [cell.strip() for cell in candidate.split("&")]
        assert len(cells) == width, (name, cells[0], len(cells), width)
        assert all(cell != "--" for cell in cells), (name, cells[0])
        unavailable += cells.count("N/R")
        rows[cells[0]] = cells
    assert rows, f"No model rows found in {name}"
    return rows, unavailable


def summary(*parts: str) -> dict:
    result = json.loads((OUTPUT.joinpath(*parts) / "metrics/summary.json").read_text())
    assert result["complete"] and result["scored_predictions"] == result["expected_predictions"], parts
    assert result["failed_predictions"] == 0, parts
    return result


def check_rounded(cells: list[str], start: int, values: list[float]) -> None:
    for cell, value in zip(cells[start:], values, strict=True):
        decimals = len(cell.split(".", 1)[1]) if "." in cell else 0
        assert abs(float(cell) - value) <= 0.5 * 10 ** -decimals + 1e-10, (
            cell,
            value,
        )


def main() -> None:
    magic_rows, magic_nr = table_rows("magicbrush.tex", 11)
    pie_rows, pie_nr = table_rows("pie_inter.tex", 14)
    magic = summary("magicbrush_test")
    single = magic["by_protocol"]["single_turn_all"]["metrics"]
    multi = magic["by_protocol"]["multi_turn_final"]["metrics"]
    magic_values = [block[key]["mean"] for block in (single, multi) for key in
                    ("l1", "l2", "clip-i", "dino")]
    magic_row = magic_rows["EditMGT~\\cite{editmgt2026}"]
    check_rounded(magic_row[3:], 0, magic_values)

    pie = summary("pie_bench")["overall"]
    inter = summary("inter_edit")["overall"]
    pie_keys = ("clip_similarity_target_image", "clip_similarity_target_image_edit_part",
                "structure_distance", "psnr_unedit_part", "lpips_unedit_part",
                "mse_unedit_part", "ssim_unedit_part")
    inter_keys = ("S_global_LPIPS", "S_in_AlphaCLIP", "S_out_AlphaCLIP",
                  "BSS_Laplacian")
    pie_values = [pie[key]["mean"] for key in pie_keys]
    inter_values = [inter[key]["mean"] for key in inter_keys]
    pie_row = pie_rows["EditMGT~\\cite{editmgt2026}"]
    check_rounded(pie_row[3:], 0, pie_values + inter_values)

    completed = 0
    for label, directory in REALTIME_ROWS.items():
        summaries = {}
        for benchmark in ("magicbrush", "inter_edit", "pie_bench"):
            path = REALTIME / directory / benchmark / "metrics/summary.json"
            if path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
                assert result["complete"] and not result["failed_predictions"], path
                summaries[benchmark] = result
        if "magicbrush" in summaries:
            result = summaries["magicbrush"]
            assert result["selected_predictions"] == 1571
            row = next(cells for key, cells in magic_rows.items() if key.startswith(label + "~\\cite{"))
            check_rounded([row[2]], 0, [result["timing"]["mean_seconds"]])
            one = result["by_protocol"]["single_turn_all"]["metrics"]
            multi = result["by_protocol"]["multi_turn_final"]["metrics"]
            values = [block[key]["mean"] for block in (one, multi)
                      for key in ("l1", "l2", "clip-i", "dino")]
            check_rounded(row[3:], 0, values)
            completed += 1
        if "pie_bench" in summaries and "inter_edit" in summaries:
            pb, ie = summaries["pie_bench"], summaries["inter_edit"]
            assert pb["selected_predictions"] == 700
            assert ie["selected_predictions"] == 3413 and ie["scored_predictions"] == 3411
            row = next(cells for key, cells in pie_rows.items() if key.startswith(label + "~\\cite{"))
            latencies = row[2].split("/")
            assert len(latencies) == 2
            check_rounded(latencies, 0, [pb["timing"]["mean_seconds"], ie["timing"]["mean_seconds"]])
            values = [pb["overall"][key]["mean"] for key in pie_keys]
            values += [ie["overall"][key]["mean"] for key in inter_keys]
            check_rounded(row[3:], 0, values)
            completed += 2
    print(f"magicbrush.tex: {len(magic_rows)} method rows, {magic_nr} N/R cells")
    print(f"pie_inter.tex: {len(pie_rows)} method rows, {pie_nr} N/R cells")
    print("Native EditMGT table values match complete local summaries")
    print(f"{completed} completed real-time result sets match local summaries")


if __name__ == "__main__":
    main()
