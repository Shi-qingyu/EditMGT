"""Write a provenance-rich Markdown report for complete DICE/GIDE runs."""

from pathlib import Path

from update_dice_gide_all import MODELS, read_complete


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/dice_gide_three_benchmarks.md"


def val(block, metric, digits=4):
    return f'{block[metric]["mean"]:.{digits}f}'


def main():
    summaries = {label: {benchmark: read_complete(model, benchmark)
                         for benchmark in ("magicbrush", "inter_edit", "pie_bench")}
                 for label, (model, _) in MODELS.items()}
    lines = [
        "# DICE and GIDE on MagicBrush, Inter-Edit and PIE-Bench",
        "",
        "All rows below are local full-split evaluations with one prediction for each selected",
        "test sample. DICE uses the released Paella backbone (32 inversion plus 32 editing",
        "steps). GIDE uses the local released pipeline with Lumina-DiMOO, SAM 2 and SAM 3.",
        "Timing is the CUDA-synchronized mean model pipeline time per image; it excludes",
        "image I/O, warmup and metric computation.",
        "",
        "For PIE-Bench, both models receive the benchmark's source and target captions;",
        "GIDE also receives subject words from the provided annotation when available.",
        "MagicBrush and Inter-Edit supply an edit instruction but no paired captions.",
        "For those two benchmarks, BLIP captions the provided source image, FLAN-T5",
        "rewrites the caption using the instruction, and the original instruction is",
        "appended to the target prompt so the requested change is retained. GIDE subject",
        "words and add/remove type are inferred from the instruction. MagicBrush standalone",
        "turns caption their supplied source; multi-turn prompts propagate the previous",
        "text prompt. Target images and benchmark edit masks are never used to make these",
        "prompts. These are adapted input protocols, so their scores should be labeled as",
        "such when compared with models that directly take only the instruction.",
        "",
        "## MagicBrush test",
        "",
        "Full selection: 1,571 turns. Multi-turn metrics use each session's final output.",
        "",
        "| Model | Latency (s) | Single L1↓ | Single L2↓ | Single CLIP-I↑ | Single DINO↑ | Multi L1↓ | Multi L2↓ | Multi CLIP-I↑ | Multi DINO↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, blocks in summaries.items():
        s = blocks["magicbrush"]
        one = s["by_protocol"]["single_turn_all"]["metrics"]
        multi = s["by_protocol"]["multi_turn_final"]["metrics"]
        cells = [label, f'{s["timing"]["mean_seconds"]:.2f}']
        cells += [val(block, k) for block in (one, multi)
                  for k in ("l1", "l2", "clip-i", "dino")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "", "## Inter-Edit test/en", "",
        "Full selection: 3,413 English cases; 3,411 have nonempty masks for scoring.",
        "Masks are used for scoring only.", "",
        "| Model | Latency (s) | S_global_LPIPS↓ | S_in_AlphaCLIP↑ | S_out_AlphaCLIP↑ | BSS_Laplacian↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, blocks in summaries.items():
        s = blocks["inter_edit"]
        cells = [label, f'{s["timing"]["mean_seconds"]:.2f}']
        cells += [val(s["overall"], k, 2 if k == "BSS_Laplacian" else 4)
                  for k in ("S_global_LPIPS", "S_in_AlphaCLIP", "S_out_AlphaCLIP", "BSS_Laplacian")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "", "## PIE-Bench v1", "", "Full selection: 700 cases.", "",
        "| Model | Latency (s) | CLIP target↑ | CLIP edit↑ | Structure distance↓ | PSNR unedit↑ | LPIPS unedit↓ | MSE unedit↓ | SSIM unedit↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, blocks in summaries.items():
        s = blocks["pie_bench"]
        metrics = s["overall"]
        cells = [label, f'{s["timing"]["mean_seconds"]:.2f}']
        cells += [val(metrics, k, 2) for k in
                  ("clip_similarity_target_image", "clip_similarity_target_image_edit_part")]
        cells += [val(metrics, "structure_distance"), val(metrics, "psnr_unedit_part", 2)]
        cells += [val(metrics, k) for k in
                  ("lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part")]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "", "## Reproduction", "",
        "- Input protocol: `third_party/table_completion/prepare_dice_gide_captions.py`.",
        "- Evaluation runner: `third_party/table_completion/run_dice_gide_all.sh`.",
        "- Fixed caption manifests: `outputs/real_time_baselines/caption_protocol/`.",
        "- Full predictions, per-image timing and metric summaries:",
        "  `outputs/real_time_baselines/{dice_paella,gide_pie}_blip_flan/` for",
        "  MagicBrush and Inter-Edit, and the untagged model directories for PIE-Bench.",
        "- Table cells: `docs/magicbrush.tex` and `docs/pie_inter.tex`.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
