# DICE and GIDE on MagicBrush, Inter-Edit and PIE-Bench

All rows below are local full-split evaluations with one prediction for each selected
test sample. DICE uses the released Paella backbone (32 inversion plus 32 editing
steps). GIDE uses the local released pipeline with Lumina-DiMOO, SAM 2 and SAM 3.
Timing is the CUDA-synchronized mean model pipeline time per image; it excludes
image I/O, warmup and metric computation.

For PIE-Bench, both models receive the benchmark's source and target captions;
GIDE also receives subject words from the provided annotation when available.
MagicBrush and Inter-Edit supply an edit instruction but no paired captions.
For those two benchmarks, BLIP captions the provided source image, FLAN-T5
rewrites the caption using the instruction, and the original instruction is
appended to the target prompt so the requested change is retained. GIDE subject
words and add/remove type are inferred from the instruction. MagicBrush standalone
turns caption their supplied source; multi-turn prompts propagate the previous
text prompt. Target images and benchmark edit masks are never used to make these
prompts. These are adapted input protocols, so their scores should be labeled as
such when compared with models that directly take only the instruction.

## MagicBrush test

Full selection: 1,571 turns. Multi-turn metrics use each session's final output.

| Model | Latency (s) | Single L1↓ | Single L2↓ | Single CLIP-I↑ | Single DINO↑ | Multi L1↓ | Multi L2↓ | Multi CLIP-I↑ | Multi DINO↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DICE | 8.35 | 0.0651 | 0.0213 | 0.8989 | 0.8527 | 0.0978 | 0.0355 | 0.8590 | 0.7749 |
| GIDE | 54.76 | 0.0978 | 0.0329 | 0.8815 | 0.8122 | 0.1285 | 0.0483 | 0.8492 | 0.7430 |

## Inter-Edit test/en

Full selection: 3,413 English cases; 3,411 have nonempty masks for scoring.
Masks are used for scoring only.

| Model | Latency (s) | S_global_LPIPS↓ | S_in_AlphaCLIP↑ | S_out_AlphaCLIP↑ | BSS_Laplacian↓ |
|---|---:|---:|---:|---:|---:|
| DICE | 8.12 | 0.4301 | 0.9340 | 0.9629 | 3.51 |
| GIDE | 49.54 | 0.3948 | 0.9327 | 0.9509 | 3.92 |

## PIE-Bench v1

Full selection: 700 cases.

| Model | Latency (s) | CLIP target↑ | CLIP edit↑ | Structure distance↓ | PSNR unedit↑ | LPIPS unedit↓ | MSE unedit↓ | SSIM unedit↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DICE | 8.10 | 24.84 | 21.72 | 0.0111 | 27.45 | 0.0444 | 0.0042 | 0.9073 |
| GIDE | 56.96 | 26.04 | 22.94 | 0.0474 | 23.62 | 0.0908 | 0.0121 | 0.7777 |

## Reproduction

- Input protocol: `third_party/table_completion/prepare_dice_gide_captions.py`.
- Evaluation runner: `third_party/table_completion/run_dice_gide_all.sh`.
- Fixed caption manifests: `outputs/real_time_baselines/caption_protocol/`.
- Full predictions, per-image timing and metric summaries:
  `outputs/real_time_baselines/{dice_paella,gide_pie}_blip_flan/` for
  MagicBrush and Inter-Edit, and the untagged model directories for PIE-Bench.
- Table cells: `docs/magicbrush.tex` and `docs/pie_inter.tex`.
