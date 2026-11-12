# Main-table provenance and protocol audit

The table source comments previously pointed to nonexistent
`docs/magicbrush_main_table_sources.md` and
`docs/interedit_main_table_sources.md`. Record checked sources here instead.

## Local released EditMGT baseline

`run_vanilla_metrics.sh` scores complete BF16 predictions already generated
with the released `checkpoint/EditMGT` model, native mask-free inference,
36 steps, CFG 6, seed 42, 1024-pixel generation and one warmup per GPU.
MagicBrush is official `osunlp/MagicBrush` test, single-turn 1,053 and
multi-turn final 535 sessions. Inter-Edit is test/en, 3,411 scored samples
(two empty masks excluded). PIE-Bench v1 is the full 700 cases. Inference
images are generated without a mask; Inter-Edit and PIE metrics use dataset
masks during *scoring*.

| Benchmark | Raw summary | Status |
|---|---|---|
| MagicBrush | [summary](../../outputs/dimo_step4000_sft8000_comparison/vanilla36/magicbrush_test/metrics/summary.json) | complete, 1,571 predictions |
| Inter-Edit | [summary](../../outputs/dimo_step4000_sft8000_comparison/vanilla36/inter_edit/metrics/summary.json) | complete, 3,411 scored |
| PIE-Bench | [summary](../../outputs/dimo_step4000_sft8000_comparison/vanilla36/pie_bench/metrics/summary.json) | complete, 700 scored |

The [EditMGT paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Chow_EditMGT_Unleashing_Potentials_of_Masked_Generative_Transformers_in_Image_Editing_CVPR_2026_paper.pdf)
reports MagicBrush L1 0.058, CLIP-I 0.911, DINO 0.881. The
[official repository](https://github.com/weichow23/EditMGT) recommends a
mask strategy for MagicBrush, while the local native baseline is mask-free.
Its locally measured MagicBrush quality must be kept as a separate protocol;
do not combine its L2 or multi-turn values with the published quality cells.
The local SFT-8000 and DiMO EMA-4000 (Ours) rows use the annotation mask as
model input, as documented in
[`docs/dimo_step4000_sft8000_results.md`](../../docs/dimo_step4000_sft8000_results.md).

## Locally evaluated real-time baselines

DreamLite-mobile uses the official 4-step main-branch weights
`carlofkl/DreamLite-mobile`, cloned official inference code
`third_party/DreamLite` at commit `e82c28a`, BF16 on eight NVIDIA H20s.
`third_party/table_completion/run_baseline.py` sends the original editing
instruction and a 1024×1024 resized source image to the official pipeline,
then resizes the result to the native source dimensions before scoring.
Each process warms up one image. The recorded latency synchronizes CUDA
around the full pipeline call, including text encoding and image decode;
it excludes the outer source/output resize, loading, image I/O, and metrics.
MagicBrush test is 1,571 predictions, Inter-Edit test/en is 3,411 scored
predictions (two empty masks excluded), and PIE-Bench v1 is 700 cases.

| Benchmark | Raw metric summary | Mean pipeline seconds |
|---|---|---:|
| MagicBrush test | [summary](../../outputs/real_time_baselines/dreamlite_mobile/magicbrush/metrics/summary.json) | 0.4431 |
| Inter-Edit test/en | [summary](../../outputs/real_time_baselines/dreamlite_mobile/inter_edit/metrics/summary.json) | 0.4309 |
| PIE-Bench v1 | [summary](../../outputs/real_time_baselines/dreamlite_mobile/pie_bench/metrics/summary.json) | 0.4463 |

ChordEdit uses official commit `a33d8cc` with `stabilityai/sd-turbo`
weights, FP32 and its published one-step plus proximal-refinement pipeline.
The pipeline itself center-crops and resizes the source to 512 pixels. The
first local run resized outside the pipeline and is excluded; only the
`chordedit_corrected` run is used. PIE-Bench supplies source and target
captions to the two-prompt method. MagicBrush and Inter-Edit supply only an
edit instruction, so their source prompt is empty and the instruction is
used as the target prompt; no ground-truth target caption is supplied.
All three splits completed, with the same scoring code and GPU timer used
for DreamLite-mobile.

| Benchmark | ChordEdit metric summary | Mean pipeline seconds |
|---|---|---:|
| MagicBrush test | [summary](../../outputs/real_time_baselines/chordedit_corrected/magicbrush/metrics/summary.json) | 0.2890 |
| Inter-Edit test/en | [summary](../../outputs/real_time_baselines/chordedit_corrected/inter_edit/metrics/summary.json) | 0.3151 |
| PIE-Bench v1 | [summary](../../outputs/real_time_baselines/chordedit_corrected/pie_bench/metrics/summary.json) | 0.2810 |

TurboEdit uses official commit `ebf545e`, `stabilityai/sdxl-turbo`
FP16 weights, and its four-step inversion/editing code with `w1=1.5`.
The official 512-pixel resize happens before the timer; the timer includes
VAE encoding, inversion, text encoding, denoising and decode. PIE-Bench
supplies source/target captions. MagicBrush and Inter-Edit have no source
caption, so their source prompt is empty and the edit instruction is the
target prompt. All three splits completed on the eight H20s.

| Benchmark | TurboEdit metric summary | Mean pipeline seconds |
|---|---|---:|
| MagicBrush test | [summary](../../outputs/real_time_baselines/turboedit/magicbrush/metrics/summary.json) | 0.4921 |
| Inter-Edit test/en | [summary](../../outputs/real_time_baselines/turboedit/inter_edit/metrics/summary.json) | 0.4934 |
| PIE-Bench v1 | [summary](../../outputs/real_time_baselines/turboedit/pie_bench/metrics/summary.json) | 0.4922 |

SwiftEdit uses official commit `230bff1`, v1.0 release weights, the
`stabilityai/sd-turbo` components, `h94/IP-Adapter` image encoder, and the
SD 2.1-base VAE/text encoder/scheduler from the public
`sd2-community/stable-diffusion-2-1-base` mirror because the old Stability AI
repository returned 401. The adapter reproduces `infer.py` in memory; it
uses the official 512-pixel resize and FP32 model defaults. Reported time
includes one-step inversion, self-generated mask and one-step editing.
PIE-Bench supplies the official source and target captions; MagicBrush and
Inter-Edit use an empty source prompt plus the edit instruction. All three
full splits completed on eight H20s.

| Benchmark | SwiftEdit metric summary | Mean pipeline seconds |
|---|---|---:|
| MagicBrush test | [summary](../../outputs/real_time_baselines/swiftedit/magicbrush/metrics/summary.json) | 0.4818 |
| Inter-Edit test/en | [summary](../../outputs/real_time_baselines/swiftedit/inter_edit/metrics/summary.json) | 0.4817 |
| PIE-Bench v1 | [summary](../../outputs/real_time_baselines/swiftedit/pie_bench/metrics/summary.json) | 0.4821 |

The remaining local runs use these protocols:

- Native FLUX.2 Klein 4B uses the official ungated weights, Diffusers
  `Flux2KleinPipeline`, BF16, four steps, guidance 1, a 1024-pixel square
  source and the edit instruction. Its raw summaries are under
  `outputs/real_time_baselines/flux2_klein/`.
- SpotEdit uses official commit `85e3fda` and its FLUX.2 implementation on
  the same public Klein 4B weights and 1024-pixel source. The authors'
  example uses gated Klein 9B with 50 steps, four initial full-compute steps
  and reset steps 13/22/31. This local *adaptation* uses four total steps,
  one initial full-compute step, no reset steps, threshold 0.4 and guidance
  1 so selective token reuse is exercised. It clears the token-metric cache
  between images. These values must not be described as the released 9B or
  FLUX.1-Kontext setting. Raw summaries are under
  `outputs/real_time_baselines/spotedit_flux2_4b/`. The separate Kontext
  row remains `N/R` because its official checkpoint is gated and the current
  environment received a 401 response.
- Sparse-LaViDa uses official commit `3b59423`,
  `jacklishufan/sparse-lavida`, the referenced SigLIP vision encoder and
  Meissonic VQ-VAE, BF16, its published 1024-pixel editing YAML and 64-step
  cached/truncated decoding. A one-image smoke test and all three full splits
  completed. Raw summaries are written under
  `outputs/real_time_baselines/sparse_lavida/`. Its runtime uses a local
  compatibility environment with Torch 2.11 and xFormers 0.0.35; the repo
  lists Torch 2.6 and xFormers 0.0.29.post3. Treat speed as a local runtime
  measurement, and use the raw execution metadata for the exact version.

## Additional local models (11 November 2026)

The same 1,571 MagicBrush predictions, 3,413 Inter-Edit test/en selections
(3,411 scored), 700 PIE-Bench cases, common scorers, and synchronized GPU
timer are used for these adapters. These are local evaluations of the named
checkpoints; their conditioning and image resolution vary by model.

- NEP-512 completed all three splits in
  [`outputs/real_time_baselines/nep_512`](../../outputs/real_time_baselines/nep_512).
  It uses its released autoregressive generator and the benchmark edit-region
  mask as input. The checkpoint's published `GPT-XL-RAR` name maps to the
  released `R-GPT-XL` implementation by verified state-dict keys and shapes.
  The unavailable DeepFloyd Flan-T5-XL was replaced by
  `google/flan-t5-xl`; this is a local compatibility substitution, so compare
  its score with the paper only with that caveat.
- Lumina-DiMOO uses `Alpha-VLLM/Lumina-DiMOO`, the official image-editing
  template, 64 steps and a 512-pixel input. MagicBrush, Inter-Edit and
  PIE-Bench full splits completed and are recorded in both tables. The
  wrapper returned a shell parse error after the final metric write because
  `run_model.sh` had been edited while its long-running shell was open;
  all three raw summaries passed `check_completed.py` and table validation.
- MMaDA-Parallel uses the released `tyfeld/MMaDA-Parallel-A` weights, its
  official TI2TI image-editing template, 64 image steps and 128 text steps.
  Its one-image smoke test completed and its full-split evaluation runs
  after Lumina-DiMOO.
- UltraEdit uses `BleachNick/SD3_UltraEdit_w_mask`, its official mask-aware
  SD3 pipeline at 50 steps and the benchmark's edit-region mask as input.
  Its first smoke attempt failed during tokenizer loading due to a Protobuf
  mismatch. The environment has been fixed, CPU model loading succeeds, and
  a smoke plus full-split retry is queued after MMaDA-Parallel.
- DICE's released Paella path requires a source description and target
  description. PIE-Bench provides both, so the adapter uses 32 inversion and
  32 editing steps with the released Paella, VQGAN, ByT5 and CLIP-H weights.
  The local MagicBrush and Inter-Edit releases provide edit instructions only;
  their DICE cells remain unavailable pending a documented caption protocol.
  A one-image PIE-Bench smoke test completed with a valid output image. Its
  full PIE-Bench evaluation is queued after UltraEdit.
- GIDE's released pipeline requires source and target descriptions plus
  source and target subject tokens. The PIE-Bench adaptation takes paired
  captions and `blended_word` when available, then uses prompt annotations or
  a text heuristic for missing subjects. It uses local Lumina-DiMOO, SAM 2
  and SAM 3 checkpoints. Its full PIE-Bench evaluation is queued after DICE.
  This adapted input protocol must be distinguished from models that receive
  only an editing instruction; MagicBrush and Inter-Edit have no matching
  source/target caption and subject annotation protocol. A one-image
  PIE-Bench smoke test completed after enabling BF16 autocast for the SAM 3
  vision encoder. A separate functional check produced valid outputs for
  five representative PIE-Bench types: object replacement, addition,
  removal, background, and style. Its output and status JSON are under
  [`pie_bench_smoke_types`](../../outputs/real_time_baselines/gide_pie/pie_bench_smoke_types);
  those concurrent smoke timings are excluded from the table.
  When both SAM 3 and attention/SAM 2 grounding yield an empty region, the
  local adapter retries inversion on the full source frame and records
  `full_frame_after_empty_grounding` in that sample's JSON. This fallback
  uses no target image or benchmark annotation mask.
  The repository file `evaluation/evaluate_magicbrush.py` instantiates
  InstructPix2Pix, so it does not provide GIDE MagicBrush scores.

## User-provided DICE/GIDE PIE-Bench results

Before local full-split evaluation, the DICE and GIDE rows in
`docs/pie_inter.tex` were transcribed from
the user-provided screenshot of Table 1, "Quantitative comparison of different
methods on PIE-Bench and PIE-Bench++." Only its **PIE-Bench** block is used.
The screenshot's paper title/URL, exact split coverage, backbone variants and
sampling step counts are not supplied. The existing method citation keys are
retained; the screenshot itself still needs a bibliographic source.

The source caption scales Dist., LPIPS and SSIM by 100 and MSE by 1000.
Divide those displayed values by the respective factors for our table;
CLIP-G, CLIP-E and PSNR retain their source values. Map CLIP-G to Target,
CLIP-E to Edit, global Dist. to Struct., and the four **background** metrics
to PSNR/LPIPS/MSE/SSIM. The source's other global metrics are not table columns.

| Method | Target | Edit | Struct. | Background PSNR | Background LPIPS | Background MSE | Background SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| DICE | 24.70 | 21.76 | 0.1784 | 9.01 | 0.4889 | 0.16070 | 0.4700 |
| GIDE | 24.70 | 22.17 | 0.0384 | 24.37 | 0.0786 | 0.00836 | 0.7924 |

At the user's request, the screenshot's timings were excluded. The local
`update_dice_pie.py` and `update_gide_pie.py` scripts replace the PIE-Bench
block only after all 700 predictions pass scoring. Inter-Edit remains `N/R`
without a source/target caption protocol.
The former `DICE (Paella)` label was shortened to `DICE` for the screenshot,
whose backbone is unknown. The queued local DICE evaluation specifically
uses Paella; its captions and raw summary should identify that variant.

## Published full-benchmark rows

- [PnP Inversion, Table 9](https://proceedings.iclr.cc/paper_files/paper/2024/file/661caac7729aa7d8c6b8ac0d39ccbc6a-Paper-Conference.pdf)
  gives InstructPix2Pix on full PIE-Bench: target CLIP 23.61, edit CLIP 21.64,
  structure distance 57.91/1000, background PSNR 20.82, LPIPS 158.63/1000,
  MSE 227.78/10000, SSIM 76.26/100. The official
  [evaluation repository](https://github.com/cure-lab/PnPInversion) is cloned
  at `third_party/PnPInversion` (commit `07f97f448150e2ca220bebd54c8f687c5c50c67a`).
- [DICE, Tables 2 and 3](https://arxiv.org/pdf/2410.08207) gives its Paella
  variant on PIE-Bench: target CLIP 23.79, edit CLIP 21.23, structure distance
  11.34/1000, background PSNR 27.29, LPIPS 52.90/1000, MSE 43.76/10000,
  SSIM 89.79/100. Its pipeline includes discrete inversion and uses target
  descriptions; identify the row as DICE + Paella if using these values.
  These are an alternative published source and were not combined with the
  user-provided Table 1 values or the local Paella evaluation.
  The paper's 256-pixel note applies to its VQ-Diffusion variant, not the
  Paella row copied here. DICE also states that metrics across base models
  and editing algorithms are not directly comparable. Its inversion plus
  editing workload is not represented by a single published step count.
- [Inter-Edit, Table 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_Inter-Edit_First_Benchmark_for_Interactive_Instruction-Based_Image_Editing_CVPR_2026_paper.pdf)
  provides full-test scores for RNI, CJT and InstructPix2Pix. The main table's
  existing RNI/CJT values are rounded from this paper. The paper evaluates the
  full multilingual test set, while local Ours scores use English only.
- [NEP, Table 1](https://arxiv.org/pdf/2508.06044) reports MagicBrush
  region-aware single-turn and multi-turn L1/L2/CLIP-I/DINO scores. Its row
  uses editing-region masks. The IP2P, IP2P-FT, and UltraEdit rows are copied
  from the same table, which reproduces the UltraEdit values below.
- [UltraEdit, Table 4](https://www.proceedings.com/content/079/079017-0100open.pdf)
  gives MagicBrush test single-turn and multi-turn scores for multiple
  UltraEdit variants, IP2P and IP2P-FT. The MagicBrush main table uses the
  region-conditioned UltraEdit variant (single-turn L1 0.0575; multi-turn
  L1 0.0745). The old draft's IP2P multi-turn 0.16/0.060/0.79/0.62 did not
  match this source and was corrected to 0.1345/0.0460/0.8304/0.7018.

## Cross-protocol results that cannot fill these cells

- [MMaDA-Parallel](https://arxiv.org/pdf/2511.09611) reports ParaBench,
  not these MagicBrush/PIE/Inter-Edit metrics.
- [Sparse-LaViDa](https://arxiv.org/pdf/2512.14008) reports editing scores on
  ImgEdit, not MagicBrush L1/L2/CLIP-I/DINO or PIE-Bench/Inter-Edit metrics.
- [DreamLite](https://github.com/ByteVisionLab/DreamLite) reports ImgEdit
  and GEdit scores. The official repository is cloned at
  `third_party/DreamLite` (commit `e82c28a300a7adbe0f8c3b96ac7ce299651e9bb8`).
- The previously checked [GIDE paper](https://arxiv.org/abs/2603.21176) reports
  GIDE-Bench and ImgEdit, not these metrics. The screenshot values above
  cannot be attributed to that paper without verifying the screenshot's
  source and method identity.
- [SpotEdit: Selective Region Editing in Diffusion Transformers](https://openaccess.thecvf.com/content/CVPR2026/papers/Qin_SpotEdit_Selective_Region_Editing_in_Diffusion_Transformers_CVPR_2026_paper.pdf)
  reports PIE-Bench++, which is not PIE-Bench v1. There is a different work
  named SpotEdit that is an editing *benchmark*; do not confuse the two.
- [SwiftEdit](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Nguyen_SwiftEdit_Lightning_Fast_CVPR_2025_supplemental.pdf)
  reports PIE-Bench on a 100-case sample, not the full 700. Do not mix that
  row into a full-set table.

## Remaining unavailable cells

`N/R` means a value was neither measured locally nor verified from a source
using the stated dataset, split, and metric. It is not a zero. The two tables
retain all requested methods to show coverage. Rows filled with `N/R` were
not evaluated under this protocol and must not be presented as measured
comparisons. Run `python third_party/table_completion/validate_tables.py`
to count the remaining unavailable cells.

Never infer unreported values or convert scores from another dataset. Different
papers' per-image latency is not comparable without local remeasurement on the
same hardware and timer boundary.
