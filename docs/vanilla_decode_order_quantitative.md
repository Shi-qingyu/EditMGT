# Vanilla EditMGT decoding order on Inter-Edit

Paper-ready figure: [PDF](figures/vanilla_decode_order_quantitative.pdf), [SVG](figures/vanilla_decode_order_quantitative.svg), [PNG](figures/vanilla_decode_order_quantitative.png), [LaTeX figure block](vanilla_decode_order_figure.tex). The [plotting script](../scripts/plot_vanilla_decode_order_quantitative.py) regenerates the exports from the saved trajectories; [figure data](figures/vanilla_decode_order_quantitative.json) contains all 36 curve points and their intervals.

We sampled 32 Inter-Edit test/en examples by a fixed SHA-256 ranking of sample keys (selection seed 2026), independent of outputs or edit quality. All 32 completed. The released vanilla EditMGT ran with 36 decoding steps, 1024-pixel inputs, CFG 6, reference strength 1.1, BF16, and no KV cache. All target tokens began masked. The Inter-Edit annotation mask was used **only after sampling** to label edit and context tokens; it was never passed to the model.

Commitment step is the first sampler step after which a token remains unmasked. Each image contributes one edit-region mean and one context mean. Confidence intervals are percentile bootstrap intervals over **images**, not tokens.

| Metric | Mean [95% bootstrap CI] |
|---|---:|
| Edit-region mean commitment step | 27.34 [26.07, 28.64] |
| Context mean commitment step | 22.82 [22.54, 23.07] |
| Paired edit minus context difference, steps | **4.52 [3.10, 5.98]** |
| Late-order AUC | **0.67 [0.61, 0.72]** |
| First-step entropy difference, edit minus context (nats) | 0.79 [0.51, 1.09] |

Late-order AUC is the probability that a randomly chosen edit token commits after a randomly chosen context token from the same image, with ties counted as 0.5. A random order has AUC 0.5. The edit mask covered 12.94% of tokens on average.

| After step | Edit tokens committed | Context tokens committed |
|---:|---:|---:|
| 6 | 2.11% | 3.65% |
| 12 | 8.25% | 14.23% |
| 18 | 18.64% | 31.04% |
| 24 | 31.38% | 52.96% |
| 30 | 49.84% | 77.67% |
| 36 | 100.00% | 100.00% |

The edit region committed later in 25 of 32 images (one-sided exact image-level sign test, p=0.00105). The trend was present in both text edits (8 images; 4.27-step mean difference; AUC 0.66) and other edits (24 images; 4.61-step difference; AUC 0.67). These subgroup results are exploratory.

This is evidence of an association between the user-annotated edit region and the vanilla sampler's commitment order. It does not identify the cause of that order or establish a full-test-set estimate.

Reproducibility: [capture script](../scripts/plot_mgm_motivation.py), [analysis script](../scripts/quantify_vanilla_decode_order.py), [capture metadata](../outputs/vanilla_decode_order_quant_32/metadata.json), [per-sample CSV](../outputs/vanilla_decode_order_quant_32/decode_order_per_sample.csv), and [machine-readable summary](../outputs/vanilla_decode_order_quant_32/decode_order_summary.json). The raw `trajectory.npz` files are under `outputs/vanilla_decode_order_quant_32/samples/`.
