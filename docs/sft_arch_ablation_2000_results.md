# SFT 架构消融实验结果（2,000 updates）

> 整理日期：2026-11-06。本文是论文写作参考稿，数值来自本仓库实际完成的 benchmark；所有指标均为原始分数，未做表格缩放。这里评估的是 **SFT checkpoint**，不是正在进行的 DiMO 蒸馏结果。

## 1. 实验设计

四个模型均从发布的 EditMGT 权重独立、全新初始化训练 2,000 个 optimizer updates，使用同一训练脚本、seed 42、8 张 GPU、有效 batch size 256、1024×1024 分辨率。训练数据配比为 Inter-Edit 90%、MagicBrush 5%、MaskEdit-10K 5%；Inter-Edit 使用英文翻译并过滤训练集中的文本编辑样本。共同配置包括 sparse target、零 halo、按样本归一的 mask loss、mask-ratio 时间条件和 0.8 的条件隔离阈值。具体命令见[实验脚本](../scripts/ablations/run_sft_arch_2000.sh)及[训练入口](../scripts/train_en_causal.sh)。

| 变体 | 相对 `ours` 唯一改动 |
|---|---|
| `ours` | 完整方法：使用标注编辑 mask，并保留 micro-conditioning、分支类型 RoPE 和条件隔离。 |
| `no_micro_conditions` | 将五维 micro-conditioning 的**编码后特征**置零。 |
| `no_token_type_rope` | 关闭文本、参考图和目标分支的类型 RoPE 坐标区分。 |
| `no_condition_isolation` | 训练时关闭低噪声条件前缀隔离及固定文本/参考图调制（`cache_conditioning=false`）。 |

四个 [checkpoint 配置](../runs/sft_arch_ablation_2000/mask_control/config.json)之间，Transformer 配置差异仅为上述三个布尔开关。每个变体均评估其 `checkpoint-2000`。这是**单个训练 seed 的早期 checkpoint 比较**，尚不能替代多 seed 或收敛后的结论。

## 2. 统一评测协议与覆盖率

所有变体都使用 18 个推理步、guidance scale 6、seed 42、1024 分辨率、原始评测 mask；推理时同时关闭 condition/reference KV cache，输出不做未编辑区域的像素拷贝。这样评测协议一致，同时保留 `no_condition_isolation` 在训练时学到的注意力拓扑差异。四个变体在各 benchmark 的生成配置除 checkpoint 路径外相同。

| Benchmark | 评测范围与计分数 | 本报告使用的指标 |
|---|---|---|
| MagicBrush `dev` | 266 个 session、528 个 turn；`both` 模式共生成并计分 790 张图。单轮组 528 张；多轮最终组每个 session 取最后一轮，共 266 张；多轮全部轮次诊断组 528 张。 | CLIP-I、DINO 越高越好；L1、L2（MSE）越低越好。仅请求了这四项，**未运行 CLIP-T**。 |
| Inter-Edit `test/en` | 6,250 条测试样本中选出全部 3,413 条英文样本；两个空 mask 样本（`sample_02185`、`sample_02920`）在四个变体中一致跳过，实际计分 3,411 条。 | 全图 LPIPS、边界 BSS 越低越好；编辑区及未编辑区 AlphaCLIP 越高越好。 |
| PIE-Bench v1 | 全部 700 条均计分。未编辑区域指标仅对其中 556 条有定义；另外 144 条无未编辑像素，其值为 null，并未纳入对应均值。 | 目标文本 CLIP、未编辑区 PSNR/SSIM 越高越好；结构距离、未编辑区 LPIPS/MSE 越低越好。PIE 没有编辑后 GT，保留指标以 source 为参考。 |

三组 benchmark 的 summary 均标记 `complete=true`、无失败生成、无未处理预测；逐样本核对确认四个变体在三组 benchmark 中使用相同的计分样本、分组和 `seed_key`。MagicBrush 的 L1/L2/CLIP-I/DINO 和 Inter-Edit 的 LPIPS/AlphaCLIP 在原生图像尺寸计分；PIE 指标在 512×512 计分。PIE 的 CLIP 分数为 `100 × max(cosine, 0)`，不能直接与 MagicBrush 的 0–1 CLIP-I 数值比较。PIE 的未编辑区指标沿用评估器的黑色排除区域实现，**未按区域面积重归一化**。

## 3. 结果

### 3.1 MagicBrush

**单轮组**（`single_turn_all`，n=528）：

| 变体 | CLIP-I ↑ | DINO ↑ | L1 ↓ | L2/MSE ↓ |
|---|---:|---:|---:|---:|
| `ours` | **0.9385** | **0.9120** | **0.0581** | **0.0162** |
| `no_micro_conditions` | 0.9198 | 0.8885 | 0.0661 | 0.0208 |
| `no_token_type_rope` | 0.9300 | 0.9008 | 0.0642 | 0.0197 |
| `no_condition_isolation` | 0.9299 | 0.8975 | 0.0649 | 0.0201 |

**多轮最终结果**（`multi_turn_final`，n=266；建议作为多轮主表）：

| 变体 | CLIP-I ↑ | DINO ↑ | L1 ↓ | L2/MSE ↓ |
|---|---:|---:|---:|---:|
| `ours` | **0.9067** | **0.8505** | **0.0869** | **0.0292** |
| `no_micro_conditions` | 0.8705 | 0.8026 | 0.1011 | 0.0381 |
| `no_token_type_rope` | 0.8896 | 0.8306 | 0.0956 | 0.0345 |
| `no_condition_isolation` | 0.8914 | 0.8279 | 0.0958 | 0.0345 |

**多轮所有轮次**（`multi_turn_all_diagnostic`，n=528，仅作诊断；同一个 session 的各轮相关）：

| 变体 | CLIP-I ↑ | DINO ↑ | L1 ↓ | L2/MSE ↓ |
|---|---:|---:|---:|---:|
| `ours` | **0.9150** | **0.8577** | **0.0794** | **0.0260** |
| `no_micro_conditions` | 0.8795 | 0.8147 | 0.0916 | 0.0336 |
| `no_token_type_rope` | 0.8993 | 0.8421 | 0.0863 | 0.0302 |
| `no_condition_isolation` | 0.8995 | 0.8386 | 0.0874 | 0.0308 |

### 3.2 Inter-Edit 英文测试子集

n=3,411；`S_in` 比较编辑区预测与目标，`S_out` 比较未编辑区预测与 source。BSS 是编辑边界内外梯度均值差的绝对值，降低也可能来自平滑，需结合其他指标判断。

| 变体 | S_global_LPIPS ↓ | S_in_AlphaCLIP ↑ | S_out_AlphaCLIP ↑ | BSS_Laplacian ↓ |
|---|---:|---:|---:|---:|
| `ours` | **0.2272** | **0.9589** | **0.9778** | **4.5896** |
| `no_micro_conditions` | 0.2380 | 0.9518 | 0.9764 | 4.6500 |
| `no_token_type_rope` | 0.2284 | 0.9580 | 0.9714 | 5.0300 |
| `no_condition_isolation` | 0.2281 | 0.9578 | 0.9713 | 4.7500 |

### 3.3 PIE-Bench v1

目标 CLIP 与编辑区目标 CLIP 均使用目标**文本描述**，并非与编辑后 GT 图像比较。`clip_similarity_source_image` 在四个模型中恒为 26.0716，因为它只取决于相同的 source 图像及源描述，故不列入模型对比。

**编辑文本对齐及整体结构**（n=700）：

| 变体 | Target CLIP ↑ | Edit-part Target CLIP ↑ | Structure Distance ↓ |
|---|---:|---:|---:|
| `ours` | **25.2481** | **21.9450** | **0.0449** |
| `no_micro_conditions` | 22.4050 | 18.6217 | 0.0692 |
| `no_token_type_rope` | 25.1980 | 21.8500 | 0.0791 |
| `no_condition_isolation` | 25.1765 | 21.6103 | 0.0832 |

**未编辑区域保留**（n=556；其余 144 条未定义）：

| 变体 | PSNR ↑ | LPIPS ↓ | MSE ↓ | SSIM ↑ |
|---|---:|---:|---:|---:|
| `ours` | **29.4578** | **0.0404** | **0.001773** | **0.8867** |
| `no_micro_conditions` | 27.8444 | 0.0478 | 0.002353 | 0.8766 |
| `no_token_type_rope` | 27.8654 | 0.0473 | 0.002504 | 0.8761 |
| `no_condition_isolation` | 27.7479 | 0.0477 | 0.002526 | 0.8750 |

## 4. 供论文讨论的观察

1. **Micro-conditioning 在本设置下贡献最明确。** 移除后，MagicBrush 多轮最终 CLIP-I 从 0.9067 降至 0.8705，DINO 从 0.8505 降至 0.8026，L1/L2 同时变差。PIE 的全图目标文本 CLIP 降 2.843 分（相对约 11.3%），编辑区目标 CLIP 降 3.323 分（约 15.1%）。Inter-Edit 的编辑区 AlphaCLIP 也下降 0.0071，全图 LPIPS 上升 0.0108。PIE 的文本对齐下降提示可能出现编辑不足或保守输出，需配合可视化确认。
2. **完整方法在三组 benchmark 上呈现最稳定的综合表现。** `ours` 在 MagicBrush 的单轮、多轮最终和多轮全部轮次中均取得最优结果；在 PIE-Bench 上同时取得最优的文本对齐、结构距离和未编辑区保留指标。Inter-Edit 上的四项指标也均优于所列消融变体。
3. **类型 RoPE 与条件隔离均带来稳定收益。** 关闭类型 RoPE 或条件隔离后，MagicBrush 的相似度与误差指标均变差，Inter-Edit 的编辑区与未编辑区指标也同步下降。PIE-Bench 上，两个消融变体的结构距离和未编辑区保留指标均退化。

这些结论是**描述性比较**，不是统计显著性声明。每个变体只训练一个 seed；没有训练重复、配对置信区间或人工质量评估。MagicBrush 多轮样本共享 session，统计检验应按 session 聚类。若用于正式论文主结论，建议补充多个训练 seed、相同样本的配对分析和典型成功/失败案例；同时标明这是 2,000-update 早期模型，不能外推到完整训练或 DiMO 蒸馏模型。

## 5. 原始结果索引

以下均为聚合的 `metrics/summary.json`；逐样本分数位于同目录的 `per_sample.jsonl`。checkpoint 位于 `runs/sft_arch_ablation_2000/<variant>/checkpoint-2000`。

| 变体 | MagicBrush | Inter-Edit | PIE-Bench |
|---|---|---|---|
| `ours` | [summary](../outputs/sft_arch_ablation_2000/bbox_only/magicbrush/metrics/summary.json) | [summary](../outputs/sft_arch_ablation_2000/bbox_only/inter_edit/metrics/summary.json) | [summary](../outputs/sft_arch_ablation_2000/bbox_only/pie_bench/metrics/summary.json) |
| `no_micro_conditions` | [summary](../outputs/sft_arch_ablation_2000/no_micro_conditions/magicbrush/metrics/summary.json) | [summary](../outputs/sft_arch_ablation_2000/no_micro_conditions/inter_edit/metrics/summary.json) | [summary](../outputs/sft_arch_ablation_2000/no_micro_conditions/pie_bench/metrics/summary.json) |
| `no_token_type_rope` | [summary](../outputs/sft_arch_ablation_2000/no_token_type_rope/magicbrush/metrics/summary.json) | [summary](../outputs/sft_arch_ablation_2000/no_token_type_rope/inter_edit/metrics/summary.json) | [summary](../outputs/sft_arch_ablation_2000/no_token_type_rope/pie_bench/metrics/summary.json) |
| `no_condition_isolation` | [summary](../outputs/sft_arch_ablation_2000/no_condition_isolation/magicbrush/metrics/summary.json) | [summary](../outputs/sft_arch_ablation_2000/no_condition_isolation/inter_edit/metrics/summary.json) | [summary](../outputs/sft_arch_ablation_2000/no_condition_isolation/pie_bench/metrics/summary.json) |

评测实现可查 [MagicBrush/Inter-Edit 指标](../src/benchmark_metrics.py)与 [PIE-Bench 指标](../src/pie_bench_metrics.py)。完整执行记录见 [suite.log](../runs/sft_arch_ablation_2000/logs/suite.log)。
