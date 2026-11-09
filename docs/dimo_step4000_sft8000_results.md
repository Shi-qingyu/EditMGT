# SFT-8000 与 DiMO EMA-4000 质量、vanilla EditMGT 速度基线及训练集可视化

## 实验设置

- 基模型：`runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-8000`，18 步、CFG=6；采用 checkpoint 训练的低噪声条件缓存，阈值 0.8。
- 蒸馏模型：`runs/region_dimo_v3_inter_brush_maskedit_bs256/checkpoint-4000` 的 EMA 导出；1 步使用独立 categorical 采样且无缓存，2/4 步使用 cosine MaskGIT 式迭代及 context K/V cache，CFG=1。
- **速度基线**：发布版 `checkpoint/EditMGT/editmgt`（vanilla EditMGT），36 步、CFG=6、reference strength=1.1、发布版默认 negative prompt、mask-free、无 KV cache；仅用于速度比较，质量表中保留模型行但对应分数留空。
- 2/4 步在首个 forward 捕获 context，此时 mask ratio=1，高于训练的 0.8 缓存阶段；后续分别重用 1/3 次。这是**近似的提前缓存**，不能称为严格等价的低噪声缓存。每张图的实际缓存元数据见 `records/*.json`。
- SFT 与 DiMO 质量评测采用相同的原始标注 mask、1024 分辨率、BF16、seed 42、8 张 H20、直接 VQ 解码且不做源图像素合成；vanilla 速度评测使用同样的硬件、分辨率、精度和 seed，但遵循其原生 mask-free 推理协议。
- MagicBrush 使用 dev 和 test 集的 single/multi both 模式；Inter-Edit 使用完整英文 test 子集；PIE-Bench 使用 v1 全集。基准计分样本在四种方法间逐条对齐。
- MagicBrush test 分片来自 [osunlp/MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush/tree/6c254560879df7851233ea0fd2ae2a62764d0ef2/data)，revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`。

## 质量结果

覆盖率：MagicBrush dev 790 张生成图；MagicBrush test 1571 张生成图；Inter-Edit 3411 张计分图；PIE-Bench 700 张计分图。所有结果 `complete=true`，失败生成数为零。箭头表示较优方向。

### MagicBrush dev：单轮（n=528）

| Model | CLIP-I ↑ | DINO ↑ | L1 ↓ | L2/MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| EditMGT SFT-8000 · 18 steps | 0.9360 | 0.9095 | 0.0589 | 0.0167 |
| DiMO EMA-4000 · 1 step | 0.9281 | 0.8866 | 0.0686 | 0.0215 |
| DiMO EMA-4000 · 2 steps + cache | 0.9262 | 0.8839 | 0.0693 | 0.0222 |
| DiMO EMA-4000 · 4 steps + cache | 0.9233 | 0.8816 | 0.0697 | 0.0226 |

### MagicBrush dev：多轮最终结果（n=266 sessions）

| Model | CLIP-I ↑ | DINO ↑ | L1 ↓ | L2/MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| EditMGT SFT-8000 · 18 steps | 0.8996 | 0.8436 | 0.0885 | 0.0303 |
| DiMO EMA-4000 · 1 step | 0.8842 | 0.8066 | 0.1048 | 0.0386 |
| DiMO EMA-4000 · 2 steps + cache | 0.8821 | 0.8032 | 0.1059 | 0.0397 |
| DiMO EMA-4000 · 4 steps + cache | 0.8802 | 0.8015 | 0.1067 | 0.0407 |

### MagicBrush test：单轮（n=1053）

| Model | CLIP-I ↑ | DINO ↑ | L1 ↓ | L2/MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| EditMGT SFT-8000 · 18 steps | 0.9315 | 0.9078 | 0.0606 | 0.0177 |
| DiMO EMA-4000 · 1 step | 0.9213 | 0.8838 | 0.0695 | 0.0220 |
| DiMO EMA-4000 · 2 steps + cache | 0.9202 | 0.8815 | 0.0701 | 0.0226 |
| DiMO EMA-4000 · 4 steps + cache | 0.9170 | 0.8770 | 0.0706 | 0.0231 |

### MagicBrush test：多轮最终结果（n=535 sessions）

| Model | CLIP-I ↑ | DINO ↑ | L1 ↓ | L2/MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| EditMGT SFT-8000 · 18 steps | 0.8924 | 0.8395 | 0.0897 | 0.0317 |
| DiMO EMA-4000 · 1 step | 0.8757 | 0.8012 | 0.1048 | 0.0391 |
| DiMO EMA-4000 · 2 steps + cache | 0.8728 | 0.7965 | 0.1060 | 0.0403 |
| DiMO EMA-4000 · 4 steps + cache | 0.8703 | 0.7910 | 0.1069 | 0.0412 |

### Inter-Edit test/en 与 PIE-Bench v1

Inter-Edit 的两条空 mask 注释在各方法中一致跳过，实际计分 3,411 张。PIE-Bench 没有编辑后 GT；Target CLIP 使用目标文本描述，Structure Distance 以源图为参考。PIE 未编辑区指标的有效样本数以原始 summary 的 `count/undefined_count` 为准；无未编辑像素的样本不进入对应均值。空白单元格表示当前没有查到对应质量分数。

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4">Inter-Edit test/en (n=3,411)</th>
      <th colspan="7">PIE-Bench v1 (n=700)</th>
    </tr>
    <tr>
      <th>LPIPS ↓</th>
      <th>AlphaCLIP-in ↑</th>
      <th>AlphaCLIP-out ↑</th>
      <th>BSS ↓</th>
      <th>Target CLIP ↑</th>
      <th>Edit-part Target CLIP ↑</th>
      <th>Structure Distance ↓</th>
      <th>PSNR ↑</th>
      <th>LPIPS ↓</th>
      <th>MSE ↓</th>
      <th>SSIM ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Vanilla EditMGT · 36 steps</td>
      <td></td><td></td><td></td><td></td>
      <td></td><td></td><td></td><td></td><td></td><td></td><td></td>
    </tr>
    <tr>
      <td>EditMGT SFT-8000 · 18 steps</td>
      <td>0.2259</td><td>0.9603</td><td>0.9736</td><td>4.6028</td>
      <td>25.3861</td><td>21.8593</td><td>0.0684</td><td>28.2630</td><td>0.0463</td><td>0.002310</td><td>0.8785</td>
    </tr>
    <tr>
      <td>DiMO EMA-4000 · 1 step</td>
      <td>0.2468</td><td>0.9549</td><td>0.9685</td><td>6.4444</td>
      <td>25.5202</td><td>22.0148</td><td>0.0643</td><td>24.1317</td><td>0.0644</td><td>0.008181</td><td>0.8568</td>
    </tr>
    <tr>
      <td>DiMO EMA-4000 · 2 steps + cache</td>
      <td>0.2465</td><td>0.9551</td><td>0.9676</td><td>6.7314</td>
      <td>25.5306</td><td>22.1097</td><td>0.0650</td><td>24.0364</td><td>0.0645</td><td>0.008365</td><td>0.8574</td>
    </tr>
    <tr>
      <td>DiMO EMA-4000 · 4 steps + cache</td>
      <td>0.2474</td><td>0.9541</td><td>0.9660</td><td>7.1214</td>
      <td>25.3743</td><td>21.9131</td><td>0.0660</td><td>23.9858</td><td>0.0646</td><td>0.008532</td><td>0.8583</td>
    </tr>
  </tbody>
</table>

## 速度对比

每张图在 GPU 同步前后计时，包含图像转张量、文本编码、source VQ、模型 forward 和 VQ 解码；不含原生尺寸 resize 与 mask 准备、首次模型加载、预热、数据读取、PNG 写盘和指标计算。DiMO 记录的时间扣除了与 vanilla/SFT 计时器同样位于推理管线外的准备及后处理时间。各方法每个 GPU 预热 1 张图。下表按相同样本键统计：Inter-Edit 的 vanilla 生成了 3413 张，但其中两张在 mask-conditioned 质量评测中因空 mask 跳过，速度表只取共同的 3411 张。MagicBrush 每个 split 均包含单轮与连续多轮两种模式。速度倍数按同一 benchmark 的平均延迟相除，**以 vanilla EditMGT 为 1.00×**。

| Benchmark | Model | Measured images | Mean s/image ↓ | P50 s ↓ | P95 s ↓ | Speedup vs vanilla ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MagicBrush dev | Vanilla EditMGT · 36 steps | 790 | 4.537 | 4.535 | 4.585 | 1.00× |
| MagicBrush dev | EditMGT SFT-8000 · 18 steps | 790 | 1.327 | 1.346 | 1.425 | 3.42× |
| MagicBrush dev | DiMO EMA-4000 · 1 step | 790 | 0.491 | 0.501 | 0.515 | 9.24× |
| MagicBrush dev | DiMO EMA-4000 · 2 steps + cache | 790 | 0.522 | 0.533 | 0.544 | 8.69× |
| MagicBrush dev | DiMO EMA-4000 · 4 steps + cache | 790 | 0.601 | 0.609 | 0.629 | 7.54× |
| MagicBrush test | Vanilla EditMGT · 36 steps | 1571 | 4.541 | 4.540 | 4.574 | 1.00× |
| MagicBrush test | EditMGT SFT-8000 · 18 steps | 1571 | 1.330 | 1.346 | 1.446 | 3.41× |
| MagicBrush test | DiMO EMA-4000 · 1 step | 1571 | 0.489 | 0.501 | 0.517 | 9.28× |
| MagicBrush test | DiMO EMA-4000 · 2 steps + cache | 1571 | 0.519 | 0.534 | 0.548 | 8.73× |
| MagicBrush test | DiMO EMA-4000 · 4 steps + cache | 1571 | 0.595 | 0.608 | 0.621 | 7.63× |
| Inter-Edit test/en | Vanilla EditMGT · 36 steps | 3411 | 4.545 | 4.540 | 4.600 | 1.00× |
| Inter-Edit test/en | EditMGT SFT-8000 · 18 steps | 3411 | 1.322 | 1.346 | 1.420 | 3.44× |
| Inter-Edit test/en | DiMO EMA-4000 · 1 step | 3411 | 0.479 | 0.501 | 0.511 | 9.47× |
| Inter-Edit test/en | DiMO EMA-4000 · 2 steps + cache | 3411 | 0.516 | 0.534 | 0.552 | 8.81× |
| Inter-Edit test/en | DiMO EMA-4000 · 4 steps + cache | 3411 | 0.589 | 0.609 | 0.622 | 7.71× |
| PIE-Bench v1 | Vanilla EditMGT · 36 steps | 700 | 4.551 | 4.550 | 4.599 | 1.00× |
| PIE-Bench v1 | EditMGT SFT-8000 · 18 steps | 700 | 1.443 | 1.394 | 1.638 | 3.15× |
| PIE-Bench v1 | DiMO EMA-4000 · 1 step | 700 | 0.489 | 0.501 | 0.515 | 9.31× |
| PIE-Bench v1 | DiMO EMA-4000 · 2 steps + cache | 700 | 0.521 | 0.534 | 0.548 | 8.73× |
| PIE-Bench v1 | DiMO EMA-4000 · 4 steps + cache | 700 | 0.597 | 0.610 | 0.627 | 7.62× |

## 供论文写作的观察

- DiMO 1 步相对 vanilla EditMGT 的平均逐图加速为：MagicBrush dev 9.24×、MagicBrush test 9.28×、Inter-Edit test/en 9.47×、PIE-Bench v1 9.31×。这些是包含文本编码与 VQ 的模型管线延迟，并非仅 Transformer forward。
- MagicBrush dev 单轮 CLIP-I 从 SFT 的 0.9360 降至 DiMO 1 步的 0.9281；多轮最终 CLIP-I 从 0.8996 降至 0.8842。Inter-Edit 全图 LPIPS 从 0.2259 升至 0.2468，显示这两个有 GT 的基准存在画质代价。
- 官方 MagicBrush test 单轮 CLIP-I：SFT 0.9315，DiMO 1/2/4 步 0.9213/0.9202/0.9170；多轮最终 CLIP-I：SFT 0.8924，DiMO 1 步 0.8757。
- PIE-Bench 的 DiMO 1 步目标文本 CLIP 为 25.5202，SFT 为 25.3861；未编辑区 PSNR 分别为 24.13 与 28.26 dB。文本对齐与背景保留不能用单一指标概括。
- 当前 2/4 步近似缓存设置未在 MagicBrush dev 单轮 CLIP-I 上改善 1 步结果（1/2/4 步分别为 0.9281/0.9262/0.9233）；它们增加延迟，其他指标的变化见表格。

## Qualitative results

从 Inter-Edit `test/en` 中筛选了 12 个 DiMO EMA-4000 1-step 定性案例，覆盖 Add、Local、Remove 和 Texture 四类编辑。每个案例的 source、mask、source-mask overlay、edited image 和 instruction 均以独立文件保存，见 [qualitative results](qualitative_results/README.md)。这些是筛选后的展示案例，不代表整个测试集的平均质量。

## Inter-Edit 训练集可视化素材

按 `sample_id` 的 SHA256 稳定选取 Add/Local/Remove/Texture 各 20 条，共 80 条英文训练样本。每条的 `source.png`、`mask.png`、`target.png`、`sft18.png`、`ema1.png`、`ema2.png`、`ema4.png` **分别保存为独立文件**；可在 [浏览页](../outputs/dimo_step4000_sft8000_comparison/inter_edit_train_visuals/index.html)筛选和查看。对应 instruction、样本 ID、类别和 seed 见 [manifest](../outputs/dimo_step4000_sft8000_comparison/inter_edit_train_visuals/manifest.json)。这些是训练集样本，仅用于 demo/cherry pick，不作为泛化证据。

## 解释与限制

- DiMO 的 2/4 步是从一步蒸馏模型构造的推理扩展，首步 context cache 为高噪声近似快照；质量变化既包含步数，也包含该缓存策略。
- vanilla、SFT 与 DiMO 的推理步数、mask 使用、缓存和提示词设置不同；速度表体现各自实际配置，不是隔离单一组件的因果消融。SFT 与 DiMO 的质量表也不能据此归因于单独一个训练或推理组件。
- MagicBrush 只评估 L1/L2/CLIP-I/DINO，没有 CLIP-T；Inter-Edit 只报告英文子集。PIE 的未编辑区指标对空区域未定义。
- MagicBrush/Inter-Edit 按原生尺寸计分；PIE 在 512×512 计分，文本 CLIP 指标为 `100 × max(cosine, 0)`，不能直接与 MagicBrush 的 0–1 CLIP-I 数值比较。
- 速度是当前 8×H20、BF16、batch=1 的 per-image 延迟；跨硬件或 batch size 不应直接外推。

## 原始结果

| Model | MagicBrush dev | MagicBrush test | Inter-Edit test/en | PIE-Bench v1 |
|---|---|---|---|---|
| EditMGT SFT-8000 · 18 steps | [summary](../outputs/dimo_step4000_sft8000_comparison/sft18/magicbrush/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/sft18/magicbrush_test/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/sft18/inter_edit/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/sft18/pie_bench/metrics/summary.json) |
| DiMO EMA-4000 · 1 step | [summary](../outputs/dimo_step4000_sft8000_comparison/ema1/magicbrush/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema1/magicbrush_test/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema1/inter_edit/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema1/pie_bench/metrics/summary.json) |
| DiMO EMA-4000 · 2 steps + cache | [summary](../outputs/dimo_step4000_sft8000_comparison/ema2/magicbrush/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema2/magicbrush_test/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema2/inter_edit/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema2/pie_bench/metrics/summary.json) |
| DiMO EMA-4000 · 4 steps + cache | [summary](../outputs/dimo_step4000_sft8000_comparison/ema4/magicbrush/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema4/magicbrush_test/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema4/inter_edit/metrics/summary.json) | [summary](../outputs/dimo_step4000_sft8000_comparison/ema4/pie_bench/metrics/summary.json) |

vanilla 速度原始记录：[MagicBrush dev](../outputs/dimo_step4000_sft8000_comparison/vanilla36/magicbrush/generation_summary.json)、[MagicBrush test](../outputs/dimo_step4000_sft8000_comparison/vanilla36/magicbrush_test/generation_summary.json)、[Inter-Edit test/en](../outputs/dimo_step4000_sft8000_comparison/vanilla36/inter_edit/generation_summary.json)、[PIE-Bench v1](../outputs/dimo_step4000_sft8000_comparison/vanilla36/pie_bench/generation_summary.json)。

训练集可视化索引：[manifest](../outputs/dimo_step4000_sft8000_comparison/inter_edit_train_visuals/manifest.json)。
