# Vanilla EditMGT 的文本与 source image 特征相似度

## 图与可复现数据

- 主图：[PDF](vanilla_condition_similarity.pdf) · [SVG](vanilla_condition_similarity.svg) · [PNG](vanilla_condition_similarity.png)：上排是任意两个 Transformer block 的相似度，下排是任意两个采样 step 的相似度；左侧文本，右侧 source image。上排虚线标出双流（block 1–14）与单流（block 15–42）的边界。
- 细节图：[PDF](vanilla_condition_adjacent.pdf) · [SVG](vanilla_condition_adjacent.svg) · [PNG](vanilla_condition_adjacent.png)：左图比较相邻 block，右图比较相邻 step。
- 数值：[平均矩阵](vanilla_condition_similarity_matrices.npz) · [统计摘要](summary.json)；[可复现采集与绘图脚本](../../scripts/plot_vanilla_condition_feature_similarity.py)；每张图的原始统计保存在 `outputs/vanilla_feature_similarity_32/*.npz`。

## 方法

采用先前固定 SHA-256 排序选出的 **32 张 Inter-Edit test/en 样本**，全部完成；选择独立于模型输出。运行发布版 vanilla EditMGT（`checkpoint/EditMGT`），图像 1024 × 1024，36 个采样 step、CFG=6、reference strength=1.1、BF16，无参考图像/条件 KV cache；不提供编辑 mask。沿用先前实验各样本的输入图像、编辑指令和随机种子。模型配置、样本键及 SHA-256 来源见 `outputs/vanilla_decode_order_quant_32/metadata.json` 与本目录 `summary.json`。

在正向指令的 CFG 分支读取每个 block 输出的文本隐状态与参考图像隐状态，包括 14 个双流 block 和 28 个单流 block。每张图的文本特征仅选 tokenizer 标记为**非 padding** 的指令 token（最多 32 个，在有效位置上等距取样）；source image 特征在固定的 1024 个参考 token 中等距取 32 个。任意 block/step 比较均对齐同一图像的相同 token 位置；逐 token 在特征维度 L2 归一化，计算余弦相似度，再依次对 token、其余 block/step、图像求均值。矩阵对角线为与自身比较，不应当作跨层/跨步相似度。真实指令 token 数为 5–24（中位数 9）。

## 结果

下表每张图先形成一个观测值，再对图像等权平均；区间为按图像重采样的 5,000 次 bootstrap 95% 区间。*任意两层/两步* 指排除对角线后的所有配对；邻近与远距离的特征不可混为一谈。

| 对比 | Text | Source image |
|---|---:|---:|
| 相邻 block | **0.97496** [0.97419, 0.97574] | **0.98015** [0.97949, 0.98076] |
| 任意两个不同 block | 0.57364 [0.56910, 0.57806] | 0.81355 [0.80998, 0.81713] |
| 相邻 step | **0.99748** [0.99694, 0.99782] | **0.99929** [0.99925, 0.99933] |
| 任意两个不同 step | 0.98496 [0.98381, 0.98607] | 0.99505 [0.99465, 0.99545] |

相邻 block 具有明显的特征冗余，source image 跨采样步尤其稳定。文本在早期双流 block 有显著变化，其中 block 13→14 的平均余弦约 **0.709**；跨越整个网络后 text block 1→42 约 **0.052**。因此论文中宜表述为 **“相邻层与相邻采样步的条件特征高度相似”**，不宜声称所有层的文本特征都近似不变。采样步 1→36 的文本/参考图像相似度分别约 0.923/0.974。余弦相似度描述特征几何关系，单靠本实验不能推断缓存或跳过计算后的输出质量。

**Suggested English caption:** *Cosine similarity of conditional features in vanilla EditMGT. The upper row compares aligned instruction-token and source-image-token features across Transformer blocks; the lower row compares the same features across sampling steps. Features are taken from the positive instruction branch after each block. Colors show the mean tokenwise cosine similarity across images; dashed lines separate two-stream and single-stream blocks. Adjacent features are highly similar, whereas instruction features exhibit larger changes between distant blocks.*

重新绘图：

```bash
MPLCONFIGDIR=/tmp/mpl-vanilla-feature .venv/bin/python scripts/plot_vanilla_condition_feature_similarity.py --stage render
```

重新采集需上述脚本运行 `--stage all` 并使用 GPU；脚本会逐张保存结果，已完成且协议一致的样本会被复用。
