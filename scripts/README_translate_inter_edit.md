# Inter-Edit 中文 prompt 翻译

默认使用 **Qwen/Qwen3.5-9B** + vLLM 离线批量翻译训练集 `instruction` 中包含汉字的指令（包括中英混合指令）。只读取 `train/metadata/train-*.jsonl.gz`，无需解压图像 tar，不修改原始数据。输出为独立的中英对照 JSONL，现有训练程序不会自动读取这些翻译。

## 环境与启动

建议单独创建推理环境，避免 vLLM 的 torch/transformers 依赖与训练环境中的固定版本冲突：

```bash
python3 -m venv /path/to/vllm-env
/path/to/vllm-env/bin/python -m pip install --upgrade -r requirements-translate.txt

PYTHON_BIN=/path/to/vllm-env/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/translate_inter_edit.sh \
  --model Qwen/Qwen3.5-9B \
  --data-dir /mnt/geminihzceph/public/data/Inter-Edit \
  --output-dir data/inter_edit_zh_to_en \
  --num-gpus 8 --tensor-parallel-size 1 --batch-size 128
```

`--model` 也支持本地模型目录。默认 8 个独立 vLLM 实例，每个实例使用一张 GPU，通过队列领取不同元数据分片。单卡需能放下模型及 KV cache。GPU 编号尊重已有 `CUDA_VISIBLE_DEVICES`，也可通过 `--gpu-ids` 显式指定实际 GPU 编号/UUID。不要再用 torchrun 或 accelerate 启动脚本，否则会重复创建实例。分片数量不足时只启动需要的实例。

如单卡显存不足，可改为 4 个双卡实例，或 1 个八卡实例；总卡数须能被 TP 大小整除，模型的 attention heads 等结构须支持对应 TP 大小。例如将 Qwen3.5-9B 放到一个八卡实例中：

```bash
PYTHON_BIN=/path/to/vllm-env/bin/python \
bash scripts/translate_inter_edit.sh \
  --model /path/to/Qwen3.5-9B \
  --num-gpus 8 --tensor-parallel-size 8 \
  --output-dir data/inter_edit_zh_to_en_qwen35_9b_tp8
```

模型聊天模板使用 `enable_thinking=False` 关闭 Qwen3.5 的思考输出。vLLM 固定使用 `language_model_only=True`，跳过视觉编码器和多模态预热，翻译只加载文本推理部分。推理环境要求 `vllm>=0.18.0`；不要沿用训练环境中的旧版 transformers。默认首轮使用贪心解码，异常结果最多额外重试 2 次，重试时温度为 0.2。

## 检查和小规模试跑

无需安装 vLLM 或使用 GPU 即可统计中文数量与已有进度；`--dry-run` 不写文件：

```bash
python3 scripts/translate_inter_edit.py --dry-run

# 先在独立目录翻译前 32 条中文指令，只需要一个实例。
PYTHON_BIN=/path/to/vllm-env/bin/python \
bash scripts/translate_inter_edit.sh \
  --num-gpus 1 --limit 32 --output-dir data/inter_edit_translation_smoke
```

默认只处理训练集。所有包含汉字的指令都会入选，英文指令不写入翻译文件。系统提示要求保留对象、属性、数字、空间关系与否定约束，并保留指令要求在图中显示的字面文字（如中文招牌），所以英文译文中允许出现中文引号内容。自动检查只检测空结果、缺少英文字母、未翻译、截断和意外思考/代码标记，翻译语义仍需抽样检查。

## 输出与断点续跑

```text
data/inter_edit_zh_to_en/
├── run_config.json              # 模型、源数据身份、翻译参数
├── shards/train-*.jsonl         # 每批 flush + fsync 的成功结果；续跑依据
├── errors/*.errors.json         # 分片最后一次执行仍失败的样本及模型输出
├── translations.jsonl          # 全部入选样本成功后，按源分片/行顺序原子合并
└── summary.json                # 完成数、剩余数、是否全部完成
```

每行格式：

```json
{"sample_id":0,"metadata_file":"train-00000-of-00275.jsonl.gz","row_index":0,"instruction_zh":"添加一双发光的筷子","instruction_en":"Add a pair of glowing chopsticks."}
```

`row_index` 是源 JSONL 中从 0 开始的行号。保留原文和 `sample_id`，可用于抽样核对和后续与训练元数据合并。脚本不复制或改写图片路径。

中断后**重新运行同一条命令**即可恢复，无需额外 `--resume`。已落盘的成功样本不会重新推理；进程在写入时中断造成的最后一行残缺会在续跑时截掉。完整行的损坏或原文不匹配会报错，避免悄悄使用错误结果。每次只允许一个翻译任务写同一个目录。Ctrl+C 和调度器发送的 SIGTERM 会清理各实例及其 vLLM 子进程。

失败结果会记录在 `errors/`，不算完成；其他样本继续翻译。只要有遗漏，就不会发布最终 `translations.jsonl`，程序返回非零状态。重启会重试遗漏样本。OOM 或进程退出也保留先前已保存的批次，最多重算正在执行、尚未保存的一批。可减小 `--batch-size`、`--max-num-seqs`，或调整显存预算。

续跑允许改变 GPU 数量、TP 大小、批大小、重试次数及显存预算。源数据路径/分片大小和修改时间、模型、revision、dtype、量化方式、token 长度、seed、翻译提示词或 `--limit` 变化时拒绝混用结果，须另选输出目录。小规模试跑和全量翻译使用不同目录。

如果旧输出目录中已经保存了其他模型（例如 Qwen2.5）的翻译，请给 Qwen3.5-9B 指定新目录，如 `--output-dir data/inter_edit_zh_to_en_qwen35_9b`。旧结果和断点会保留。

如截断持续失败，可在新目录中增大 `--max-tokens` 和 `--max-model-len`。`--revision` 可固定模型提交版本。`--enforce-eager` 可关闭 CUDA graph，`--quantization` 可加载适配的量化权重。更多参数见 `--help`。

接口依据：[vLLM 离线推理文档](https://docs.vllm.ai/en/latest/serving/offline_inference/)、[Qwen3.5 官方模型集合](https://huggingface.co/collections/Qwen/qwen35)、[Qwen3.5-9B 模型说明](https://huggingface.co/Qwen/Qwen3.5-9B)、[vLLM Qwen3.5 推理指南](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)。

CPU 回归验证（模拟 vLLM，不下载模型）：

```bash
python3 -m unittest discover -s tests -p 'test_translate_inter_edit.py' -v
```
