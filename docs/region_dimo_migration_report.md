# Region-DiMO → EditMGT 迁移实施报告

实施日期：2026-10-28。依据用户确认的 `Region_DiMO_Codex_Modification_Plan_20261028.md` 执行。所有修改位于独立工作目录的 `feature/region-dimo` 分支。没有 Push、合并 PR 或启动正式长训。

## 1. 审查基线与阶段

- EditMGT：执行 `git fetch upstream main` 后，从最新 `a8c0a2961173448b14ad24e5f1868ca4ca0543eb` 建分支；不是从旧实验主干覆盖文件。
- 本地来源：用户代码包 `DiMO_current_code_20261028_153609.zip` 的 `project/src/dimo/`；代码包 SHA256 为 `86fb6c91902de0f2c1f6eecd136a545d98cf85ffca40462e838e2db5f44ad681`。同时审查了实际训练工程，33 个原工程 Python 源文件仍与打包快照一致。
- 官方 DiMO：审查 `24613741ec9ca730273a6ebc822327878d5a086b` 的 `train_DIMO_Meissonic.py`、`configs/DIMO_Meissonic.yaml`、`Meissonic_src/transformer.py`，核对冻结、扰动位置/公式、scheduler、交替更新、FKL、EMA。第三方源码未整份复制入 PR。
- HF SparseEdit：读取官方当前文件树，实际发布结构为 `checkpoint-8000/transformer`，也有 SFT optimizer/scheduler 等文件；新 loader 只读 Transformer，不加载这些 SFT 训练状态。此结构是审查发现，不是 loader 的固定默认。

Stage 1 已完成：移植 DiMO 模块，使用新 main 的真实 Tiny Transformer，先跑原配方的一次更新、checkpoint 重载与下一步一致性验证，再修改算法。Stage 2 已完成：加入算法改动与不变量测试。Stage 3 已完成最小本机验收：CPU、CUDA、DDP、真实权重单步、推理及两样本双卡 Benchmark。Stage 4 提供本地 feature 提交、diff/stat、PR 文件清单及本文中的 PR 文案；不上传远端。

## 2. M01–M10 实施状态

| 项目 | 状态 | 实现及证据 |
|---|---|---|
| M01 Student Embedding 冻结 | 完成 | 新 preset 默认 true；建 AdamW 前冻结 `student.embed`；只传入 requires_grad 参数；冻结权重不更新、state dict keys 不变。 |
| M02 MLM Head / Auxiliary 冻结 | 完成 | `student.mlm_layer`、`auxiliary.embed`、`auxiliary.mlm_layer` 默认冻结；保留 head 向中间层传梯度；Tiny backward、gradient checkpointing 和 DDP 已验证。 |
| M03 Embedding Perturbation | 完成 | `sqrt(1-sigma²)*e + sigma*randn_like(e)`，Target embed 后/down_block 前；Student 两阶段生成 sigma=.3，Teacher/Aux 查询 0；默认 0 与原 main 输出/RNG 逐位一致。 |
| M04 100-Step Warmup | 完成 | 两个独立 constant_with_warmup scheduler；每个各随自己的 optimizer.step 计数，未交由 Accelerator 按 world size 扩增；保存/恢复和延长 max_steps 保持轨迹。 |
| M05 Teacher CFG=6 | 完成 | 去除 core 的固定 4 和 trainer CFG4 强制校验，显式传到 Teacher logits；cond/uncond 的输入 Token、原始 Mask、参考分支和时间相同，仅两路文本条件不同；Student/Aux CFG 保持 1。 |
| M06 AdamW | 完成/保持 | LR 1e-6、WD=.01、betas=(.9,.999)、eps=1e-8；没有换 Adam 或低精度 optimizer。 |
| M07 FKL | 完成/保持 | detached `p_A-p_T`、每样本 Q token mean；测试验证精确 logit gradient 及 Q 外零梯度；原函数 AST 与代码包一致。 |
| M08 r_init | 完成/保持 | 默认 1，支持 .5；G 内掩码/随机 VQ 初始化，G 外 Source Token hard lock。 |
| M09 EMA | 完成/保持 | .9999，`min(decay,(1+step)/(10+step))`，每次 Student 更新后、所有 Auxiliary 轮完成后更新一次；原函数 AST 未改变。 |
| M10 Halo=1 | 完成 | 源 SFT profile 保留 0/false，运行 profile 1/true；所有 forward 传 M，G 在真实 stem grid 扩张一次，Q⊆G，两种损失只计算 Q。 |

两阶段顺序仍是：Batch A Student FKL backward/step → 新 Batch B 使用**更新后的 Student**重新采样 → Auxiliary Hard CE (.1 smoothing)/step → EMA；fake_rounds 默认 1。时间输入保持浮点，mask_ratio/legacy 模式适配已测试，lora_part_enable 与所选 SFT 路径一致。

注意：sigma=.3 在整个 Target Embedding 上施加，不是仅在 M 内注入；卷积 stem 可能把 G 外 Target 扰动传播到邻近特征。Source Embedding 分支不加扰动，最终 G 外生成 Token 仍锁定 Source。

## 3. 工程项目状态

| 项目 | 状态 | 说明 |
|---|---|---|
| P01 上游功能分支 | 完成 | 新 main、新工作目录、新 feature 分支；没有操作师兄 main 的文件内容。 |
| P02 主干接口 | 完成 | 主干只增加 8 行可选 perturb 接口；pipeline/sparse/mask/v2/SFT/旧 benchmark 保持原字节内容；Tiny 原 forward 等价性已验证。 |
| P03 配置 | 完成 | v3 正式/公开 example、B0 example；旧缺省值与新 preset 分离；resolved config、checkpoint、指标、export 记录实际参数及源码 SHA/hash。 |
| P04 Teacher 来源 | 完成 | 指定 SFT run 内 latest 完整 step、显式 Transformer/Checkpoint 目录、固定 HF SHA 的本地 snapshot；严格 keys/shapes/profile/codebook/mask 检查、输出 pin；没有回退其他源。HF 路径有本地 Tiny 结构测试，未重新下载大型模型。 |
| P05 GPU/Batch/Precision | 完成；四卡真实权重运行待服务器验证 | 尊重 CUDA_VISIBLE_DEVICES，只有 GPU_IDS 显式设置才覆盖；进程数取可见卡并校验；记录型号、显存、world size、batch、角色参数及阶段峰值；单卡实权重与双卡 Tiny NCCL BF16 已通过。 |
| P06 数据 | 完成 | 固定 Manifest SHA/计数/路径/样本有效性，或显式本地训练流；缺失 manifest 立即失败；未宣称 1:1:1；没有重叠扫描。 |
| P07 独立完整兼容测试专项 | **按计划取消** | 仅运行本次新增逻辑必须的导入、真实 Tiny 前后向、Mask、checkpoint、推理及接口边界回归；没有开展独立全量兼容专项。 |
| P08 Checkpoint | 完成 | 新 v3 格式、每 2000 Student update、临时写入/哈希验证/原子发布、最近 5 份+里程碑；只清理当前 output/config 所属 v3，不碰 B0；严格 resume。 |
| P09 Benchmark | 完成最小验收；全量指标待独立评测 | 复用当前 main 的选择/数据顺序/Prompt/Mask/多轮依赖/GPU 分片/指标；新增 raw/EMA categorical runner；实际双卡两样本生成+评分通过；未跑全量新对照。 |
| P10 重叠审计专项 | **按计划取消** | 无训练/测试交叉扫描，没有声称排除了泄漏。 |
| P11 公开单步推理 | 完成 | infer + export；真实 EMA checkpoint 的 1024 BF16 单步 CLI 成功，记录 CFG1/Halo1/sigma0/direct decode；Tiny Seed/RNG 可复现性及 export/reload 通过。 |
| P12 遥测/数值 | 完成 | 复用既有 logits；默认 10/50/100/200/500 周期；固定 probe、预览恢复 RNG；不增加诊断网络 forward，集体归约全部 rank 参与；CUDA loss finite 检查异步，不按 microbatch .item。 |
| P13 PR 清理 | 完成 | 范围限定 DiMO 模块、一个 Transformer 可选补丁、公开 presets/launcher/tests/docs；不含服务器路径、权重、素材、结果、日志、官方源码副本或缓存。 |

`src/pipeline.py`、`src/sparse_utils.py`、`src/v2_utils.py`、`src/mask_utils.py`、`train/train.py`、`scripts/train.sh`、`src/benchmark_utils.py`、`src/benchmark_parallel.py`、两个原 benchmark 入口均已逐字节与 upstream/main 比较。旧 6000 checkpoint 的 COMPLETE 也与用户代码包中的标记一致；权重没有被加载为新实验 resume 或被写入覆盖。

## 4. 旧 B0 与新 v3 的有效配置差异

| 配置 | 旧 B0 兼容 preset | 新 v3 preset |
|---|---|---|
| Student / Auxiliary embed、MLM head | 可训练 | 全部冻结 |
| Student embedding perturb | 0 | .3，普通推理仍 0 |
| LR Scheduler | constant | constant_with_warmup |
| 每角色 LR warmup | 0 | 100 次本角色 optimizer update |
| Teacher 训练 CFG | 4 | 6 |
| Student / Auxiliary 训练 CFG | 1 / 1 | 1 / 1 |
| AdamW LR / WD / betas | 1e-6 / .01 / .9,.999 | 保持 |
| Loss / reduction | FKL / per-sample Q mean | 保持 |
| r_init | 1，支持 .5 | 保持 |
| EMA / warmup | .9999 / true | 保持 |
| 蒸馏 Halo / supervise | 1 / true | 保持；源 SFT 仍按真实 profile 检查 |
| fake_rounds / Auxiliary smoothing | 1 / .1 | 保持 |
| 保存周期 / retention | 历史实现无本迁移 retention | 2000 Student update；最近 5 + 指定里程碑 |
| 新增推理协议 | 历史 worker 按自己的元数据解释 | 公开 categorical 一步 CFG1，sigma0，direct VQ decode |

**旧缺省与 B0 preset 不同**：没有 freeze/noise/warmup 字段的旧 JSON 仍取 false/0/constant；缺少 ema_warmup 时仍取 false。B0 preset 则明确记录当前约 6000 步 B0 已使用的 EMA warmup=true。新 preset 与 B0 preset 都必须填私有路径、从同一指定 SFT 新建 output；旧 v1/v2 训练状态不能跨配方恢复。

源 Teacher 实测：SFT checkpoint-8000，sparse_target=true，Halo=0/supervise=false，cache_conditioning=true，mask_ratio time；保存到新输出后运行 Halo=1/supervise=true。源 condition threshold=.3，保持在模型 forward 中；它不同于 Teacher Benchmark 常用显式 cache 阈值 .8。Student 禁用推理 KV 重用，不会把训练得到的 cache-conditioned attention profile 整体关闭。所有真实配置均另存，不能说换 Halo 后与 SFT 训练完全等价。

## 5. 真实测试命令与结果

以下命令在 feature 工作目录执行。`PYTHON` 为实际准备好的 Python 3.10.21 环境，torch=2.5.1+cu121、accelerate=1.0.1、transformers=4.47.1、diffusers=.32.1。`REVIEW_DIR`、`ONE_STEP_CONFIG`、`ONE_STEP_OUTPUT`、`BASE_MODEL_DIR`、`INTEREDIT_TEST_ROOT` 是本机私有路径变量，不提交到公开配置。机器实际 GPU 为 A800 80GB；这仅是测试记录，不是程序默认要求。

```bash
"$PYTHON" -m compileall -q src/dimo src/transformer.py tests/test_dimo*
bash -n scripts/dimo/run_formal.sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" -m pytest -q \
  tests/test_dimo* tests/test_sparse_target.py tests/test_legacy_time_conditioning.py \
  tests/test_benchmark_parallel.py tests/test_magicbrush_masks.py
```

**通过：119 passed**。内容包括冻结不更新且梯度穿过 head、sigma=0 与实际 upstream forward/RNG 逐位一致、sigma=.3 精确公式/分支、CFG6 两路相同 Token/Mask/时间、Warmup 到 100 的计数、两角色 scheduler 精确恢复/预算扩展、Mask/FKL/初始化、原交替更新、旧状态/损坏 checkpoint 拒绝、retention 不删除 B0、manifest seek、跨 rank 遥测、路径/launcher/接口、真实 Tiny VQ 推理与 export。针对已有接口的回归不是 P07 独立专项。

```bash
# Stage 1：算法修改前的上游对接 smoke 已通过。
# Stage 3：下列是 v3 smoke，各目录全新。
OMP_NUM_THREADS=1 "$PYTHON" -m src.dimo.smoke --steps 1 --output "$REVIEW_DIR/stage3_cpu_v3"
OMP_NUM_THREADS=1 "$PYTHON" -m src.dimo.smoke --workers 2 --steps 1 --output "$REVIEW_DIR/stage3_ddp_gloo"
CUDA_VISIBLE_DEVICES="$ONE_FREE_GPU_UUID" OMP_NUM_THREADS=1 "$PYTHON" -m src.dimo.smoke \
  --device cuda --mixed-precision bf16 --steps 1 --output "$REVIEW_DIR/stage3_cuda_tiny"
CUDA_VISIBLE_DEVICES="$TWO_FREE_GPU_UUIDS" OMP_NUM_THREADS=1 "$PYTHON" -m src.dimo.smoke \
  --device cuda --workers 2 --mixed-precision bf16 --steps 1 --output "$REVIEW_DIR/final_ddp_nccl"
```

**全部通过**：CPU FP32；双进程 Gloo FP32；单 GPU BF16；双 GPU NCCL BF16。均覆盖冻结、gradient checkpointing、两阶段 backward、两套 optimizer/scheduler、EMA、完整 checkpoint、RNG/数据位置恢复，并与不中断的**下一步**逐项一致。Tiny smoke 中显式使用诊断 LR=1e-4，正式 preset LR 仍是 1e-6；首步 warmup 的 LR=0，下一步已验证非零更新。

```bash
# 私有配置显式选择与 B0 相同的 SFT、相同固定 50K Manifest、1024/BF16。
# 仅为最小预检设置 max_steps=preflight_steps=1、micro_batch=accumulation=1，
# 不使用正式 global batch 32/卡数布局，也不用于质量或吞吐结论。
CUDA_VISIBLE_DEVICES="$ONE_FREE_GPU_UUID" DIMO_NUM_PROCESSES=1 OMP_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 "$PYTHON" -u -m src.dimo.train \
  --config "$ONE_STEP_CONFIG" --stop-after-preflight
```

**真实权重单步通过，之后进程退出**：Student FKL=0.0007897217874；Aux CE=8.3487510681；EMA decay=.1；checkpoint 保存/完整哈希/重载、RNG、数据回放、梯度隔离通过。预检峰值 46,539,544,064 bytes（约 43.34 GiB）；Student/Aux 阶段记录分别为 34,686,725,120 / 42,631,048,192 bytes。不能据此声称正式多卡批量或长期稳定性已通过。

实际每角色总参数 1,010,879,488；Student 与 Auxiliary 各可训练 991,937,536；Teacher/EMA 可训练 0。四个指定冻结模块在报告中逐角色列出。首个真实 optimizer update 使用 warmup LR=0；更新后日志记录**下一步** LR=1e-8。因此此 smoke checkpoint 只验证流程，不能作为已完成新配方训练或质量进步的证据。

```bash
CUDA_VISIBLE_DEVICES="$ONE_FREE_GPU_UUID" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  "$PYTHON" -m src.dimo.infer --checkpoint "$ONE_STEP_OUTPUT/checkpoint-1" \
  --model ema --base-model-dir "$BASE_MODEL_DIR" \
  --source "$SMOKE_SOURCE" --mask "$SMOKE_MASK" --prompt 'Replace the heart with a tulip.' \
  --output "$REVIEW_DIR/infer_gpu_ema_seed42.png" --halo 1 --seed 42
CUDA_VISIBLE_DEVICES="$TWO_FREE_GPU_UUIDS" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  "$PYTHON" -m src.dimo.benchmark --benchmark inter_edit --backend ema \
  --checkpoint "$ONE_STEP_OUTPUT/checkpoint-1" --model-path "$BASE_MODEL_DIR" \
  --data-dir "$INTEREDIT_TEST_ROOT" --language en --limit 2 \
  --output-dir "$REVIEW_DIR/benchmark_gpu2_smoke" --tar-cache-dir "$REVIEW_DIR/tar_index" \
  --guidance-scale 1 --num-inference-steps 1 --sparse-target-halo 1 \
  --num-gpus 2 --bf16 --warmup 0 --no-preserve-unmasked-pixels --metrics l1,l2 --seed 42
```

**真实公开推理 CLI 通过**：1024 BF16、EMA、CFG1、Halo1、r_init1、sigma0、直接 VQ decode，已产生 PNG 和协议 sidecar。**双卡 Benchmark 通过**：上游英文 test 顺序前 2 个样本，生成 2/2、评分 2/2、失败 0、complete=true；run.json/分片执行/逐样本/汇总均成功。只选 L1/L2 验证代码路径，没有把 2 个样本称为正式评测结果。

故障与修正记录：早期迁移中 3 个遥测测试仍假定 token 周期 100，现已显式设置历史周期，并新增默认 200 周期测试，最终全部通过。首个实权重单步尝试在 AdamW 状态初始化时 OOM：所选 GPU 上另有约 50GB 的进程；未杀其他进程、未降精度或改分辨率，保留失败日志和目录后，显式选择空闲 GPU UUID、全新输出目录重试成功。源码指纹记录各测试当时的实际文件；单步真实权重通过后，仅补充了无训练数学变化的日志/协议元数据与边界检查，再通过最终单测及双卡端到端 smoke。

**未执行/待验证**：四卡真实权重 DDP、正式 micro_batch/accumulation 预算的 20-step 预检、10000-step 长训、全量 Teacher/Raw/EMA 指标和速度对照、独立完整 P07、独立 P10。没有 GPU 结果伪造，没有自动正式续训。

## 6. 逐文件修改报告 / PR 文件清单

下表为本次 PR 的完整范围，除 Transformer 为修改，其余为新增。无权重、数据、日志、旧主干文件副本或服务器绝对路径。

| 文件 | 修改及目的 |
|---|---|
| `src/transformer.py` | 仅 +8 行：默认 0 的可选 sigma 参数、范围校验及 Target stem 前 Gaussian perturb。 |
| `src/dimo/__init__.py` | DiMO 包声明，保持训练模块独立。 |
| `src/dimo/core.py` | 原区域 Mask/FKL/CE/EMA；显式 Teacher CFG、Student sigma 路由及 categorical 一步；数学核心保留。 |
| `src/dimo/train.py` | 移植并适配新 main；冻结→optimizer→独立 scheduler；Teacher pin、两阶段循环、rank 硬件、固定 probe、预检、严格 resume/保存/清理、OOM 诊断与周期日志。 |
| `src/dimo/config.py` | 旧缺省值解析、env 显式覆盖、来源唯一性、路径/精度/batch/loss/数据/算法校验、Git SHA 与实现哈希。 |
| `src/dimo/optimization.py` | 冻结模块、各角色参数报告、仅 trainable AdamW、按各自 optimizer update 计数的 warmup。 |
| `src/dimo/origin.py` | 扩展 run/direct/HF 来源；检查真实布局、profile、keys/shapes、SHA，拒绝不完整/不兼容来源和隐式替代。 |
| `src/dimo/checkpoint.py` | 完整三模型、双 optimizer/scheduler、rank RNG/数据进度；v3 隔离、原子发布/哈希、严格重载/轨迹检查和仅当前实验 retention。 |
| `src/dimo/data.py` | 保留独立数据 RNG 的 deterministic replay，不重置 CUDA 模型采样 RNG。 |
| `src/dimo/manifest_data.py` | 固定 Manifest SHA/组成、唯一 ID、根目录约束、source/target/mask/prompt 检查、消费时 mask/hash 校验、确定性 seek。 |
| `src/dimo/device.py` | 本地 rank/可见 GPU 绑定与 DDP 映射验证。 |
| `src/dimo/telemetry.py` | 复用已有 logits 的熵/gap/多样性、周期控制、所有 rank 归约，Teacher 指标去除 CFG4 固定标签。 |
| `src/dimo/preflight.py` | 不启动训练的 Teacher/base/manifest 严格检查，输出源/运行 Halo 区别。 |
| `src/dimo/smoke.py` | 真实 Tiny EditMGT 的 CPU/Gloo/CUDA/NCCL 训练、冻结、checkpoint/恢复 smoke；只由显式命令触发。 |
| `src/dimo/infer.py` | 公开 raw/EMA 角色加载、categorical 一步、Source hard lock、VQ decode、Seed/RNG 隔离、可选明确 RGB 合成、来源/协议记录。 |
| `src/dimo/export.py` | 推理 safetensors 导出，保存有效配置、来源、源码信息与文件哈希；拒绝覆盖。 |
| `src/dimo/benchmark.py` | 命令内作用域 adapter + spawn trampoline；复用上游顺序/分片/评分；run.json 记录 Teacher 与 Student 的不同真实协议。 |
| `configs/dimo/region_dimo_v3.json` | 独立正式 v3 preset；私人路径为空，必须显式填入。 |
| `configs/dimo/region_dimo_v3.example.json` | 同配方公开模板，不携带任何服务器目录或 GPU 假设。 |
| `configs/dimo/region_dimo_b0.example.json` | 显式历史 B0 配方，用于从同源 SFT 新初始化的对照。 |
| `scripts/dimo/run_formal.sh` | 独立 launcher，兼容主干 GPU 选择契约，支持显式 Python/私有 config/进程数与 stop-after-preflight。 |
| `tests/test_dimo_update.py` | 迁入必须的旧更新顺序、Fresh Batch、gradient isolation、EMA、随机初始化和 checkpoint 回归；移除整份官方源码 fixture 依赖。 |
| `tests/test_dimo_telemetry.py` | 精确概率统计、token 统计、周期、全 rank 归约与非破坏性监控。 |
| `tests/test_dimo_manifest.py` | 固定清单的样本顺序、seek/resume、数据位置与随机性检查。 |
| `tests/test_dimo_interfaces.py` | direct/latest/HF 来源、配置/缺失数据失败、真实 Tiny VQ 推理/导出、Benchmark adapter 和 GPU launcher 契约。 |
| `docs/region_dimo.md` | 方法、来源/许可、旧缺省、新配方、两个环境启动方式、Teacher/数据/GPU、preflight、resume、推理及评测说明。 |
| `docs/region_dimo_migration_report.md` | 本报告：逐项状态、真实验收、未测项、配置差异、逐文件清单及 PR 文案。 |

没有新增依赖：所用库已在师兄 requirements 中；没有修改其 PyTorch 声明或全项目依赖锁。

## 7. 两个环境的启动方式

完整示例见 `docs/region_dimo.md`；都是**预检停止**命令，不是本次自动执行的长训命令。

```bash
# 环境 A：repo .venv + 调度器已设置 CUDA_VISIBLE_DEVICES
export DIMO_CONFIG=./private/dimo_v3.json
.venv/bin/python -m src.dimo.preflight --config "$DIMO_CONFIG"
bash scripts/dimo/run_formal.sh --stop-after-preflight

# 环境 B：外部 Python + 显式可见 GPU；私有 JSON 中填独立 Teacher/base/data/output
export DIMO_PYTHON=/path/to/env/bin/python
export DIMO_CONFIG=/path/to/private/dimo_v3.json
export GPU_IDS=0,1
export NUM_PROCESSES=2
"$DIMO_PYTHON" -m src.dimo.preflight --config "$DIMO_CONFIG"
bash scripts/dimo/run_formal.sh --stop-after-preflight
```

只有用户以后决定正式训练时才去掉 stop 标志；续训必须是**同一个新 v3 实验**，相同源码/配置/数据/world size，并显式 `--resume latest`。旧 B0 路径不应用在该命令中。师兄仓库写权限未测试，不需要为本地交付申请；若以后无写权限，用自己的 Fork + feature 分支准备 Draft PR，仍不直接推 main。

## 8. 建议 PR 文案（未向远端创建 PR）

**Title**: Add isolated Region-DiMO v3 distillation and categorical one-step inference

**Body**:

Region-DiMO previously lived in a server-specific snapshot and could not be integrated by copying its old Transformer/Pipeline over EditMGT. This change adds an isolated DiMO package on current main, with a default-off target-embedding perturbation interface as the sole backbone change. The SFT trainer, Pipeline and existing benchmark entries are unchanged.

The v3 preset freezes Student/Auxiliary embeddings and MLM heads, uses sigma .3, per-optimizer 100-step LR warmup, Teacher CFG6 and distillation Halo1. It preserves AdamW, regional FKL, fresh-batch Student/Auxiliary alternating updates and EMA. Public configs require independent Teacher/base/data/output paths; strict checkpoint provenance, atomic v3 checkpoints, raw/EMA categorical CFG1 inference and an upstream benchmark adapter are included.

Validation: 119 targeted tests; CPU/Gloo and CUDA/NCCL BF16 Tiny training with exact continuation; one real-SFT 1024 BF16 Student/Aux update with full checkpoint/RNG/data roundtrip; real EMA inference; two-sample, two-GPU Inter-Edit generation/scoring. The first warmup update uses LR0, so this is correctness evidence rather than a new quality result. Four-GPU full-model preflight, long training and full benchmarks remain unrun. P07/P10 dedicated projects were intentionally cancelled. No weights, datasets, personal server paths, generated results or official DiMO source dumps are included.
