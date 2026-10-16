<h1 align="center" style="line-height: 50px;">
  ✨ EditMGT: Unleashing the Potential of Masked Generative Transformer in Image Editing ✨
</h1>

<div align="center">
Wei Chow<sup>1,*</sup>, Linfeng Li<sup>1,*</sup>, Lingdong Kong<sup>1</sup>, Zefeng Li<sup>1</sup>, Qi Xu<sup>1</sup>, Hang Song<sup>1</sup>, Tian Ye<sup>4</sup>, Xian Wang<sup>1</sup>, Jinbin Bai<sup>3</sup>, Shilin Xu<sup>1</sup>, Xiangtai Li<sup>1</sup>, Junting Pan<sup>1</sup>, Shaoteng Liu<sup>1</sup>, Ran Zhou<sup>1</sup>, Tianshu Yang<sup>1</sup>, Songhua Liu<sup>2</sup>

<sup>1</sup>ByteDance, <sup>2</sup>Shanghai Jiao Tong University, <sup>3</sup>National University of Singapore, <sup>4</sup>The Hong Kong University of Science and Technology (Guangzhou)
  
<small>*Equal Contribution</small>

[![arXiv](https://img.shields.io/badge/arXiv-2512.11715-b31b1b.svg)](https://arxiv.org/abs/2512.11715)
[![Dataset](https://img.shields.io/badge/🤗%20CrispEdit2M-Dataset-yellow)](https://huggingface.co/datasets/WeiChow/CrispEdit-2M)
[![Checkpoint](https://img.shields.io/badge/🧨%20EditMGT-CKPT-blue)](https://huggingface.co/WeiChow/EditMGT)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/weichow23/editmgt/tree/main)
[![Page](https://img.shields.io/badge/🏠%20Home-Page-b3.svg)](https://weichow23.github.io/EditMGT/)
</div>

## 🚀 Project Introduction

EditMGT is a novel framework that leverages Masked Generative Transformers for advanced image editing tasks. Our approach enables precise and controllable image modifications while preserving original content integrity.

## ⚡ Quick Start  

First, clone the repository and navigate to the project root:  
```shell
git clone https://github.com/weichow23/editmgt
cd editmgt
```

## 🔧 Environment Setup

```bash
# Create and activate conda environment
conda create --name editmgt python=3.9.2
conda activate editmgt

# Optional: Install system dependencies
sudo apt-get install libgl1-mesa-glx libglib2.0-0 -y

# Install Python dependencies
pip3 install git+https://github.com/openai/CLIP
pip3 install -r requirements.txt
```

The Hugging Face data tools use `datasets==2.21.0` and `pyarrow==20.0.0`.
Older environments combining `datasets==2.14.4` with PyArrow 21 or newer fail
on import with `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`;
the class was removed in [Arrow 21](https://arrow.apache.org/blog/2025/07/17/21.0.0-release/).
To update an existing repository virtual environment, use:

```bash
# If the virtual environment has no pip (e.g. an unseeded uv venv):
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install "datasets==2.21.0" "pyarrow==20.0.0"
```

If `uv` is available, it can install directly into the environment without pip:

```bash
uv pip install --python .venv/bin/python "datasets==2.21.0" "pyarrow==20.0.0"
```

Inter-Edit training reads the local tar/JSON files directly and imports Hugging
Face `datasets` only when a Hugging Face dataset reader is selected. Updating
`requirements.txt` alone does not change packages already installed in a venv.

⚠️ **Note**: If you encounter any strange environment library errors, please refer to [Issues](https://github.com/viiika/Meissonic/issues/14) to find the correct version that might fix the error.

## 🎯 Training

The training script defaults to the local Inter-Edit dataset with full-reference,
sparse-target editing. See the mask-region-aware training section below for the
dataset layout and entry points.

### 📝 Configuration
The `train/train_edit.yaml` file controls training behavior. Key settings include:

| Parameter | Description | Default |
|-----------------|-----------------------------------------------------------------------------|----------|
| `mixed_precision` | Training Precision | `no` for float32; `bf16` for bfloat16 |
| `resume_from_checkpoint` | Resume Training | `direct local path` or `latest` |
| `wandb_id` | Weights & Biases ID | keep empty or id number |
| `pretrained_model_name_or_path` | Base Model | `MeissonFlow/Meissonic` |
| `output_dir` | Checkpoint Save Location | `./runs/editmgt` |
| `train_batch_size` | Batch Size per GPU | `4` |
| `gradient_accumulation_steps` | Steps before Weight Update | `8` |
| `learning_rate` | Initial Learning Rate | `1e-4` |
| `max_grad_norm` | Gradient Clipping Value | `10` |
| `text_encoder_architecture` | Text Encoder Model | `CLIP_Gemma2` |
| `resolution` | Image Resolution | `1024` |
| `lr_scheduler` | Learning Rate Schedule | `constant` |
| `max_train_steps` | Total Training Steps | `500000` |
| `checkpointing_steps` | Save Frequency | `200` |
| `logging_steps` | Log Metrics Frequency | `10` |

Weights & Biases logging uses the account's default entity when `WANDB_ENTITY`
is unset. Set `WANDB_ENTITY` only to select an existing team/user, and optionally
`WANDB_PROJECT` to override the project name `editmgt`. Online logging requires
an authenticated W&B account (for example, `.venv/bin/wandb login` or
`WANDB_API_KEY`). `--wandb_id` resumes an existing run when that ID exists.
Accelerate owns the single W&B run for both training metrics and preview images.

To train without external trackers, while keeping local Inter-Edit test previews:

```bash
SKIP_INSTALL=1 REPORT_TO=none bash train/train.sh
```

To keep W&B records locally without online authentication:

```bash
SKIP_INSTALL=1 WANDB_MODE=offline bash train/train.sh
```

#### 🧩 Enabling LoRA
To use LoRA (reduces memory usage by ~70%), add the following to the `train.sh` file under `train`:
```shell
--use_lora \
--lora_r 32 \
--lora_alpha 128
```
Here, `lora_r` and `lora_alpha` represent the rank and alpha parameters of LoRA, respectively.

#### 🚗 Run Training
Use the script to start training. Specify the config file and enable LoRA if needed:
```shell
bash train/train.sh
```

The launcher preserves `CUDA_VISIBLE_DEVICES` and defaults to one worker per
visible GPU. It no longer forces GPUs `0..7`. Set `GPU_IDS` to explicitly select
devices in the container's CUDA namespace; `NUM_PROCESSES` may not exceed the
visible device count. A single worker runs without the distributed launcher.
For example, after preparing the environment, select two GPUs with:

```bash
SKIP_INSTALL=1 GPU_IDS=0,1 NUM_PROCESSES=2 bash train/train.sh
```

Omit `GPU_IDS` to retain the scheduler's device assignment. `SKIP_INSTALL=1`
skips environment/package setup on subsequent launches; `MAIN_PROCESS_PORT`
overrides the default distributed port `25012`.

Training defaults to eager mode. `torch.compile` is opt-in: the observed PyTorch
2.11 Inductor backward failure (`AssertionError: 4096 v.s. 1024`) occurs during
reduction fusion and is avoided by leaving compilation disabled. The training
entry point also disables Accelerate's automatic Dynamo wrapping, so a launcher
configuration cannot silently compile the eager path.

```bash
SKIP_INSTALL=1 bash train/train.sh
```

For compiler investigation, `COMPILE_MODEL=1 COMPILE_BACKEND=aot_eager` enables
AOTAutograd without Inductor; `COMPILE_BACKEND=inductor` opts back into Inductor
and may encounter the reported failure. These correspond to the training flags
`--compile_model --compile_backend ...`. Adapters, checkpointing and EMA are
configured before optional compilation. Saving/loading always operates on the
original model, without requiring `_orig_mod.` parameter prefixes, so the same
checkpoint format is used by both modes.

The default optimizer remains 8-bit AdamW. The CUDA 13 PyTorch build requires
`bitsandbytes==0.48.2` from `requirements.txt`; the old `0.45.5` wheel does not
include `libbitsandbytes_cuda130.so`. Its CPU fallback leaves the legacy CUDA
optimizer function table undefined, causing
`NameError: str2optimizer8bit_blockwise` at the first update. CUDA 13 support
was added in [bitsandbytes 0.48.0](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.48.0).
Update an existing environment explicitly when using `SKIP_INSTALL=1`:

```bash
.venv/bin/python -m pip install "bitsandbytes==0.48.2"
```

To use native PyTorch AdamW without installing/updating bitsandbytes:

```bash
SKIP_INSTALL=1 USE_8BIT_ADAM=0 bash train/train.sh
```

Native AdamW uses more optimizer-state memory; it uses `foreach=False` here to
avoid foreach temporary buffers. Learning rate, betas and weight decay stay as
configured. Quantized optimizer states are not interchangeable with native
AdamW states: resume with the same optimizer type, or start a new optimizer
from separately loaded model weights. Requested 8-bit optimization checks the
installed version and native CUDA library before the first training update.

An OOM at `Accelerator(...)` / `torch.cuda.set_device`, before model loading,
concerns CUDA context/distributed initialization rather than a training batch.
The training entry point prints each local rank's mapping, binds its device
before NCCL initialization, and rejects multiple local workers per visible GPU.
Inspect `nvidia-smi` memory/process information for existing GPU allocations;
restrict the job to assigned devices with enough free memory. Selecting fewer
devices also changes the effective global batch size with the current fixed
per-device batch and accumulation settings.

⚠️ **Important**: Since Meissonic has not released bf16 weights, fp16 weights may cause training instability. Therefore, we use fp32 for training (although this will be slower).

## 🔍 Inference
We provide a standard example script:

```shell
python3 infer.py
```

### 🃏 GEditBench-EN
We also provide evaluation scripts for GEditBench. Refer to the [Official Repo](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md) for implementation details.

You'll need to install additional dependencies beyond our `requirements.txt`:
```shell
pip3 install megfile==4.1.4
```
Then use the scirpt:

```shell
python3 eval/geditbench/infer.py
```

Generate and organize your images in the following directory structure:
```
results/
├── {method_name}/
│   └── fullset/
│       └── {edit_task}/
│           └── en/  # English instructions
│               ├── key1.png
│               ├── key2.png
│               └── ...
```

Run the inference script:
```shell
PYTHON_PATH='./' python3 eval/geditbench/infer.py
```

For GPT-4.1 evaluation, set up your API keys in `eval/geditbench/rate.py` at lines `L103` and `L105` for GPT4.1 access, then run:

```shell
PYTHON_PATH='./' python3 eval/geditbench/rate.py --model_name editmgt --save_dir eval/geditbench/score_dir --backbone gpt4.1 --edited_images_dir eval/geditbench/results/ --instruction_language en
```

Run the analysis script to get scores for semantics, quality, and overall performance:
```shell
PYTHON_PATH='./' python3 eval/geditbench/stat.py --model_name editmgt --save_path eval/geditbench/score_dir --backbone gpt4o --language en
```

This will output scores broken down by edit category and provide aggregate metrics.

**Note**: GEditBench scores can fluctuate significantly due to random generation results and GPT version differences. Fluctuations around our reported scores are normal. For AnyBench, EmuEdit, and MagicBrush, we recommend reducing guidance scale and steps for the first two, and using our mask strategy for the latter.

## Full benchmark inference and metrics

Two standalone entry points generate benchmark images and score their results:

- `benchmark_inter_edit.py`, wrapped by `scripts/benchmark_inter_edit.sh`
- `benchmark_magicbrush.py`, wrapped by `scripts/benchmark_magicbrush.sh`

Both accept our full Transformer checkpoints, final-model directories, and LoRA
checkpoints through `--checkpoint`. `--vanilla` instead uses the released
`checkpoint/EditMGT/editmgt` weights. Choose a separate output directory for
each checkpoint, shift, masking protocol or sampling configuration.

### Metric dependencies

The scripts use the existing environment and do not install packages. Additional
metric dependencies are listed in `requirements-benchmark.txt`. For a uv-managed
environment without pip inside the venv:

```bash
uv pip install --python .venv/bin/python -r requirements-benchmark.txt
```

Inter-Edit's regional metrics additionally need
[AlphaCLIP](https://github.com/SunzeY/AlphaCLIP), its `loralib` dependency, and the
matching visual checkpoint, such as `clip_l14_grit1m_fultune_8xe.pth` for
`ViT-L/14`. Follow AlphaCLIP's installation/model-zoo instructions and pass
the downloaded file through `--alpha-clip-ckpt`. These weights are separate
from EditMGT's text encoder weights.

LPIPS, CLIP and DINO may download their official model weights on first use.
`--model-cache-dir` configures CLIP/AlphaCLIP downloads and the PyTorch Hub
directory. Offline DINO evaluation accepts `--dino-repo /path/to/dino` and
`--dino-weights /path/to/dino_deitsmall16_pretrain.pth`.

### Inter-Edit test benchmark

```bash
bash scripts/benchmark_inter_edit.sh \
  --checkpoint /path/to/checkpoint-N \
  --language en \
  --alpha-clip-ckpt /path/to/clip_l14_grit1m_fultune_8xe.pth \
  --inference-shift 1.5 \
  --output-dir outputs/benchmark_inter_edit_en
```

The default data root is `/mnt/geminihzceph/public/data/Inter-Edit`. Only the
official **test** metadata is read. Images are read directly from its tar
archives, using the existing member-offset cache; extracted official files are
also supported. The default `--language en` selects the complete 3,413-example
English subset. `--language all` selects all 6,250 examples. Original test IDs
and prompts are retained. `--limit N` or `--sample-indices ...` creates an
explicitly recorded partial subset rather than a full-benchmark result.

Default metrics follow the
[public objective evaluator](https://github.com/Delong-liu-bupt/Inter-Edit/blob/main/evaluation/objective_metrics.py):

| Saved metric | Definition | Direction |
| --- | --- | --- |
| `S_global_LPIPS` | AlexNet LPIPS between generated and GT images | Lower |
| `S_in_AlphaCLIP` | AlphaCLIP cosine similarity between generated/GT features under the original mask | Higher |
| `S_out_AlphaCLIP` | AlphaCLIP cosine similarity between generated/source features under the inverse original mask | Higher |
| `BSS_Laplacian` | Absolute difference of mean Sobel gradient magnitudes in the inner/outer boundary rings | Lower |

The boundary key preserves the public evaluator's name; its implementation uses
Sobel gradients and an 11x11 erosion/dilation kernel. Empty boundary rings return
zero, as in that evaluator. `S_ref_AlphaCLIP` is recorded as unavailable because
the public source/GT/mask release has no additional reference-object image.
Source is not substituted for such an image.

Scoring aligns all images to native source dimensions unless
`--metric-resolution` explicitly selects a square scoring resolution. The
**original annotation mask** defines all regional metrics; generation halo and
mask dilation never expand the scoring region. Metrics are averaged per image,
with additional breakdowns by edit type and language. Use `--metrics` to select
a declared subset, for example `--metrics lpips,boundary` without AlphaCLIP, or
`--metrics l1,l2` for pixel errors only. Partial metric sets are labeled in the
summary and do not replace the full default metric set.

### MagicBrush dev single-turn and multi-turn evaluation

MagicBrush evaluation defaults to the existing **held-out dev split** under
`data/magicbrush/data/dev-*.parquet`: 528 edit turns across 266 sessions. These
dev shards are excluded from the mixed training stream. Both independent
single-turn and continuous multi-turn evaluation are enabled by default:

```bash
bash scripts/benchmark_magicbrush.sh --checkpoint /path/to/checkpoint-N
```

The defaults are `--split dev --mode both`, metrics `l1,l2,clip-i,dino`, and
output directory `outputs/benchmark_magicbrush_dev/`. No caption file is needed
for these four metrics. Local dev does not contain target global captions, so
CLIP-T is opt-in when those captions are supplied. Reports retain the `dev`
split label and record the actual evaluated metric set.

The separately distributed official test archive remains available through
an explicit `--split test`. Obtain it from the
[authors' dataset access page](https://github.com/OSU-NLP-Group/MagicBrush#dataset-access)
and point the script at its extracted root. The official image-folder format is:

```text
magicbrush_test/
├── edit_sessions.json
├── global_description.json       # may instead be supplied via --caption-file
└── images/
    └── <image_id>/
        ├── <image_id>-input.png
        ├── <image_id>-output1.png
        ├── <image_id>-mask1.png
        └── ...
```

```bash
bash scripts/benchmark_magicbrush.sh \
  --checkpoint /path/to/checkpoint-N \
  --data-dir /path/to/magicbrush_test \
  --split test --mode both \
  --caption-file /path/to/global_description.json \
  --metrics l1,l2,clip-i,dino,clip-t \
  --inference-shift 1.5 \
  --output-dir outputs/benchmark_magicbrush_test
```

The reader supports official `{image_id: [turns]}` `edit_sessions.json`, an
explicit `--manifest` with flat JSON/JSONL turns, and local split-specific
`test-*.parquet` / `dev-*.parquet` releases. Flat JSON turns specify `img_id`,
`turn_index`, `input`, `output`, `mask` and `instruction`; image paths are
resolved under the selected root or its `images/<image_id>/` directory. Parquet
uses `img_id`, `turn_index`, `source_img`, `target_img`, `mask_img`, `instruction`.
Selection scans only small metadata columns, and image rows are streamed with
a bounded row cache. Turn indices are normalized when an entire session is
zero-based, and incomplete/duplicate sessions are rejected.

`--mode single` performs every edit independently using that turn's provided
source. `--mode multi` starts from the first source and feeds **generated output
from the preceding turn** into each subsequent turn. `--mode both` computes
both; their identical first turn is generated/scored once and shared.
Incomplete generated sessions never fall back to a GT intermediate image.
Text/reference KV caches are recreated per pipeline call as the source/prompt
changes between turns.

Results follow the authors'
[image evaluator](https://github.com/OSU-NLP-Group/MagicBrush/blob/main/evaluation/image_eval.py):

| Saved metric | Definition | Direction |
| --- | --- | --- |
| `l1` | Mean absolute RGB pixel error in [0,1] | Lower |
| `l2` | Mean squared RGB error, without square root | Lower |
| `clip-i` | OpenAI CLIP ViT-B/32 generated/GT image cosine similarity | Higher |
| `dino` | DINO v1 ViT-S/16 generated/GT image cosine similarity | Higher |
| `clip-t` | Generated image / target global-caption CLIP cosine similarity | Higher |
| `clip-t-oracle` | GT image / the same target caption, for reference | Higher |

The main groups in `metrics/summary.json` are `single_turn_all` (all independently
edited turns) and `multi_turn_final` (one last output per session). An additional
`multi_turn_all_diagnostic` group is labeled separately. Scores from independent
and iterative protocols are not pooled into a combined quality number. Each
image contributes equally within its group. Pixel metrics compare at native
GT size; feature metrics use the official CLIP/DINO transforms. An optional
`--metric-resolution` override is recorded and changes that evaluation setting.

CLIP-T reads **target global descriptions**, including the official nested
`captions[image_id][target_filename]` mapping. It does not use the edit prompt.
Enable CLIP-T explicitly with `--metrics l1,l2,clip-i,dino,clip-t` plus a
`--caption-file`; the default four metrics require no captions. Masks are required in the default mask-provided
inference mode; no mask is synthesized from the GT image. `--mask-mode none`
selects mask-free inference and disables sparse-target generation. The chosen
masking protocol is recorded for comparisons. Missing test data does not cause
a fallback to dev or train.

The default output directory follows the selected split: `benchmark_magicbrush_dev`
for dev, and `benchmark_magicbrush_test` for an explicit test run.

### Saved outputs, resume and offline scoring

```text
<output-dir>/
├── run.json                      # data scope, checkpoint identity and generation settings
├── selection.json                # original IDs, prompts, GT/source/mask locators and protocol groups
├── execution.json                # execution hardware and loaded Transformer policy
├── generation_summary.json       # generation completion counts and timing
├── images/                       # edited PNGs, restored to the inference input's native size
├── records/<key>.json             # per-image generation status, seed, timing and cache records
├── inputs/<key>/                 # optional source/GT/original mask with --save-inputs
└── metrics/
    ├── per_sample.jsonl          # per-image scores and prediction paths
    ├── summary.json              # means, standard deviations, counts and dataset breakdowns
    └── errors.jsonl              # scoring failures
```

Inter-Edit predictions use `images/00000.png` with the original official test
index. MagicBrush uses `images/<id>/<id>_1.png`, `<id>_inde_2.png`,
`<id>_iter_2.png`, etc., matching the official independent/iterative naming.
Original dataset assets are not rewritten or copied unless `--save-inputs` is
requested. Stored dataset locators remain absolute, so the source dataset must
remain accessible for later scoring.

`--resume` reuses completed images when checkpoint identity, generation settings,
selection and protocol match. Changed settings require a separate output
directory. Downstream iterative turns are regenerated when their generated
parent is regenerated. Each image has a deterministic seed based on the base
seed, original sample/session ID and turn, independent of subset size and
interruption points.

`--stage generate` saves images/records without loading metric dependencies.
`--stage metrics` scores a saved run without loading EditMGT or requiring its
checkpoint again. The latter uses the saved selection/generation settings,
and permits attaching a caption file later:

```bash
# Generate first, even before AlphaCLIP has been installed.
bash scripts/benchmark_inter_edit.sh \
  --checkpoint /path/to/checkpoint-N --language en \
  --stage generate --output-dir outputs/benchmark_inter_edit_en

# Score the existing images later.
bash scripts/benchmark_inter_edit.sh \
  --stage metrics --output-dir outputs/benchmark_inter_edit_en \
  --alpha-clip-ckpt /path/to/clip_l14_grit1m_fultune_8xe.pth

bash scripts/benchmark_magicbrush.sh \
  --stage metrics --output-dir outputs/benchmark_magicbrush_dev
```

The default `--stage all` releases EditMGT before loading metric models, keeping
their GPU memory separate. Generation timing synchronizes CUDA around the
pipeline call, excludes warmup, disk I/O and metric computation, and includes
prompt encoding, VQ encoding, Transformer sampling and decoding. One unmeasured
warmup is used per generation process by default (`--warmup 0` disables it).
Records include peak allocated/reserved GPU bytes and the actual shifted mask
schedule plus cache capture/reuse steps. Native-size output resizing occurs
outside the measured pipeline call.

Errors stop the run by default while preserving completed outputs. With
`--continue-on-error`, failures are recorded and remaining independent samples
continue. Summaries always record expected, successful and failed/unprocessed
counts; incomplete results exit with a nonzero status. Missing outputs are
never treated as successful zero-error predictions. VLM/API scoring is not
performed by these scripts.

## Mask-region-aware training and analysis

Training and dataset-based analysis now default to the local Inter-Edit release
at `/mnt/geminihzceph/public/data/Inter-Edit`. Its official `train/` and `test/`
splits are used directly: 1,099,964 training pairs and 6,250 test pairs. No custom
split manifest or archive extraction is required.

`InterEditDataset` reads gzip JSONL training metadata and source/target/mask
images directly from the tar shards. Member-offset indexes are built on first
access and cached in `data/inter_edit_tar_index` (override with
`--dataset_cache_dir`); the public dataset stays read-only. Distributed ranks
and DataLoader workers own disjoint metadata shards. Source, target and mask
receive the same resize/crop, and a crop that misses a localized edit is moved
to include a positive mask pixel. Masks use white = edit, black = preserve;
both `0/1` and `0/255` image encodings are accepted. When reducing mask
resolution, binary max pooling preserves thin edit regions; upsampling uses
nearest neighbours. The same mask conversion is used for fixed test previews.
Both values of `better_data` are included.

Training rows with genuinely empty edit annotations are skipped before batching.
Each worker remembers rejected IDs for its current iterator so repeated passes
do not reread those assets. Every exclusion is recorded under
`<dataset_cache_dir>/empty_masks/train/` (default:
`data/inter_edit_tar_index/empty_masks/train/`) in separate rank/worker/PID JSONL
journals. Records include
sample ID, mask archive/path, original mode and value range. Console warnings
are limited to the first five and every 1,000 exclusions per worker. If a
repeated worker stream has no valid samples at all, it raises an explicit error.
The dataset length remains the nominal metadata count; training is step based.

`train/train.sh` enables sparse target and edit-only diffusion, corrupting and
supervising only edit-region target tokens. Preserve-region tokens come from
the source image. To analyze the official test split on a GPU:

```bash
python scripts/analyze_edit_attention.py \
  --model-path checkpoint/EditMGT \
  --data-dir /mnt/geminihzceph/public/data/Inter-Edit \
  --num-cases 1 --layers 28 29 30 31
```

The analysis saves generated images, masks, per-layer/per-token heatmaps,
overlays, attention enrichment metrics, JSONL results, and a dataset summary.

For a separate training run using only original English prompts in both splits:

```bash
SKIP_INSTALL=1 bash train/train_en.sh
```

This reuses the existing environment. Omit `SKIP_INSTALL=1` to use the regular
training launcher's dependency setup. GPU selection, optimizer, learning rate,
batch size, sparse target, edit-only diffusion and validation sampling settings
are inherited from `train/train.sh`. Additional CLI arguments are forwarded,
for example `--max_train_steps 10000` or `--validation_num_images 10`.

The launcher sets `--dataset_language en` for both training and test previews:

| Split | Selection rule | Eligible samples before empty-mask exclusions |
|---|---|---:|
| Train | No language labels are provided; keep instructions with Latin letters and no Han characters | 475,890 |
| Test | Keep the official `English` language label | 3,413 |

Chinese and mixed Chinese/English training instructions are excluded without
translation. Official English test instructions can still contain literal
Chinese text to render in the edited image. Language selection occurs before
image/tar asset reads. Per-shard language counts are cached under
`data/inter_edit_tar_index/language_counts/`, with file locks, atomic writes and
metadata identity checks. `len(dataset)` uses the language-filtered count for
the current rank; genuinely empty masks are still excluded during iteration.
The source dataset is never rewritten.

The default output is `runs/editmgt_sparse_inter_edit_en/`, keeping the mixed
run in `runs/editmgt_sparse_inter_edit/` separate. A new English run loads the
released `checkpoint/EditMGT/editmgt` weights; automatic resume searches only
the English output directory. Override the destination with `TRAIN_OUTPUT_DIR`.
`train/train.sh` retains the original `all` language default.

Test visualizations select five original English instructions, preferring
different edit types, rather than translating the previous five Chinese cases.
Selection uses the original official test indices and is fixed across validation
steps. Explicit `--validation_sample_indices` outside the selected language
raise an error. Reports remain at startup, every 200 optimizer steps and the
final step. `validation/samples.json` records the actual cases and prompts;
`validation/selection.json` records the eligible count and selected official
indices. Each step's metadata includes the language filter.

To infer an English-trained checkpoint with the same exported English test cases:

```bash
bash scripts/infer_inter_edit_sparse.sh \
  --checkpoint runs/editmgt_sparse_inter_edit_en/checkpoint-1000 \
  --samples-manifest runs/editmgt_sparse_inter_edit_en/validation/samples.json \
  --output-dir outputs/inter_edit_sparse_en_checkpoint_1000
```

For English training with cache-consistent text and reference branches:

```bash
SKIP_INSTALL=1 bash train/train_en_cache.sh
```

This adds `--cache_conditioning --condition_cache_threshold 0.3` and writes to
`runs/editmgt_sparse_inter_edit_en_cache/`. It keeps the English-only data filter,
full reference, sparse target, original 2D RoPE positions, and single final
scatter. The separate output directory prevents automatic resume from an older
English checkpoint trained without the new attention/modulation rules.
Override the stage threshold with `CONDITION_CACHE_THRESHOLD`.

Above the saved scheduled mask-ratio threshold, all branches retain joint
attention and text uses the target's current timestep modulation. At or below
the threshold, the model uses the following access rules (rows are queries):

| Query / K,V | Text | Full reference | Active target |
|---|---|---|---|
| Text | Yes | Yes | No |
| Full reference | Yes | Yes | No |
| Active target | Yes | Yes | Yes |

In this low-noise phase, text and reference use a fixed timestep of zero in
their conditioning modulation; target continues to use its current timestep.
This applies in both double and single blocks, including the text connector
when it is time-conditioned. The training batch may contain different phases
for different samples. Prefix and target attention use separate query groups
with broadcast key masks, avoiding an additional dense Q-by-K block mask.
Training computes all branches with gradients and never persists or detaches
their K/V. Phase selection uses the scheduled unresolved fraction inside the
edit region (`mask_prob` in training), independently of legacy timestep scaling.

On inference, the saved model config automatically selects joint text/reference
KV caching. The first eligible low-noise forward recomputes the prefix under
the trained mask and captures every layer's full text/reference K/V after
normalization and RoPE, plus static pooled conditioning. Later forwards take
only target queries and skip text/reference projections, queries, modulation
and FFNs. No target K/V is cached. CFG halves have separate cache entries, and
every generation gets a fresh cache that is cleared before VQ decoding.

For example, after saving a new checkpoint:

```bash
bash scripts/infer_inter_edit_sparse.sh \
  --checkpoint runs/editmgt_sparse_inter_edit_en_cache/checkpoint-1000 \
  --samples-manifest runs/editmgt_sparse_inter_edit_en_cache/validation/samples.json \
  --output-dir outputs/inter_edit_sparse_en_cache_checkpoint_1000
```

Pass `--disable-condition-kv-cache` to recompute both context branches using the
same trained attention/modulation rules. `--disable-reference-kv-cache` also
disables both caches by default for these checkpoints. The existing cache-capture
threshold can delay capture, but cannot move it before the trained low-noise
boundary. Attention visualization and local guidance recompute the prefix to
keep live text query rows available. Per-case `condition_kv_cache` metadata
records the active branches, phase/capture thresholds, capture/reuse steps,
full text/reference lengths and layer count.

The new config fields are saved in full Transformer checkpoints and in
`condition_cache_config.json` for LoRA. Older checkpoints retain their previous
behavior and their approximate reference-only snapshot cache; joint condition
caching requires weights trained with `cache_conditioning` enabled.

To mix the local MagicBrush training release into this English/cache-conditioning run:

```bash
SKIP_INSTALL=1 bash train/train_en_cache_magicbrush.sh
```

This reads `data/magicbrush/data/train-*.parquet` directly using PyArrow, without
Hugging Face `datasets` or an image extraction step. The provided `source_img`,
`target_img`, `mask_img` and `instruction` supply the same aligned image/edit-mask
training fields as Inter-Edit. Each MagicBrush turn uses its own supplied source
image, including previously edited sources on later turns. Provided masks use
white = edit and receive the existing binary normalization, coverage-preserving
resize and shared crop. Missing/empty masks are excluded and journaled under
`data/inter_edit_tar_index/magicbrush/empty_masks/train/`; no mask is predicted
and no all-image replacement mask is generated.

The default per-example sampling probabilities are **80% Inter-Edit / 20%
MagicBrush**. Both streams repeat, so MagicBrush is sampled at that configured
rate rather than being weighted only by its smaller dataset size. Override the
probability or data location with environment variables:

```bash
SKIP_INSTALL=1 MAGICBRUSH_PROBABILITY=0.1 \
  MAGICBRUSH_DATA_DIR=./data/magicbrush \
  bash train/train_en_cache_magicbrush.sh
```

The equivalent Python options are `--magicbrush_data_dir` and
`--magicbrush_probability`. `--magicbrush_read_batch_size` defaults to 4, bounding
the image-containing parquet rows materialized together in each worker.
Initialization reads parquet footers and only the instruction column for
language-filtered sample counts. Each dataset owns disjoint rank/worker shards;
the combined stream is not sharded again by Accelerate. Mixture samples include
`dataset_name` and a dataset-prefixed `sample_key` to distinguish overlapping
numeric sample IDs. Training remains step based and retains all existing sparse
target, cache-conditioning and edit-only loss settings.

Only MagicBrush **train** shards are used for updates; its dev/test split is
excluded. Training previews remain the fixed original English cases from
Inter-Edit test. The output directory is
`runs/editmgt_sparse_inter_edit_en_cache_magicbrush/`, with the same checkpoint
and validation layout. `data_mixture/rank_XXXX.json` records each rank's data
roots, language-filtered sample counts before empty-mask exclusions, and sampling
probabilities. The previous training entry points retain their existing datasets.

### Training timestep sampling for discrete mask diffusion

Time sampling operates on `r`, the unresolved fraction of **generation-region VQ tokens**,
with `r=1` fully masked and `r=0` clean. With halo supervision enabled, this
population includes the dilated halo; preserved tokens outside it are excluded.
For context-only halo checkpoints, only the original edit region is included. The original sampler draws
`u ~ Uniform(0,1)` and sets `r=cos(pi*u/2)`; its mask ratios favor high noise
rather than being uniform in `r`.

The configurable sampler borrows bounded **logit-normal** sampling and rational
shifting from [SD3, sections 3.1 and 5.3.2](https://arxiv.org/html/2403.03206v1).
These are adapted to discrete mask probabilities, rather than treating masked
tokens as Gaussian-corrupted continuous latents. For the main component:

```text
cosine:       u ~ Uniform(0,1); r = cos(pi*u/2)
uniform:      r ~ Uniform(0,1)
logit_normal: z ~ Normal(mean, std^2); r = sigmoid(z)
shift:        r_shifted = shift*r / (1 + (shift-1)*r)
```

`shift > 1` emphasizes higher masking; `shift < 1` emphasizes lower masking.
For logit-normal draws, shift is exactly equivalent to adding `log(shift)` to
the normal mean, so these two controls do not provide independent distributions.
EDM-style unbounded log-normal noise scales are not used as mask probabilities.

An experimental English Inter-Edit/MagicBrush training entry point is provided:

```bash
SKIP_INSTALL=1 bash train/train_en_cache_magicbrush_lognormal.sh
```

| Setting | New entry-point default | Meaning |
| --- | --- | --- |
| Main distribution | `logit_normal` | Middle-noise coverage |
| Normal mean / std | `0.0 / 1.0` | Distribution before sigmoid |
| Shift | `1.0` | Identity initially; configurable |
| Original cosine mixture | `0.2` | Retain high-noise and endpoint coverage |
| Low-noise uniform mixture | `0.2` | Additional examples in `[min_masking_rate, 0.3]` |
| Remaining main mixture | `0.6` | Determined by the other two probabilities |
| Time conditioning | `mask_ratio` | Same normalized noise units in training and inference |
| CE reduction | `sample` | Normalize by each example's supervised-token count, then average examples |

The two auxiliary components are unshifted. The low-noise mixture's upper bound
is `--condition_cache_threshold`, shared with prefix isolation and fixed context
modulation. Its probability is a guaranteed sampling component, not the total
fraction below the threshold: the main and cosine components can also draw
low-noise examples. With mean=0, std=1, shift=1 and threshold=0.3, approximately
36% of draws are in the cache-conditioning phase, before finite-token rounding.
The 60/20/20 mixture is a proposed starting point, not a measured improvement.

Examples of parameter overrides:

```bash
# Emphasize high-noise examples in the main component only.
SKIP_INSTALL=1 TIMESTEP_SHIFT=1.5 \
  bash train/train_en_cache_magicbrush_lognormal.sh

# Pure logit-normal ablation, with a distinct output directory.
SKIP_INSTALL=1 TIMESTEP_COSINE_MIX=0 TIMESTEP_LOW_NOISE_MIX=0 \
  TRAIN_OUTPUT_DIR=./runs/editmgt_en_cache_lognormal_pure \
  bash train/train_en_cache_magicbrush_lognormal.sh

# Original sampling distribution, using the new time units and sample-mean loss
# to isolate the effect of the distribution in a comparison.
SKIP_INSTALL=1 TIMESTEP_SAMPLING=cosine \
  TIMESTEP_COSINE_MIX=0 TIMESTEP_LOW_NOISE_MIX=0 \
  TRAIN_OUTPUT_DIR=./runs/editmgt_en_cache_cosine_normalized \
  bash train/train_en_cache_magicbrush_lognormal.sh
```

Python flags mirror these settings: `--timestep_sampling`,
`--timestep_logit_mean`, `--timestep_logit_std`, `--timestep_shift`,
`--timestep_cosine_mix`, `--timestep_low_noise_mix`, `--time_conditioning` and
`--mask_loss_reduction`. The shell equivalents use the same uppercase names;
the last two are `TIME_CONDITIONING` and `MASK_LOSS_REDUCTION`.

New `mask_ratio` time conditioning fixes the legacy train/inference unit
discrepancy: training previously passed `r*1000` to a model that scaled time by
another 1000, whereas inference passed integer scheduler step IDs. The new model
derives target time from `mask_ratio` before both the optional text connector
and Transformer modulation, with one x1000 scaling at each embedding input.
Inference already passes the before-forward cosine/linear mask ratio, so model
time no longer depends on the chosen number of inference steps. Scheduler step
IDs still index the sampling loop. Low-noise text/reference modulation remains
fixed at zero for cache reuse.

**Training distribution and inference step placement are separate controls.**
The training sampler does not implicitly change inference's cosine/linear
remasking schedule. Inference shifting is configured separately below; it
does not change the temperature schedule or prefix-cache threshold. Continuous-diffusion SNR/MSE
loss weights are not copied into discrete cross-entropy. Sample-mean reduction
and the nonuniform distribution intentionally change training weighting; this
is not importance sampling that preserves the original token-mean objective.

The new entry point writes to
`runs/editmgt_sparse_inter_edit_en_cache_magicbrush_lognormal/` and retains the
existing English test visualizations. `timestep_sampling_config.json` is saved
at the run root and inside every checkpoint; resume checks that sampling, time
units and loss reduction match. `low_noise_sample_fraction` and
`avg_masking_rate` are logged alongside loss. Full Transformer configs save
`time_conditioning`; LoRA additionally saves `time_conditioning_config.json`.
Older checkpoints default to `legacy`, and all older training entry points
retain their original cosine sampler, legacy time units and token-mean loss.
Use a new run for the changed time convention; an old checkpoint is not silently
reinterpreted during inference or automatic resume.

### Inference mask-schedule shift

The pipeline accepts `inference_shift`, independent of the training sampler's
`TIMESTEP_SHIFT`. For a base cosine/linear mask ratio `r`, inference applies
`r_shifted = s*r / (1 + (s-1)*r)`. The shift must be positive and finite.
The default is `1.0`, preserving the unshifted schedule; values greater than
one retain more masked tokens at each intermediate step, and values below one
reveal tokens earlier. The first state remains fully masked and the terminal
state is exactly clean.

```bash
bash scripts/infer_inter_edit_sparse.sh \
  --inference-shift 1.5 \
  --output-dir outputs/inter_edit_sparse_shift_1p5
```

The existing `--checkpoint` and `--samples-manifest` options select the trained
weights and exported test cases. Use a separate `--output-dir` for each shift
when comparing generated images and timing. The vanilla comparison script also
accepts `--inference-shift`, and the original demo accepts the environment:

```bash
INFERENCE_SHIFT=1.5 python infer.py
```

In Python, pass `pipe(..., inference_shift=1.5)`. Omitting the argument uses
`pipe.scheduler.config.inference_shift`, whose default is 1.0. Per-call overrides
do not modify that saved default or leak into later calls. Scheduler configs
without the new field load with the identity shift.

For training-time English test previews:

```bash
SKIP_INSTALL=1 VALIDATION_INFERENCE_SHIFT=1.5 \
  bash train/train_en_cache_magicbrush_lognormal.sh
```

The equivalent Python option is `--validation_inference_shift`. This changes
validation only; the training distribution remains controlled by
`--timestep_sampling` and `--timestep_shift`, and the two shifts need not match.

The scheduler precomputes an `N+1` mask-ratio table. Forward `i` receives `r[i]`,
and the sampling step remasks to `r[i+1]`. The shifted table controls actual
remasking, normalized model time, the fixed low-noise prefix policy, and both
reference-only and joint text/reference cache capture. Cache decisions use
host-side scheduled values without reading GPU token counts. Integer rounding,
minimum-one-token handling and the existing requirement to make progress can
make actual mask counts differ slightly from the scheduled fractions.

For `time_conditioning="mask_ratio"` checkpoints, target time is the shifted
mask fraction. Legacy checkpoints keep their original step-based units: for
nonidentity shifts, the model receives the equivalent fractional position in
the original cosine/linear schedule, clamped to the historical endpoint times.
Integer scheduler step IDs remain unchanged and only index sampling. A shift
of 1.0 uses the original legacy model times exactly. This does not reinterpret
old weights as having been trained with normalized mask-ratio conditioning.

Because the cache threshold remains a mask fraction, a shift above one usually
delays capture and leaves fewer reuse steps; a shift below one usually does the
reverse. Shift does not by itself reduce the number of Transformer evaluations.
The temperature schedule, active target layout, complete reference and single
final feature scatter retain their existing behavior.

Validation and standalone inference metadata record `inference_shift`, the
full before-forward/terminal ratio table and legacy model times where relevant
under `inference_schedule`. Cache records include the shift and actual
capture/reuse step indices. These settings are available for comparisons; no
quality or latency improvement has been measured here.

### Training-time test visualizations

`train/train.sh` saves previews before the first update (or after resuming),
every **200 optimizer steps**, and at the final step. The default fixed test
indices are **0, 1, 2, 8, 12**, covering Local, Add, Text Editing, Texture and
Remove. Each case always uses seed `42 + sample_id`, so checkpoints are compared
on the same inputs and sampling seeds. Ground truth is used only for display.

Reports are saved under `runs/editmgt_sparse_inter_edit/validation/`:

```text
validation/
├── samples.json                 # fixed cases, full instructions and sizes
├── test_inputs.png              # source / mask overlay / GT preview
├── inputs/sample_00000/          # source, mask, mask overlay and GT PNGs
├── index.html                   # links to all completed validation steps
└── step_000200/
    ├── index.html               # full-resolution comparisons with Chinese instructions
    ├── contact_sheet.png        # source / mask overlay / GT / generated, one row per case
    ├── metadata.json            # seeds, sampling settings, EMA flag and elapsed time
    └── sample_00000/
        ├── generated.png
        └── comparison.png
```

Open `validation/index.html` in a browser, or inspect each step's
`contact_sheet.png`. When W&B logging is enabled, the comparisons are also
recorded as `validation/inter_edit` and `validation/contact_sheet`.

Use `--validation_steps` to change the interval (`0` disables previews),
`--validation_num_images` to change the case count, or
`--validation_sample_indices 0 1 2 8 12` to choose explicit test indices.
`--validation_num_inference_steps`, `--validation_guidance_scale`,
`--validation_reference_strength`, and `--validation_seed` control sampling;
`--validation_data_dir` overrides the Inter-Edit data root. Remove
`--validation_at_start` to skip the initial preview.

Validation inference enables low-noise reference KV reuse by default. Use
`--validation_reference_kv_cache_threshold 0.3` to set the switch point, or
`--validation_disable_reference_kv_cache` to recompute reference every step.
Each case's `metadata.json` record includes the actual cache capture step,
reuse steps, layer count and full reference sequence length. Training forwards
continue to process the complete reference normally and never use this cache.

Test images are read/resized once before training and reused at every interval.
Generation runs one case at a time on rank zero with the current model weights
(EMA weights when `--use_ema` is enabled). Other ranks wait during validation.
Training mode, EMA weights, reference-strength overrides and RNG states are
restored afterward, including when generation fails. Validation uses its own
scheduler instance and does not compute training losses.

To compare the same five preview cases with the original EditMGT without a mask:

```bash
bash scripts/infer_inter_edit_vanilla.sh
```

`infer_inter_edit_vanilla.py` follows `infer.py`, loading the original local
`checkpoint/EditMGT/editmgt` weights. It reads the exact source PNGs from
`runs/editmgt_sparse_inter_edit/validation/samples.json` (test indices
0, 1, 2, 8, 12), and uses built-in English translations of those five Chinese
instructions. Literal image text such as `京A88888` is preserved. The report
and metadata record both the original instruction and the actual English prompt.
For new cases, provide `instruction_en` in the manifest; entries explicitly
labelled English can use their original instruction. The training data and
training-preview prompts remain unchanged.
No mask is read or passed to the pipeline; ground
truth is used only for comparison images. Sparse target, token-type RoPE,
edit-only diffusion, local guidance and reference KV reuse are disabled.
Defaults match the sampling settings in `infer.py`: 1024 resolution, 36 steps,
CFG 6, reference strength 1.1 and FP32. Seeds are `42 + sample_id`, as in
training previews. Pass `--num-inference-steps 18 --reference-strength 1.0`
to match the training-preview step count and reference strength.
Results are saved to `outputs/inter_edit_vanilla_en/`, including an `index.html`,
`contact_sheet.png`, `metadata.json`, and per-case source/GT/generated/comparison
PNGs. Override `--samples-manifest`, `--output-dir`, `--model-path`, or `--device`
as needed. The launcher uses the existing `.venv` and installs no dependencies.

To infer the same English-prompt cases with our trained sparse pipeline:

```bash
bash scripts/infer_inter_edit_sparse.sh
```

This loads the full Transformer from
`runs/editmgt_sparse_inter_edit/checkpoint-1000/transformer`; frozen text encoders,
VQ and the sampler come from `checkpoint/EditMGT`. It reuses the same five
sources, English translations and exported test edit masks. Sparse target,
token-type RoPE and edit-only diffusion are enabled, with the halo loaded from
the trained configuration (1 for checkpoint-1000). Full reference KV reuse is
enabled below the 0.3 mask-ratio threshold; pass
`--disable-reference-kv-cache` to recompute reference every step.
Defaults match training previews: 1024 resolution, 18 steps, CFG 6, reference
strength 1.0, FP32 and seed `42 + sample_id`. Pass
`--num-inference-steps 36 --reference-strength 1.1` to match the vanilla inference
sampling settings. The original Chinese instructions and actual English inputs,
checkpoint path and per-case cache statistics are saved in the report metadata.
Results are saved under `outputs/inter_edit_sparse_checkpoint_1000_en/`, with
input PNGs in `inputs/` and generated images, comparisons, HTML and metadata in
`step_001000/`. Override `--checkpoint`, `--output-dir`, `--samples-manifest`, or
`--device` as needed. This launcher also uses the existing environment.

Before training, run a zero-shot smoke test with the original editing checkpoint:

```bash
bash scripts/test_zero_shot.sh
```

The shell wrapper creates/updates the root `.venv` with uv and runs eight cases
after loading the checkpoint once. Additional CLI arguments override its
defaults, for example
`bash scripts/test_zero_shot.sh --sample-index 10 --num-cases 16 --bf16`.
The uv setup installs `torch==2.11.0` and `torchvision==0.26.0` from the official
CUDA 13.0 wheel index before installing the remaining requirements. Override
`TORCH_VERSION`, `TORCHVISION_VERSION`, or `PYTORCH_INDEX_URL` if needed.
The smoke test defaults to the stable math SDPA backend. After it passes, use
`--sdpa-backend auto` to benchmark fused attention kernels.
On NVIDIA H20 with torch 2.1, `--bf16` is automatically downgraded to float32
because that combination can terminate in the first bf16 Linear GEMM with
SIGFPE. Upgrade torch before benchmarking bf16 on H20.

Run two-pass zero-shot editing where cross attention, rather than the dataset GT
mask, defines the editable region:

```bash
bash scripts/test_attention_mask.sh \
  --sample-index 0 --num-cases 5 --bf16 \
  --mask-threshold 0.4 --mask-closing 1 --mask-dilation 2
```

The first pass performs full-image generation and extracts
generation-image-query to instruction-text-key attention. The pooled map is
min-max normalized per case, and positions at or above `--mask-threshold` form
the second-pass edit mask. Token-grid morphological closing fills holes and
joins nearby response regions, then dilation expands the boundary for a more
complete edit region. No fixed-area top-k selection or GT mask is used.
By default the shell script uses five cases from the official Inter-Edit test
split. Pass `--assets-manifest assets/attention_mask_cases.json` to use the
standalone image cases instead.
It pools cross attention globally over all valid instruction tokens, together
with all attention heads and the selected model layers. No localization keywords
are used. Each case also saves `original_editmgt.png`, an unmasked EditMGT
baseline generated with the same seed and denoising settings as the masked run.
The masked result is saved only as `generated_token_preserve.png`; no pixel-space
compositing is applied.

Visualize the real token reveal order on held-out data cases:

```bash
bash scripts/visualize_generation_order.sh --bf16
```

For every case this writes all step frames, `generation.gif`, a contact sheet,
an edit-vs-preserve token progress curve, and a token reveal-time heatmap. The
script runs only the original EditMGT full-generation path: all image tokens
start masked, and the dataset GT mask is never passed to the model. It is used
only after inference to measure whether preserve-region tokens were actually
revealed earlier than edit-region tokens.

### Full reference + sparse target

`train/train.sh` enables `--sparse_target --sparse_target_halo 2` on Inter-Edit and
writes to `runs/editmgt_sparse_inter_edit`. For a custom training command, use
`--instance_dataset InterEditDataset --instance_data_dir /mnt/geminihzceph/public/data/Inter-Edit`
and add the sparse flags along
with `--train_edit_model True`; sparse mode automatically enables
`--edit_only_diffusion` and requires dataset edit masks.

The target edit mask is max-pooled onto the Transformer grid after the target
stem/down block, then expanded by the halo radius (in Transformer-grid tokens).
New training defaults to **two rings** of context, using a square 5x5 max-pooling
kernel, up from the previous one-ring 3x3 kernel. With the 1024-pixel input,
64x64 VQ grid and 32x32 Transformer grid, radius 2 adds up to approximately
64 pixels of context on each side, clipped at image boundaries. New sparse
training defaults to `--supervise_target_halo`: active Transformer cells are
mapped back onto the VQ grid to form the expanded generation region. Tokens
masked inside either the original ROI or this halo receive the same target-GT
cross-entropy supervision. The expanded region shares the sampled masking rate,
so halo tokens are trained for reconstruction rather than receiving a trivial
copy loss on an always-clean source input. Preserved VQ tokens outside the
expanded region still come from source and receive no loss. Reference remains
complete.

Transformer gather receives the **original** edit mask and dilates it once;
corruption, CE and the sampler use the expanded mask. This prevents accidental
double dilation. Whole VQ patches covered by active Transformer cells are
included, including alignment to the stem grid at radius zero.

All training wrappers accept `SPARSE_TARGET_HALO` (or `--sparse_target_halo`):

```bash
SKIP_INSTALL=1 SPARSE_TARGET_HALO=2 \
  bash train/train_en_cache_magicbrush_lognormal.sh
```

Set the radius to 1 for the previous context width, or 0 to disable the extra
context ring. The radius is saved with the sparse checkpoint; sparse inference
and test visualization use the saved value unless explicitly overridden.
Full checkpoints also save `supervise_target_halo`; LoRA stores it alongside the
radius in `sparse_target_config.json`. Older checkpoints without that field use
context-only halo. To train that earlier behavior, pass
`--no-supervise_target_halo` to any training wrapper. The new default can also
be enabled when resuming previous weights; no extra learned parameters or
optimizer state are introduced. Training logs include `generation_token_ratio`
and `halo_supervised_token_count`. The debug mask panel shows the expanded
generation mask next to the original pixel mask.

Only these active target features are gathered before the first block. They
pass continuously through every double and single Transformer block, without
target KV caching or intermediate scatter. The reference sequence remains full
at the normal stem resolution. After the last block, active features are
scattered once into the saved pre-Transformer target grid, retaining background
features, before the projection, up block and MLM head.

RoPE IDs use `[token_type, original_row, original_column]`, with text/reference/
target types `0/1/2`. Gathering preserves the original spatial coordinates.
Different samples retain their own active sets and position IDs; batches pad
to the largest active count and mask padded keys in every attention block.
The conditioning includes each sample's active bounding-box `(h,w)`, including
halo, measured on the Transformer grid; empty masks use `(0,0)`. These two values
receive sinusoidal embeddings and a residual conditioning MLP alongside the
existing CLIP/size/crop/aesthetic conditioning. Its output projection starts at
zero, preserving the original conditioning weights when upgrading a checkpoint.

Use the pipeline with a source image and an edit mask (white = edit):

```python
result = pipe(
    prompt="Add a giraffe in the field.",
    reference_image=[source_image],
    mask_image=edit_mask,
    height=1024,
    width=1024,
    sparse_target=True,
    sparse_target_halo=2,
    use_reference_kv_cache=True,
    reference_kv_cache_threshold=0.3,
)
```

For halo-supervised checkpoints, sparse inference generates the same expanded
region used for training, and source VQ tokens outside it are preserved at
every denoising step. For older/context-only checkpoints, the original edit mask
defines generation and the halo remains clean source context.
The edit/halo active set stays fixed as masked tokens are revealed. A sparse
checkpoint selects sparse inference automatically when `mask_image` is present;
`sparse_target=False` explicitly runs the full target sequence. Calls without a
mask use the full sequence, including attention-based mask discovery.

Original dense checkpoints still load with their original conditioning and
RoPE behavior. `model.enable_sparse_target(halo=1)` upgrades a loaded model;
call it before creating the optimizer or adding LoRA adapters. Sparse training
saves the new conditioning weights and mode in the checkpoint; LoRA also saves
the trainable shape MLP. Start a new run when migrating from dense training,
since the added parameters change the optimizer state. Fine-tuning is needed
to learn the new shape conditioning and adapt to branch-type RoPE; zero-shot
quality and speedups are not guaranteed.

### Low-noise reference KV reuse

Inference enables `use_reference_kv_cache=True` by default when a reference is
provided. The source image is VQ-encoded once per generation. High-noise steps
run the full joint reference branch. At the first eligible low-noise step, one
full reference forward captures K/V at **every double and single block**. K is
stored after QK normalization and reference RoPE; V retains the full reference
sequence. Subsequent forwards pass `reference_image_hidden_states=None` and
`reference_image_ids=None`, skipping the reference stem, queries, Transformer
updates and MLPs. Their attention uses text/active-target queries against
text/active-target keys plus the cached full-reference K/V. Sparse target tokens
still traverse every block and scatter only once at the end.

`reference_kv_cache_threshold` is the scheduled remaining masked fraction
**within editable tokens**, before a forward: `cos(i / N * pi / 2)` for cosine,
or `1 - i / N` for linear scheduling. It is a schedule proxy, not an exact GPU
mask count. The default `0.3` captures only when this fraction reaches 30% or
less. Larger values begin reuse earlier; `0` or `use_reference_kv_cache=False`
recomputes reference every step. Capture is skipped if no later step can reuse
the cache. For 18 steps with the cosine schedule and threshold `0.3`, capture
occurs at step index 15 and reuse at 16 and 17 (indices start at zero).

Caches belong to a single generation, retain separate unconditional/conditional
CFG batches, and are released before VQ decoding. They never cache target K/V
or survive into another image, prompt, edit mask, model update or training step.
Reference strength and attention visualization remain supported during reuse;
`pipe.last_reference_cache_info` records the switch and reuse steps.

This is an approximate inference optimization: reference states in the joint
model depend on the target, so freezing a low-noise snapshot is not equivalent
to recomputing the full branch at each step. The threshold controls this quality
and speed tradeoff. The implementation does not make reference attention
independent of target or change the training attention pattern.

Run the CPU regression tests in a working repository environment:

```bash
python -m pytest tests -q
```

## 📄 License
This project is licensed under the `CCBY-4.0` License. See `LICENSE` for details.

## 📑 Citation
```
@article{chow2025editmgt,
  title={EditMGT: Unleashing Potentials of Masked Generative Transformers in Image Editing},
  author={Chow, Wei and Li, Linfeng and Kong, Lingdong and Li, Zefeng and Xu, Qi and Song, Hang and Ye, Tian and Wang, Xian and Bai, Jinbin and Xu, Shilin and others},
  journal={arXiv preprint arXiv:2512.11715},
  year={2025}
}
```

## 🙏 Acknowledgements
We thank all contributors and the research community for their valuable feedback and support.
