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
`checkpoint/EditMGT/editmgt` weights. It reads the exact source PNGs and original
instructions from `runs/editmgt_sparse_inter_edit/validation/samples.json`
(test indices 0, 1, 2, 8, 12). No mask is read or passed to the pipeline; ground
truth is used only for comparison images. Sparse target, token-type RoPE,
edit-only diffusion, local guidance and reference KV reuse are disabled.
Defaults match the sampling settings in `infer.py`: 1024 resolution, 36 steps,
CFG 6, reference strength 1.1 and FP32. Seeds are `42 + sample_id`, as in
training previews. Pass `--num-inference-steps 18 --reference-strength 1.0`
to match the training-preview step count and reference strength.
Results are saved to `outputs/inter_edit_vanilla/`, including an `index.html`,
`contact_sheet.png`, `metadata.json`, and per-case source/GT/generated/comparison
PNGs. Override `--samples-manifest`, `--output-dir`, `--model-path`, or `--device`
as needed. The launcher uses the existing `.venv` and installs no dependencies.

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

`train/train.sh` enables `--sparse_target --sparse_target_halo 1` on Inter-Edit and
writes to `runs/editmgt_sparse_inter_edit`. For a custom training command, use
`--instance_dataset InterEditDataset --instance_data_dir /mnt/geminihzceph/public/data/Inter-Edit`
and add the sparse flags along
with `--train_edit_model True`; sparse mode automatically enables
`--edit_only_diffusion` and requires dataset edit masks.

The target edit mask is max-pooled onto the Transformer grid after the target
stem/down block, then expanded by the halo radius (in Transformer-grid tokens).
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
    sparse_target_halo=1,
    use_reference_kv_cache=True,
    reference_kv_cache_threshold=0.3,
)
```

Sparse inference keeps source VQ tokens outside the edit mask at every denoising
step. Halo tokens provide context but are not added to the generation/loss mask.
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
