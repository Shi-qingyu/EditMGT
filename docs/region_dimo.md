# Region-DiMO on EditMGT

This feature adds region-conditioned, categorical one-step distillation without replacing EditMGT's SFT trainer, Pipeline, benchmark entries or sparse utilities. The only backbone change is `Transformer2DModel.forward(noise_emb_perturb=0.0)`: target embeddings can receive DiMO noise before `down_block`; the source embedding branch is unchanged. Zero consumes no RNG and is bitwise equivalent to the reviewed main forward.

## Recipe and compatibility

Use `configs/dimo/region_dimo_v3.example.json` (also supplied as `region_dimo_v3.json`). Fill a **private copy**; empty paths deliberately fail validation. Student/Auxiliary `embed` and `mlm_layer` freeze before AdamW construction; the frozen head remains differentiable with respect to its input. New defaults are sigma=0.3, Teacher CFG=6, 100 optimizer-update warmup steps for each optimizer, training Halo=1/supervise=true. Student and Auxiliary always use CFG=1.

Preserved: AdamW (LR 1e-6, WD .01, betas .9/.999), detached FKL logit gradient `p_A - p_T`, per-sample Q mean, r_init=1 (explicit .5 ablation supported), fresh-data Student → Auxiliary sequence, hard CE smoothing .1, fake_rounds=1, EMA .9999 with `min(decay,(1+step)/(10+step))` once per Student update. The first warmup optimizer update uses LR=0, matching diffusers' constant-with-warmup semantics; the scheduler reaches nominal LR after 100 updates. Auxiliary warmup counts its own updates when fake_rounds>1.

Old JSON files that omit freeze/noise/warmup fields resolve freeze=false, sigma=0, constant LR, Teacher CFG=4. `region_dimo_b0.example.json` explicitly reproduces the old B0 algorithm **from fresh SFT initialization**. This does not permit loading historical v1/v2 optimizer states into v3. The historical 6000-step run is not modified. Use a new disjoint output directory for every recipe change.

Source Teacher profiles remain separate from runtime Halo: the loaded checkpoint's sparse/time/cache/codebook profile is checked against `expected_profile`; only afterward are the four Transformer copies configured for distillation Halo. The original mask M is passed to every forward, G is expanded once on the actual stem grid, Q is a subset of G, and tokens outside G are locked to the source. Teacher training queries use one random pseudo-mask at a floating time ratio; they are not the 18-step benchmark Teacher algorithm. No training forward uses inference KV caches. The checkpoint's cache-conditioned attention prefix/threshold remains part of its forward mathematics; disabling inference KV reuse does not disable this trained profile. Student noise is enabled in the differentiable generation and in updated-Student Auxiliary target generation, not in Teacher/Auxiliary distribution queries.

## Explicit assets and data

Select **one** Teacher source:

- `sft_run_dir` / `SFT_RUN_DIR`: search only this run's `checkpoint-N/transformer` directories; select the largest complete numeric step, report incomplete candidates, fail on an incompatible profile. Pin a checksum-verified copy in the new output before ranks load it.
- `teacher_transformer_dir` / `DIMO_TEACHER_DIR`: an explicit full Transformer directory or its checkpoint-N parent; fail rather than trying other models.
- `teacher_hf`: `{ "repo_id": "QingyuShi/SparseEdit", "revision": "<40-character commit SHA>", "cache_dir": "./cache/huggingface", "transformer_subdir": "<inspected directory>" }`. Only an already-downloaded local immutable hub snapshot is used. Empty `transformer_subdir` is accepted only if inspection finds exactly one Transformer. Do not use `main` as a revision. The reviewed release layout was `checkpoint-8000/transformer`; this path is not a default in the loader. Pre-download the explicitly pinned snapshot separately if needed.

Set `base_model_dir` / `BASE_MODEL_DIR` independently. It must contain explicit local `editmgt`, CLIP `text_encoder`/`tokenizer`, `llm_encoder` (including tokenizer), `vqvae`, `scheduler`. Missing local Gemma assets fail before upstream's remote fallback. Full strict keys/shapes, finite weights, codebook and mask-token compatibility are required. Unmerged adapter checkpoints are rejected; `lora_part_enable` is explicit and remains true for the reviewed SFT path.

For a fixed manifest, set `data_manifest`, its SHA256, `data_roots`, `expected_dataset_counts` and `verify_asset_hashes`. Relative asset locators must remain under their dataset root; every source/target/mask locator and English prompt is checked, and decoded masks/checksums are checked when samples are consumed. Historical 50K composition: Inter-Edit 33237, MagicBrush 8763, MaskEdit 8000; not a 1:1:1 mixture. The length must divide the requested world size. There is no manifest-to-stream fallback. Alternatively explicitly select `data_mode=local_stream`, configure both local training roots, cache directory, language and MagicBrush probability. Data readers retain the upstream preprocessing and rank sharding. No independent train/test overlap audit is performed; do not claim leakage was excluded.

## Launch and bounded preflight

Prepare the repository `.venv` yourself, or set `DIMO_PYTHON` to an existing interpreter. No package installation or precision fallback is implicit. Required dependencies already appear in upstream requirements; the migration was validated with Python 3.10, torch 2.5.1+cu121, accelerate 1.0.1, transformers 4.47.1, diffusers .32.1. Upstream requirements are retained unchanged; a new environment must select a compatible CUDA/PyTorch build deliberately.

Environment A, scheduler-managed GPUs and repository `.venv`:

```bash
export DIMO_CONFIG=./private/dimo_v3.json
# Fill this private JSON with Teacher/base/data/output paths first.
# Preserve the scheduler's CUDA_VISIBLE_DEVICES; NUM_PROCESSES defaults to visible GPUs.
.venv/bin/python -m src.dimo.preflight --config "$DIMO_CONFIG"
bash scripts/dimo/run_formal.sh --stop-after-preflight
```

Environment B, externally managed Python and explicitly chosen GPUs:

```bash
export DIMO_PYTHON=/path/to/prepared/env/bin/python
export DIMO_CONFIG=/path/to/private/dimo_v3.json
export GPU_IDS=0,1                 # or explicit GPU UUIDs; optional override
export NUM_PROCESSES=2             # optional; validated against visible count
export BASE_MODEL_DIR=/path/to/EditMGT-base
export DIMO_TEACHER_DIR=/path/to/SFT/checkpoint-N/transformer
export DIMO_OUTPUT_DIR=/path/to/new-distillation-run
# Leave sft_run_dir and teacher_hf unset/empty when choosing DIMO_TEACHER_DIR.
"$DIMO_PYTHON" -m src.dimo.preflight --config "$DIMO_CONFIG"
bash scripts/dimo/run_formal.sh --stop-after-preflight
```

A formal preflight performs `preflight_steps` (20 in presets), saves/reloads full training state and verifies data replay, then stops with this flag. For an explicitly bounded one-update server smoke use a separate private config/output with `max_steps=preflight_steps=1`, and explicitly choose the test batch layout. This one-update warmup smoke has zero LR; nonzero updates and exact continuation are also tested with Tiny models. Starting the full budget is a later user decision. Do not run historical control scripts or point v3 at B0.

The launcher respects CUDA_VISIBLE_DEVICES unless GPU_IDS is explicitly set; process count defaults to the actual visible count. GPU UUIDs can make external allocations unambiguous. Every DDP rank holds Teacher/Student/Auxiliary/EMA replicas plus encoders/VQ. DDP does not divide four-model memory across cards. Hardware, per-phase peaks, world size and `effective_batch=world_size*micro_batch*accumulation` are recorded. OOM fails with diagnostic suggestions; it never changes precision, batch or resolution.

## Checkpoints, inference and export

Checkpoints save every 2000 **Student optimizer updates** and at preflight/end. Each contains all three trainable/EMA states, both optimizers/schedulers, rank RNG/data offsets and resolved algorithm/environment/source metadata. Temporary files and hashes are verified before atomic publication. Only then can retention remove this exact v3 run's older owned checkpoints; keep the latest 5 plus explicit milestones. Old B0 and unrelated directories are excluded. `--resume latest` is strict on config, source fingerprint and world size; only max_steps may increase. Extending a constant-with-warmup budget does not restart either scheduler. Full local training checkpoints use trusted PyTorch serialization; inference exports use safetensors.

```bash
python -m src.dimo.infer --checkpoint /path/to/new-run/checkpoint-2000 \
  --model ema --base-model-dir /path/to/EditMGT-base \
  --source ./source.png --mask ./mask.png --prompt 'Make the object red.' \
  --output ./edit.png --halo 1 --seed 42
python -m src.dimo.export --checkpoint /path/to/new-run/checkpoint-2000 \
  --model ema --output ./exports/dimo-ema
```

This is one independent Categorical sample of Student/EMA logits at CFG=1, with source-token hard lock, M/G/r_init and direct VQ decode; it is not `Pipeline(num_inference_steps=1)`. Model, text encoders and VQ are eval/frozen, and sampling is no-grad. Source/mask are full-frame square-resized as in upstream benchmarks and output is restored to native size. Ordinary inference uses sigma=0; explicit diagnostic noise is labeled in the output JSON. Halo defaults to the saved runtime profile unless explicitly overridden. Optional `--preserve-unmasked-pixels` enables RGB compositing and labels it; the default is direct decode. Every generated image has a sidecar containing the real inference protocol and model origin.

## Benchmark protocol

```bash
python -m src.dimo.benchmark --benchmark inter_edit --backend ema \
  --checkpoint /path/to/new-run/checkpoint-2000 --model-path /path/to/EditMGT-base \
  --data-dir /path/to/Inter-Edit --language en --output-dir ./outputs/dimo-ema \
  --guidance-scale 1 --num-inference-steps 1 --sparse-target-halo 1 \
  --num-gpus 2 --bf16 --no-preserve-unmasked-pixels --metrics l1,l2
```

This lightweight example explicitly selects L1/L2. Current upstream Inter-Edit defaults are LPIPS/AlphaCLIP-in/AlphaCLIP-out/boundary; MagicBrush defaults are L1/L2/CLIP-I/DINO/CLIP-T. Select only metric names/assets supported by the current upstream suite, and explicitly supply its offline metric-model options if required. Inter-Edit CLIP-T requires true target captions; never replace captions with editing instructions. For MagicBrush, use `--benchmark magicbrush` with its upstream `--mode`, `--manifest`/`--data-dir` and caption options. The wrapper reuses upstream selection, multi-turn source dependencies, prompt/mask decoding, sampling seed keys, GPU sharding and scoring. A process-local adapter supplies the Student/EMA runner; original benchmark entry files remain byte-for-byte unchanged, including spawned-worker behavior.

For Teacher comparisons choose `--backend teacher`, the explicit SFT checkpoint, `--guidance-scale 6 --num-inference-steps 18` (or 1), and explicit Halo/cache thresholds. Both benchmark run.json and per-image records distinguish Teacher iterative CFG6/cache policy from Student CFG1/no-cache categorical sampling. Use identical dataset selection, seeds, output composition and chosen comparison Halo; never mix RGB composites and direct decodes in one table. Historical runs must be interpreted using their own saved metadata, not inferred from this recipe. No new full benchmark is launched by this migration.

Telemetry reuses existing logits: loss/LR/nonfinite checks every 10, grad norms 50, entropy/Teacher–Aux gap 100, token diversity 200, fixed per-rank direct-decode samples 500. Collective aggregation is called by all ranks. Preview sampling preserves training RNG; fixed inputs are stored separately under validation. `source_metadata` records Git SHA and effective source file hashes, because local uncommitted feature revisions also need identification.

## Sources and license

Reviewed EditMGT main: [Shi-qingyu/EditMGT](https://github.com/Shi-qingyu/EditMGT/tree/a8c0a2961173448b14ad24e5f1868ca4ca0543eb), retaining its [CC-BY-4.0 notice](https://github.com/Shi-qingyu/EditMGT/blob/a8c0a2961173448b14ad24e5f1868ca4ca0543eb/LICENSE) and existing copyright headers. Imported Region-DiMO modules are adapted from the supplied code snapshot. Algorithm reference: [yuanzhi-zhu/DiMO](https://github.com/yuanzhi-zhu/DiMO/tree/24613741ec9ca730273a6ebc822327878d5a086b); reviewed train_DIMO_Meissonic.py, config and Transformer. No official third-party source files are vendored (the reviewed official revision has no root LICENSE file).

Differences from the official recipe are explicit: editing masks/Halo/reference branch, source hard lock, r_init=1 versus .5, per-sample regional FKL normalization versus global reduction, AdamW WD=.01 versus Adam WD=0, Teacher CFG6 versus4, and conditional Student/Auxiliary queries. Freeze/noise/warmup/EMA formulas follow the reviewed implementation.
