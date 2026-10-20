"""Shared benchmark generation, resumable outputs and offline scoring."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import time
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.mask_schedule import validate_inference_shift
from src.mask_utils import MAGICBRUSH_MASK_DECODING, normalize_edit_mask_image, resize_binary_edit_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def safe_name(value):
    value = str(value)
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip(".") or "sample"
    return clean if clean == value else clean + "_" + hashlib.sha256(value.encode()).hexdigest()[:10]


def sample_seed(base_seed, key):
    return (base_seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")) % (2**63 - 1)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def add_benchmark_arguments(parser, benchmark, native_editmgt=False):
    from src.benchmark_metrics import DEFAULT_METRICS

    model = parser.add_mutually_exclusive_group()
    model.add_argument("--checkpoint", type=Path, help="Our full or LoRA checkpoint; required for generation unless --vanilla.")
    model.add_argument("--vanilla", action="store_true", help="Use the released EditMGT Transformer.")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoint/EditMGT"))
    suffix = "_vanilla" if native_editmgt else ""
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / f"benchmark_{benchmark}{suffix}")
    parser.add_argument("--stage", choices=["all", "generate", "metrics"], default="all")
    parser.add_argument("--resume", action="store_true", help="Reuse completed images only when generation config/selection match.")
    parser.add_argument("--continue-on-error", action="store_true", help="Write failures and continue; incomplete benchmarks exit nonzero.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-device", default=None, help="Defaults to --device; scoring runs after EditMGT is released.")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=18)
    parser.add_argument("--inference-shift", type=validate_inference_shift, default=1.0,
                        help="Compatibility option; only 1 is accepted, and benchmarks always use cosine.")
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--negative-prompt", default="",
                        help="Negative text for both vanilla and sparse models; empty avoids suppressing text/logo tasks.")
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--reference-kv-cache-threshold", type=float, default=0.8)
    parser.add_argument("--condition-cache-threshold", type=float, default=0.8,
                        help="Runtime text/reference prefix phase; overrides the checkpoint's saved threshold.")
    parser.add_argument("--cache-first-step", action="store_true",
                        help="Ablation: capture text/reference KV at the first forward, then reuse it for "
                             "all remaining forwards. Overrides capture timing, retaining the configured prefix phase.")
    parser.add_argument("--disable-reference-kv-cache", action="store_true")
    parser.add_argument("--disable-condition-kv-cache", action="store_true")
    parser.add_argument("--sparse-target-halo", type=int, default=0,
                        help="Transformer-cell halo; default 0 disables dilation, matching training.")
    parser.add_argument("--preserve-unmasked-pixels", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt into source-RGB compositing at generation and native export resolutions; default preserves only source tokens.")
    parser.add_argument("--mask-mode", choices=["provided", "none"], default="provided")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup generations before each generation process.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first N samples (Inter-Edit/PIE-Bench) or WHOLE sessions (MagicBrush).")
    parser.add_argument("--save-inputs", action="store_true", help="Also save source, original mask and GT when available; uses additional disk space.")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS[benchmark]))
    parser.add_argument("--metric-resolution", type=int, default=None, help="Optional square scoring size; default follows native benchmark sizes.")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--alpha-clip-ckpt", type=Path, default=None)
    parser.add_argument("--alpha-clip-model", default="ViT-L/14")
    parser.add_argument("--dino-repo", type=Path, default=None, help="Optional local facebookresearch/dino clone for offline use.")
    parser.add_argument("--dino-weights", type=Path, default=None)
    parser.set_defaults(native_editmgt=native_editmgt)
    if native_editmgt:
        from src.v2_model import negative_prompt

        # Original dense EditMGT weights and infer.py sampling defaults. These
        # entry points never accept an editing mask or composite source pixels.
        parser.set_defaults(
            vanilla=True, mask_mode="none", num_inference_steps=36,
            guidance_scale=6.0, reference_strength=1.1, negative_prompt=negative_prompt,
            sparse_target_halo=0, preserve_unmasked_pixels=False,
            disable_reference_kv_cache=True, disable_condition_kv_cache=True,
            reference_kv_cache_threshold=0.0,
        )


def validate_arguments(parser, args):
    if args.native_editmgt and (
        args.checkpoint is not None or not args.vanilla or args.mask_mode != "none"
        or not args.disable_reference_kv_cache or not args.disable_condition_kv_cache
        or args.preserve_unmasked_pixels or args.sparse_target_halo != 0
    ):
        parser.error(
            "Native EditMGT entries use the released --model-path without a training --checkpoint, "
            "mask input, sparse target, KV reuse or pixel compositing. Use the general benchmark entry for other protocols."
        )
    if args.stage != "metrics" and args.checkpoint is None and not args.vanilla:
        parser.error("Generation requires --checkpoint or --vanilla")
    if args.resolution < 32 or args.resolution % 32 or args.num_inference_steps < 1 or args.warmup < 0:
        parser.error("Use resolution divisible by 32, positive inference steps and nonnegative warmup")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.sparse_target_halo is not None and args.sparse_target_halo < 0:
        parser.error("--sparse-target-halo must be nonnegative")
    if not 0 <= args.reference_kv_cache_threshold <= 1:
        parser.error("Cache threshold must be in [0,1]")
    if not 0 < args.condition_cache_threshold <= 1:
        parser.error("--condition-cache-threshold must be in (0,1]")
    if args.cache_first_step and (args.vanilla or args.num_inference_steps < 2
                                 or args.disable_reference_kv_cache or args.disable_condition_kv_cache):
        parser.error("--cache-first-step requires a trained cache-conditioned checkpoint, at least two "
                     "inference steps and enabled condition/reference caching")
    if args.metric_resolution is not None and args.metric_resolution < 1:
        parser.error("--metric-resolution must be positive")
    if not all(math.isfinite(value) for value in (args.guidance_scale, args.reference_strength)):
        parser.error("Guidance and reference strength must be finite")


def generation_config(args, benchmark=None):
    keys = ("model_path", "vanilla", "device", "resolution", "num_inference_steps", "inference_shift",
            "guidance_scale", "negative_prompt", "reference_strength", "reference_kv_cache_threshold",
            "disable_reference_kv_cache", "disable_condition_kv_cache", "sparse_target_halo",
            "mask_mode", "preserve_unmasked_pixels", "bf16", "seed", "warmup", "save_inputs")
    config = {key: getattr(args, key) for key in keys}
    if not args.native_editmgt:
        # Include the runtime phase so --resume cannot mix the historical
        # checkpoint phase with the new inference policy.
        config["condition_cache_threshold"] = args.condition_cache_threshold
    if args.cache_first_step:
        # Only add this policy when selected, preserving resume compatibility
        # for existing runs that use the normal threshold-based schedule.
        config.update(cache_first_step=True, condition_kv_cache_reuse_steps=args.num_inference_steps - 1,
                      allow_early_condition_cache=True)
    config["checkpoint"] = str(args.checkpoint.resolve()) if args.checkpoint else None
    if args.checkpoint:
        root = args.checkpoint.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        files = sorted(set(root.glob("*.json")) | set(root.glob("*.safetensors")) | set(root.glob("*.bin")) |
                       set(root.glob("transformer/*.json")) | set(root.glob("transformer/*.safetensors")) |
                       set(root.glob("transformer/*.bin")))
        config["checkpoint_files"] = [file_identity(path) for path in files]
    config["input_preprocessing"] = "full-frame square resize; prediction restored to input size; no GT used for inference"
    config["output_compositing"] = (
        "original_pixel_mask_at_generation_and_native_resolution"
        if args.mask_mode == "provided" and args.preserve_unmasked_pixels else "none"
    )
    if args.native_editmgt:
        config["model_protocol"] = "native_editmgt_mask_free_no_cache"
    if benchmark == "magicbrush" and args.mask_mode == "provided":
        # Prevent --resume from mixing images made with the old luminance mask
        # and the corrected alpha decoder. Mask-free baseline settings stay intact.
        config["edit_mask_decoding"] = MAGICBRUSH_MASK_DECODING
    return config


class BenchmarkRun:
    def __init__(self, args, benchmark, selection=None, dataset=None):
        import fcntl

        self.root = args.output_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / ".benchmark.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_path = self.root / "run.json"
            if args.stage == "metrics":
                self.info = json.loads(run_path.read_text(encoding="utf-8"))
                self.selection = json.loads((self.root / "selection.json").read_text(encoding="utf-8"))
                if self.info["benchmark"] != benchmark:
                    raise ValueError("Saved output belongs to a different benchmark")
            else:
                if not selection:
                    raise ValueError("The selected benchmark subset is empty")
                self.selection = selection
                generation_selection = [{key: value for key, value in task.items() if key != "caption"} for task in selection]
                settings = {"benchmark": benchmark, "dataset": dataset, "generation": generation_config(args, benchmark),
                            "selection_sha256": hashlib.sha256(json.dumps(generation_selection, sort_keys=True).encode()).hexdigest()}
                if run_path.exists():
                    self.info = json.loads(run_path.read_text(encoding="utf-8"))
                    if not args.resume or self.info["settings"] != settings:
                        raise ValueError("Output already exists: use --resume with identical generation settings/selection, or a new output directory")
                    # Captions affect scoring, not image generation. A later
                    # caption file can be attached without regenerating images.
                    atomic_json(self.root / "selection.json", selection)
                else:
                    self.info = {"version": 1, "benchmark": benchmark, "settings": settings,
                                 "created_at": datetime.now(timezone.utc).isoformat()}
                    atomic_json(self.root / "selection.json", selection)
                    atomic_json(run_path, self.info)
            self.by_key = {task["key"]: task for task in self.selection}
            if len(self.by_key) != len(self.selection):
                raise ValueError("Duplicate benchmark sample keys")
        except BaseException:
            self._lock.close()
            raise

    def close(self):
        if self._lock is not None:
            self._lock.close()

    def record_path(self, key):
        return self.root / "records" / (safe_name(key) + ".json")

    def record(self, key):
        path = self.record_path(key)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def image_path(self, task):
        return self.root / task["prediction"]


class SkippedBenchmarkSample(ValueError):
    """A known invalid benchmark annotation that should not abort a worker."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def inter_edit_benchmark_mask(args, run, resolver, task):
    """Check Inter-Edit mask eligibility before model loading or scoring."""
    spec = task.get("mask")
    if (not spec or (spec.get("kind") == "tar" and not spec.get("member"))
            or (spec.get("kind") == "file" and not spec.get("path"))):
        raise SkippedBenchmarkSample("missing_mask", "No mask annotation")
    try:
        mask = resolver.edit_mask(spec, "inter_edit")
    except (FileNotFoundError, KeyError) as error:
        raise SkippedBenchmarkSample("missing_mask", str(error)) from error
    if mask.getbbox() is None:
        raise SkippedBenchmarkSample("empty_mask", "Annotation mask has no editable pixels")
    generation = run.info["settings"]["generation"]
    if generation.get("mask_mode", args.mask_mode) == "provided":
        resolution = generation.get("resolution", args.resolution)
        if resize_binary_edit_mask(mask, (resolution, resolution)).getbbox() is None:
            raise SkippedBenchmarkSample("empty_mask_after_resize", "Mask has no editable pixels at generation resolution")
    return mask


def record_skipped_benchmark_sample(run, task, error, stage):
    record = {"status": "skipped", "key": task["key"], "reason": error.reason,
              "error": str(error), "stage": stage}
    for field in ("test_index", "edit_type", "language"):
        if field in task:
            record[field] = task[field]
    atomic_json(run.record_path(task["key"]), record)
    print(f"Skipping {task['key']}: {error.reason}: {error}", flush=True)


def skipped_benchmark_samples(run):
    skipped = []
    for task in run.selection:
        record = run.record(task["key"])
        if record and record.get("status") == "skipped":
            skipped.append(record)
    return skipped


class InferenceRunner:
    def __init__(self, args):
        from src.editmgt import init_edit_mgt
        from src.transformer import load_lora_adapter

        self.args, self.warmed_up = args, False
        checkpoint, transformer_path = args.checkpoint, None
        if checkpoint:
            if (checkpoint / "transformer" / "config.json").is_file():
                transformer_path = checkpoint / "transformer"
            elif (checkpoint / "config.json").is_file():
                transformer_path = checkpoint
            elif not (checkpoint / "lora_config.json").is_file():
                raise FileNotFoundError(f"No full Transformer or LoRA checkpoint at {checkpoint}")
        self.pipe = init_edit_mgt(args.device, enable_bf16=args.bf16, base_model_path=args.model_path,
                                  transformer_path=str(transformer_path) if transformer_path else None,
                                  condition_cache_threshold=args.condition_cache_threshold)
        if checkpoint and transformer_path is None:
            load_lora_adapter(self.pipe.transformer, str(checkpoint))
            # LoRA loading restores its saved phase after base initialization.
            from src.editmgt import set_inference_cache_threshold

            set_inference_cache_threshold(self.pipe.transformer, args.condition_cache_threshold)
        if args.native_editmgt:
            config = self.pipe.transformer.config
            if (config.sparse_target or config.token_type_rope or config.cache_conditioning
                    or config.time_conditioning != "legacy"):
                raise ValueError(
                    "Native EditMGT requires the original dense Transformer in --model-path/editmgt; "
                    "the supplied model enables sparse/type-RoPE/cache-conditioned or normalized-time changes."
                )
        for module in (self.pipe.transformer, self.pipe.vqvae, self.pipe.text_encoder, self.pipe.text_encoder_t5):
            if module is not None:
                module.eval().requires_grad_(False)
        self.pipe.local_guidance = 0.0
        self.pipe.set_progress_bar_config(disable=True)
        device = torch.device(args.device)
        self.metadata = {
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "dtype": "bfloat16" if args.bf16 else "float32",
            "time_conditioning": self.pipe.transformer.config.time_conditioning,
            "cache_conditioning": self.pipe.transformer.config.cache_conditioning,
            "condition_cache_threshold": self.pipe.transformer.config.condition_cache_threshold,
            "checkpoint_condition_cache_threshold": getattr(
                self.pipe.transformer, "_loaded_condition_cache_threshold", None,
            ),
            "supervise_target_halo": self.pipe.transformer.config.supervise_target_halo,
            "warmup_generations_per_process": args.warmup,
        }
        if args.native_editmgt:
            self.metadata["model_protocol"] = "native_editmgt_mask_free_no_cache"

    @torch.inference_mode()
    def generate(self, source, mask, instruction, seed):
        args, pipe = self.args, self.pipe
        native_source = source.convert("RGB")
        native_size = native_source.size
        source = native_source.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        provided = args.mask_mode == "provided"
        native_mask = None
        if provided:
            if mask is None:
                raise ValueError("Mask-provided evaluation requires an annotation; no mask is synthesized")
            mask = normalize_edit_mask_image(mask)
            native_mask = resize_binary_edit_mask(mask, (native_size[1], native_size[0]))
            mask = resize_binary_edit_mask(mask, (args.resolution, args.resolution))
            if mask.getbbox() is None:
                raise ValueError("Empty benchmark mask")
        sparse = provided and pipe.transformer.config.sparse_target

        def forward():
            return pipe(
                prompt=[instruction], reference_image=[source], mask_image=mask if provided else None,
                height=args.resolution, width=args.resolution, num_inference_steps=args.num_inference_steps,
                inference_shift=args.inference_shift, guidance_scale=args.guidance_scale,
                reference_strength=args.reference_strength, negative_prompt=args.negative_prompt,
                generator=torch.Generator(device=pipe._execution_device).manual_seed(seed),
                edit_only_diffusion=provided, sparse_target=sparse, sparse_target_halo=args.sparse_target_halo,
                preserve_unmasked_pixels=args.preserve_unmasked_pixels,
                use_reference_kv_cache=not args.disable_reference_kv_cache,
                use_condition_kv_cache=False if args.disable_condition_kv_cache else None,
                reference_kv_cache_threshold=args.reference_kv_cache_threshold,
                condition_kv_cache_reuse_steps=args.num_inference_steps - 1 if args.cache_first_step else None,
                allow_early_condition_cache=args.cache_first_step,
                lora_part_enable=not args.native_editmgt,
            ).images[0].convert("RGB")

        if not self.warmed_up:
            for _ in range(args.warmup):
                forward()
            self.warmed_up = True
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        image = forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        info = {
            "seed": seed, "inference_seconds": elapsed,
            "execution": self.metadata,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "reference_cache": getattr(pipe, "last_reference_cache_info", None),
            "condition_cache": getattr(pipe, "last_condition_cache_info", None),
            "inference_schedule": getattr(pipe, "last_inference_schedule_info", None),
            "sparse_target": sparse, "supervise_target_halo": pipe.transformer.config.supervise_target_halo,
            "sparse_target_halo": args.sparse_target_halo if args.sparse_target_halo is not None else pipe.transformer.config.sparse_target_halo,
            "edit_region": getattr(pipe, "last_edit_region_info", None),
            "preserve_unmasked_pixels": bool(provided and args.preserve_unmasked_pixels),
            "output_compositing": (
                "original_pixel_mask_at_generation_and_native_resolution"
                if provided and args.preserve_unmasked_pixels else "none"
            ),
        }
        if provided:
            mask_pixels = np.asarray(mask, dtype=np.uint8) > 0
            info["input_mask"] = {
                "size": list(mask.size), "edit_pixels": int(mask_pixels.sum()),
                "edit_fraction": float(mask_pixels.mean()), "edit_bbox": mask.getbbox(),
                "native_size": list(native_mask.size), "native_edit_bbox": native_mask.getbbox(),
            }
        image = image.resize(native_size, Image.Resampling.BICUBIC)
        if provided and args.preserve_unmasked_pixels:
            # Resizing can spread generated colors across a boundary. Apply
            # the original annotation again at the exported/native resolution.
            image = Image.composite(image, native_source, native_mask)
        return image, info

    def close(self):
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_benchmark(args, run, resolver, *, output_dir=None, progress_desc="Generate"):
    # Sharded workers share image/record paths, but keep process metadata and
    # partial summaries separate until the parent publishes the global result.
    output_dir = Path(output_dir) if output_dir is not None else run.root
    runner = None
    regenerated = set()
    try:
        for task in tqdm(run.selection, desc=progress_desc):
            parent = task.get("input_prediction_key")
            previous = run.record(task["key"])
            try:
                # Validate mask-conditioned Inter-Edit cases even on resume,
                # before a missing/empty mask can reach model initialization.
                # Native mask-free generation still does not read annotations;
                # its invalid cases are excluded when regional scoring starts.
                checked_mask = None
                if run.info["benchmark"] == "inter_edit" and args.mask_mode == "provided":
                    checked_mask = inter_edit_benchmark_mask(args, run, resolver, task)
                if (previous and previous["status"] == "ok" and run.image_path(task).is_file()
                        and parent not in regenerated):
                    continue
                regenerated.add(task["key"])
                if parent:
                    upstream = run.record(parent)
                    if not upstream or upstream["status"] != "ok":
                        raise ValueError(f"Previous generated turn {parent} is unavailable; refusing a GT-source fallback")
                    with Image.open(run.image_path(run.by_key[parent])) as image:
                        source = image.convert("RGB")
                else:
                    source = resolver.image(task["source"], "RGB")
                # Annotation masks are not loaded during mask-free generation.
                # Inter-Edit scoring reads its original masks in the metrics
                # stage; they never become inputs to native EditMGT.
                mask = checked_mask
                if run.info["benchmark"] != "inter_edit":
                    mask = (resolver.edit_mask(task["mask"], run.info["benchmark"])
                            if args.mask_mode == "provided" and task.get("mask") else None)
                if runner is None:
                    runner = InferenceRunner(args)
                    atomic_json(output_dir / "execution.json", runner.metadata)
                image, info = runner.generate(source, mask, task["instruction"], sample_seed(args.seed, task["seed_key"]))
                if mask is not None and run.info["benchmark"] == "magicbrush":
                    info["edit_mask_decoding"] = MAGICBRUSH_MASK_DECODING
                save_png(run.image_path(task), image)
                if args.save_inputs:
                    directory = run.root / "inputs" / safe_name(task["key"])
                    save_png(directory / "source.png", source)
                    if task.get("target") is not None:
                        save_png(directory / "target.png", resolver.image(task["target"], "RGB"))
                    if mask is not None:
                        save_png(directory / "mask.png", normalize_edit_mask_image(mask))
                atomic_json(run.record_path(task["key"]), {"status": "ok", "key": task["key"], **info})
            except SkippedBenchmarkSample as error:
                record_skipped_benchmark_sample(run, task, error, "generate")
            except Exception as error:
                atomic_json(run.record_path(task["key"]), {"status": "error", "key": task["key"],
                                                          "error_type": type(error).__name__, "error": str(error)})
                if not args.continue_on_error:
                    raise
    finally:
        if runner is not None:
            runner.close()
        write_generation_summary(run, output_dir)


def write_generation_summary(run, output_dir=None):
    completed = []
    skipped = []
    for task in run.selection:
        record = run.record(task["key"])
        if record and record["status"] == "ok" and run.image_path(task).is_file():
            completed.append(record)
        elif record and record.get("status") == "skipped":
            skipped.append(record)
    expected = len(run.selection) - len(skipped)
    summary = {
        "selected": len(run.selection), "expected": expected, "completed": len(completed),
        "skipped": len(skipped), "skipped_samples": skipped,
        "skipped_by_reason": {reason: sum(record["reason"] == reason for record in skipped)
                              for reason in sorted({record["reason"] for record in skipped})},
        "complete": expected > 0 and len(completed) == expected,
        "full_selection_generated": len(completed) == len(run.selection), "timing": timing_summary(completed),
    }
    atomic_json(Path(output_dir or run.root) / "generation_summary.json", summary)
    return summary


def timing_summary(records):
    seconds = [record["inference_seconds"] for record in records if record.get("inference_seconds") is not None]
    peaks = [record["peak_allocated_bytes"] for record in records if record.get("peak_allocated_bytes") is not None]
    return {"count": len(seconds), "mean_seconds": float(np.mean(seconds)) if seconds else None,
            "p50_seconds": float(np.percentile(seconds, 50)) if seconds else None,
            "p95_seconds": float(np.percentile(seconds, 95)) if seconds else None,
            "max_peak_allocated_bytes": max(peaks) if peaks else None,
            "device_names": sorted({record["execution"]["device_name"] for record in records if record.get("execution")}),
            "scope": "synchronized pipeline call including encoders, VQ and decode; excludes warmup, image I/O and metric models"}


def validate_metric_inputs(args, benchmark, selection):
    from src.benchmark_metrics import parse_metrics

    names = parse_metrics(args.metrics, benchmark)
    if args.stage == "generate":
        return names
    required = {"lpips": "lpips", "alpha-in": "alpha_clip", "alpha-out": "alpha_clip",
                "clip-i": "clip", "clip-t": "clip", "boundary": "cv2"}
    if benchmark == "pie_bench":
        for name in names:
            if name.startswith("lpips"):
                required[name] = "lpips"
            elif name.startswith("ssim"):
                required[name] = "torchmetrics"
            elif name.startswith("clip_"):
                required[name] = "transformers"
    for package in {required[name] for name in names if name in required}:
        if importlib.util.find_spec(package) is None:
            raise ImportError(
                f"Metric dependency {package!r} is missing from the benchmark environment. "
                "Install metric dependencies into .venv with: bash scripts/setup_benchmark_env.sh"
            )
    if {"alpha-in", "alpha-out"} & set(names):
        if args.alpha_clip_ckpt is None or not args.alpha_clip_ckpt.is_file():
            raise FileNotFoundError("Provide --alpha-clip-ckpt, or explicitly select metrics without AlphaCLIP")
    if "clip-t" in names:
        missing = [task["key"] for task in selection if not task.get("caption")]
        if missing:
            raise ValueError(f"CLIP-T needs target global captions; missing {len(missing)} entries, first={missing[:3]}. Supply --caption-file or explicitly omit clip-t.")
    return names


def metric_summary(rows):
    values = defaultdict(list)
    for row in rows:
        for name, score in row["metrics"].items():
            values[name].append(score)
    return {name: {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "count": len(scores)}
            for name, scores in sorted(values.items())}


def score_benchmark(args, run, resolver, names, *, output_dir=None, model_lock=None,
                    progress_desc="Metrics", quiet=False):
    from src.benchmark_metrics import BenchmarkMetrics

    # Serialize model initialization/downloads across local GPU workers. The
    # actual scoring runs concurrently after releasing this cache lock.
    with ExitStack() as stack:
        if model_lock is not None:
            import fcntl

            lock = stack.enter_context(Path(model_lock).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        if run.info["benchmark"] == "pie_bench":
            from src.pie_bench_metrics import PIEBenchMetrics

            suite = PIEBenchMetrics(
                names, device=args.metric_device or args.device, model_cache_dir=args.model_cache_dir,
                dino_repo=args.dino_repo, dino_weights=args.dino_weights, metric_resolution=args.metric_resolution,
            )
        else:
            suite = BenchmarkMetrics(
                run.info["benchmark"], names, device=args.metric_device or args.device,
                alpha_clip_ckpt=args.alpha_clip_ckpt, alpha_clip_model=args.alpha_clip_model,
                model_cache_dir=args.model_cache_dir, dino_repo=args.dino_repo, dino_weights=args.dino_weights,
                metric_resolution=args.metric_resolution,
            )
    if getattr(args, "caption_file", None):
        suite.protocol["caption_file"] = file_identity(args.caption_file)
    if run.info["benchmark"] == "inter_edit":
        suite.protocol["invalid_mask_policy"] = "skip missing/empty masks; preserve original selection and report exclusions"
    metrics_dir = Path(output_dir) if output_dir is not None else run.root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(metrics_dir / "protocol.json", suite.protocol)
    rows, failures, records = [], [], []
    with (metrics_dir / "per_sample.jsonl").open("w", encoding="utf-8") as scores_file, \
            (metrics_dir / "errors.jsonl").open("w", encoding="utf-8") as errors_file:
        for task in tqdm(run.selection, desc=progress_desc):
            try:
                record = run.record(task["key"])
                if record and record.get("status") == "skipped":
                    continue
                checked_mask = (inter_edit_benchmark_mask(args, run, resolver, task)
                                if run.info["benchmark"] == "inter_edit" else None)
                if not record or record["status"] != "ok":
                    raise ValueError(f"No completed generation record: {record.get('error') if record else 'missing record'}")
                with Image.open(run.image_path(task)) as image:
                    prediction = image.convert("RGB")
                if run.info["benchmark"] == "pie_bench":
                    scores = suite.score(
                        prediction, source=resolver.image(task["source"], "RGB"),
                        mask=resolver.edit_mask(task["mask"], "pie_bench"),
                        source_caption=task["source_caption"], target_caption=task["caption"],
                    )
                else:
                    scores = suite.score(
                        prediction, resolver.image(task["target"], "RGB"),
                        source=resolver.image(task["source"], "RGB") if run.info["benchmark"] == "inter_edit" else None,
                        mask=checked_mask,
                        caption=task.get("caption"),
                    )
                row = {**task, "metrics": scores, "seed": record["seed"], "inference_seconds": record["inference_seconds"]}
                rows.append(row)
                records.append(record)
                scores_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                scores_file.flush()
            except SkippedBenchmarkSample as error:
                record_skipped_benchmark_sample(run, task, error, "metrics")
            except Exception as error:
                failure = {"key": task["key"], "error_type": type(error).__name__, "error": str(error)}
                failures.append(failure)
                errors_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                errors_file.flush()
                if not args.continue_on_error:
                    break
    summary = build_metric_summary(run, rows, failures, records, suite.protocol)
    atomic_json(metrics_dir / "summary.json", summary)
    if not quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["complete"]


def build_metric_summary(run, rows, failures, records, protocol):
    """Aggregate per-image scores, including shards with unequal sample counts."""
    skipped = skipped_benchmark_samples(run)
    expected = len(run.selection) - len(skipped)
    summary = {
        "benchmark": run.info["benchmark"], "dataset": run.info["settings"]["dataset"],
        "generation": run.info["settings"]["generation"], "metric_protocol": protocol,
        "selected_predictions": len(run.selection), "expected_predictions": expected, "scored_predictions": len(rows),
        "skipped_predictions": len(skipped), "skipped_samples": skipped,
        "skipped_by_reason": {reason: sum(record["reason"] == reason for record in skipped)
                              for reason in sorted({record["reason"] for record in skipped})},
        "failed_predictions": len(failures), "unprocessed_predictions": expected - len(rows) - len(failures),
        "complete": expected > 0 and len(rows) == expected,
        "full_selection_scored": len(rows) == len(run.selection), "timing": timing_summary(records),
    }
    if run.info["benchmark"] == "pie_bench":
        from src.pie_bench_metrics import pie_metric_summary

        summary["overall"] = pie_metric_summary(rows)
        groups = sorted({task["edit_type"] for task in run.selection})
        summary["by_edit_type"] = {
            group: {"expected": sum(task["edit_type"] == group for task in run.selection),
                    "scored": sum(row["edit_type"] == group for row in rows),
                    "metrics": pie_metric_summary([row for row in rows if row["edit_type"] == group])}
            for group in groups
        }
    elif run.info["benchmark"] == "inter_edit":
        summary["overall"] = metric_summary(rows)
        for field in ("edit_type", "language"):
            groups = sorted({row.get(field, "unknown") for row in rows})
            summary["by_" + field] = {group: metric_summary([row for row in rows if row.get(field, "unknown") == group])
                                      for group in groups}
    else:
        groups = sorted({group for task in run.selection for group in task["groups"]})
        summary["by_protocol"] = {
            group: {"expected": sum(group in task["groups"] for task in run.selection),
                    "scored": sum(group in row["groups"] for row in rows),
                    "metrics": metric_summary([row for row in rows if group in row["groups"]])}
            for group in groups
        }
    return summary
