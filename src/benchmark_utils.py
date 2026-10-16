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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.mask_schedule import validate_inference_shift
from src.mask_utils import normalize_edit_mask_image, resize_binary_edit_mask


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


def add_benchmark_arguments(parser, benchmark):
    from src.benchmark_metrics import DEFAULT_METRICS

    model = parser.add_mutually_exclusive_group()
    model.add_argument("--checkpoint", type=Path, help="Our full or LoRA checkpoint; required for generation unless --vanilla.")
    model.add_argument("--vanilla", action="store_true", help="Use the released EditMGT Transformer.")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoint/EditMGT"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / f"benchmark_{benchmark}")
    parser.add_argument("--stage", choices=["all", "generate", "metrics"], default="all")
    parser.add_argument("--resume", action="store_true", help="Reuse completed images only when generation config/selection match.")
    parser.add_argument("--continue-on-error", action="store_true", help="Write failures and continue; incomplete benchmarks exit nonzero.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-device", default=None, help="Defaults to --device; scoring runs after EditMGT is released.")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=18)
    parser.add_argument("--inference-shift", type=validate_inference_shift, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--reference-kv-cache-threshold", type=float, default=0.3)
    parser.add_argument("--disable-reference-kv-cache", action="store_true")
    parser.add_argument("--disable-condition-kv-cache", action="store_true")
    parser.add_argument("--sparse-target-halo", type=int, default=None)
    parser.add_argument("--mask-mode", choices=["provided", "none"], default="provided")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup generations before each generation process.")
    parser.add_argument("--limit", type=int, default=None, help="Optional first N samples (Inter-Edit) or WHOLE sessions (MagicBrush).")
    parser.add_argument("--save-inputs", action="store_true", help="Also save source, GT and original mask; uses additional disk space.")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS[benchmark]))
    parser.add_argument("--metric-resolution", type=int, default=None, help="Optional square scoring size; default follows native benchmark sizes.")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--alpha-clip-ckpt", type=Path, default=None)
    parser.add_argument("--alpha-clip-model", default="ViT-L/14")
    parser.add_argument("--dino-repo", type=Path, default=None, help="Optional local facebookresearch/dino clone for offline use.")
    parser.add_argument("--dino-weights", type=Path, default=None)


def validate_arguments(parser, args):
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
    if args.metric_resolution is not None and args.metric_resolution < 1:
        parser.error("--metric-resolution must be positive")
    if not all(math.isfinite(value) for value in (args.guidance_scale, args.reference_strength)):
        parser.error("Guidance and reference strength must be finite")


def generation_config(args):
    keys = ("model_path", "vanilla", "device", "resolution", "num_inference_steps", "inference_shift",
            "guidance_scale", "reference_strength", "reference_kv_cache_threshold",
            "disable_reference_kv_cache", "disable_condition_kv_cache", "sparse_target_halo",
            "mask_mode", "bf16", "seed", "warmup", "save_inputs")
    config = {key: getattr(args, key) for key in keys}
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
                settings = {"benchmark": benchmark, "dataset": dataset, "generation": generation_config(args),
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
        self._lock.close()

    def record_path(self, key):
        return self.root / "records" / (safe_name(key) + ".json")

    def record(self, key):
        path = self.record_path(key)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def image_path(self, task):
        return self.root / task["prediction"]


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
                                  transformer_path=str(transformer_path) if transformer_path else None)
        if checkpoint and transformer_path is None:
            load_lora_adapter(self.pipe.transformer, str(checkpoint))
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
            "supervise_target_halo": self.pipe.transformer.config.supervise_target_halo,
            "warmup_generations_per_process": args.warmup,
        }

    @torch.inference_mode()
    def generate(self, source, mask, instruction, seed):
        from src.v2_model import negative_prompt

        args, pipe = self.args, self.pipe
        native_size = source.size
        source = source.convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
        provided = args.mask_mode == "provided"
        if provided:
            if mask is None:
                raise ValueError("Mask-provided evaluation requires an annotation; no mask is synthesized")
            mask = resize_binary_edit_mask(normalize_edit_mask_image(mask), (args.resolution, args.resolution))
            if mask.getbbox() is None:
                raise ValueError("Empty benchmark mask")
        sparse = provided and pipe.transformer.config.sparse_target

        def forward():
            return pipe(
                prompt=[instruction], reference_image=[source], mask_image=mask if provided else None,
                height=args.resolution, width=args.resolution, num_inference_steps=args.num_inference_steps,
                inference_shift=args.inference_shift, guidance_scale=args.guidance_scale,
                reference_strength=args.reference_strength, negative_prompt=negative_prompt or None,
                generator=torch.Generator(device=pipe._execution_device).manual_seed(seed),
                edit_only_diffusion=provided, sparse_target=sparse, sparse_target_halo=args.sparse_target_halo,
                use_reference_kv_cache=not args.disable_reference_kv_cache,
                use_condition_kv_cache=False if args.disable_condition_kv_cache else None,
                reference_kv_cache_threshold=args.reference_kv_cache_threshold, lora_part_enable=True,
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
        }
        return image.resize(native_size, Image.Resampling.BICUBIC), info

    def close(self):
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_benchmark(args, run, resolver):
    runner = None
    regenerated = set()
    try:
        for task in tqdm(run.selection, desc="Generate"):
            parent = task.get("input_prediction_key")
            previous = run.record(task["key"])
            if (previous and previous["status"] == "ok" and run.image_path(task).is_file()
                    and parent not in regenerated):
                continue
            regenerated.add(task["key"])
            try:
                if parent:
                    upstream = run.record(parent)
                    if not upstream or upstream["status"] != "ok":
                        raise ValueError(f"Previous generated turn {parent} is unavailable; refusing a GT-source fallback")
                    with Image.open(run.image_path(run.by_key[parent])) as image:
                        source = image.convert("RGB")
                else:
                    source = resolver.image(task["source"], "RGB")
                mask = resolver.image(task["mask"], "L") if task.get("mask") else None
                if runner is None:
                    runner = InferenceRunner(args)
                    atomic_json(run.root / "execution.json", runner.metadata)
                image, info = runner.generate(source, mask, task["instruction"], sample_seed(args.seed, task["seed_key"]))
                save_png(run.image_path(task), image)
                if args.save_inputs:
                    directory = run.root / "inputs" / safe_name(task["key"])
                    save_png(directory / "source.png", source)
                    save_png(directory / "target.png", resolver.image(task["target"], "RGB"))
                    if mask is not None:
                        save_png(directory / "mask.png", normalize_edit_mask_image(mask))
                atomic_json(run.record_path(task["key"]), {"status": "ok", "key": task["key"], **info})
            except Exception as error:
                atomic_json(run.record_path(task["key"]), {"status": "error", "key": task["key"],
                                                          "error_type": type(error).__name__, "error": str(error)})
                if not args.continue_on_error:
                    raise
    finally:
        if runner is not None:
            runner.close()
        records = [run.record(task["key"]) for task in run.selection]
        completed = [record for record in records if record and record["status"] == "ok"]
        atomic_json(run.root / "generation_summary.json", {
            "expected": len(records), "completed": len(completed), "complete": len(completed) == len(records),
            "timing": timing_summary(completed),
        })


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
    for package in {required[name] for name in names if name in required}:
        if importlib.util.find_spec(package) is None:
            raise ImportError(f"Metric dependency {package!r} is missing; see requirements-benchmark.txt / README")
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


def score_benchmark(args, run, resolver, names):
    from src.benchmark_metrics import BenchmarkMetrics

    suite = BenchmarkMetrics(
        run.info["benchmark"], names, device=args.metric_device or args.device,
        alpha_clip_ckpt=args.alpha_clip_ckpt, alpha_clip_model=args.alpha_clip_model,
        model_cache_dir=args.model_cache_dir, dino_repo=args.dino_repo, dino_weights=args.dino_weights,
        metric_resolution=args.metric_resolution,
    )
    if getattr(args, "caption_file", None):
        suite.protocol["caption_file"] = file_identity(args.caption_file)
    metrics_dir = run.root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows, failures, records = [], [], []
    with (metrics_dir / "per_sample.jsonl").open("w", encoding="utf-8") as scores_file, \
            (metrics_dir / "errors.jsonl").open("w", encoding="utf-8") as errors_file:
        for task in tqdm(run.selection, desc="Metrics"):
            try:
                record = run.record(task["key"])
                if not record or record["status"] != "ok":
                    raise ValueError(f"No completed generation record: {record.get('error') if record else 'missing record'}")
                with Image.open(run.image_path(task)) as image:
                    prediction = image.convert("RGB")
                scores = suite.score(
                    prediction, resolver.image(task["target"], "RGB"),
                    source=resolver.image(task["source"], "RGB") if run.info["benchmark"] == "inter_edit" else None,
                    mask=resolver.image(task["mask"], "L") if run.info["benchmark"] == "inter_edit" else None,
                    caption=task.get("caption"),
                )
                row = {**task, "metrics": scores, "seed": record["seed"], "inference_seconds": record["inference_seconds"]}
                rows.append(row)
                records.append(record)
                scores_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                scores_file.flush()
            except Exception as error:
                failure = {"key": task["key"], "error_type": type(error).__name__, "error": str(error)}
                failures.append(failure)
                errors_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                errors_file.flush()
                if not args.continue_on_error:
                    break
    summary = {
        "benchmark": run.info["benchmark"], "dataset": run.info["settings"]["dataset"],
        "generation": run.info["settings"]["generation"], "metric_protocol": suite.protocol,
        "expected_predictions": len(run.selection), "scored_predictions": len(rows),
        "failed_predictions": len(failures), "unprocessed_predictions": len(run.selection) - len(rows) - len(failures),
        "complete": len(rows) == len(run.selection), "timing": timing_summary(records),
    }
    if run.info["benchmark"] == "inter_edit":
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
    atomic_json(metrics_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["complete"]
