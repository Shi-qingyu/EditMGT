"""Generate full benchmark predictions with third-party editing models.

Selection and metric scoring reuse the same manifests and scorer as the local
EditMGT runs. This script only handles inference and synchronized timing.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark_data import BenchmarkImages  # noqa: E402
from src.benchmark_utils import (  # noqa: E402
    atomic_json, safe_name, sample_seed, save_png, write_generation_summary,
)


REFERENCE = ROOT / "outputs/dimo_step4000_sft8000_comparison/vanilla36"
REFERENCE_NAME = {
    "magicbrush": "magicbrush_test",
    "inter_edit": "inter_edit",
    "pie_bench": "pie_bench",
}
OUTPUT = ROOT / "outputs/real_time_baselines"
MODEL_CONFIG = {
    "dreamlite_mobile": {
        "model_repo": "third_party/DreamLite",
        "model_weights": "carlofkl/DreamLite-mobile@main",
        "num_inference_steps": 4,
        "resolution": 1024,
        "prompt_protocol": "edit instruction",
    },
    "chordedit": {
        "model_repo": "third_party/ChordEdit",
        "model_weights": "stabilityai/sd-turbo@main",
        "num_inference_steps": 1,
        "resolution": 512,
        "prompt_protocol": "PIE source/target captions; MagicBrush/Inter-Edit empty source plus edit instruction",
    },
    "flux2_klein": {
        "model_repo": "third_party/SpotEdit (official FLUX.2 model via Diffusers)",
        "model_weights": "black-forest-labs/FLUX.2-klein-4B@main",
        "num_inference_steps": 4,
        "resolution": 1024,
        "prompt_protocol": "edit instruction",
    },
    "turboedit": {
        "model_repo": "third_party/turbo-edit",
        "model_weights": "stabilityai/sdxl-turbo@main (fp16)",
        "num_inference_steps": 4,
        "resolution": 512,
        "prompt_protocol": "PIE source/target captions; MagicBrush/Inter-Edit empty source plus edit instruction",
    },
    "swiftedit": {
        "model_repo": "third_party/SwiftEdit",
        "model_weights": "SwiftEdit v1.0 release, stabilityai/sd-turbo, SD2.1-base, h94/IP-Adapter",
        "num_inference_steps": 1,
        "resolution": 512,
        "prompt_protocol": "PIE source/target captions; MagicBrush/Inter-Edit empty source plus edit instruction",
    },
    "sparse_lavida": {
        "model_repo": "third_party/Sparse-LaViDa",
        "model_weights": "jacklishufan/sparse-lavida@main, google/siglip-so400m-patch14-384@main, MeissonFlow/Meissonic VQ-VAE@main",
        "num_inference_steps": 64,
        "resolution": 1024,
        "prompt_protocol": "edit instruction, official sparse cached decoding",
    },
    "spotedit_flux2_4b": {
        "model_repo": "third_party/SpotEdit",
        "model_weights": "black-forest-labs/FLUX.2-klein-4B@main",
        "num_inference_steps": 4,
        "resolution": 1024,
        "prompt_protocol": "edit instruction; SpotEdit adapted to four-step Klein 4B with one warm-up step",
    },
    "ultraedit": {
        "model_repo": "third_party/UltraEdit",
        "model_weights": "BleachNick/SD3_UltraEdit_w_mask@e5beecb69a3ef1399718be2f99b61730b8bf0346",
        "num_inference_steps": 50,
        "resolution": 512,
        "prompt_protocol": "edit instruction plus benchmark annotation mask (white=edit)",
    },
    "lumina_dimoo": {
        "model_repo": "third_party/Lumina-DiMOO",
        "model_weights": "Alpha-VLLM/Lumina-DiMOO@main",
        "num_inference_steps": 64,
        "resolution": 512,
        "prompt_protocol": "official image editing template with original instruction",
    },
    "mmada_parallel_a": {
        "model_repo": "third_party/MMaDA-Parallel/MMaDA-Parallel-A",
        "model_weights": "tyfeld/MMaDA-Parallel-A@main",
        "num_inference_steps": 64,
        "resolution": 512,
        "prompt_protocol": "official TI2TI editing template with original instruction; 128 text steps",
    },
    "nep_512": {
        "model_repo": "third_party/NEP",
        "model_weights": "hmwhwm/NEP/NEP_512.pt@main, peizesun/llamagen_t2i/vq_ds16_t2i.pt, google/flan-t5-xl",
        "num_inference_steps": "autoregressive, variable edit-token count",
        "resolution": 512,
        "prompt_protocol": "edit instruction and benchmark edit-region mask; official NEP generator",
    },
    "dice_paella": {
        "model_repo": "third_party/DICE/image_editing/paella",
        "model_weights": "dome272/Paella@main, google/byt5-xl@main, laion/CLIP-ViT-H-14-laion2B-s32B-b79K@main",
        "num_inference_steps": "32 inversion + 32 editing",
        "resolution": 512,
        "prompt_protocol": "PIE-Bench official source and target captions; DICE CFG 10, lambda 0.7, tau 0.9",
    },
    "gide_pie": {
        "model_repo": "third_party/GIDE with SAM 2 and SAM 3",
        "model_weights": "Alpha-VLLM/Lumina-DiMOO@main, facebook/sam2-hiera-large@main, facebook/sam3@main",
        "num_inference_steps": "64 inversion + 64 editing/refinement",
        "resolution": 512,
        "prompt_protocol": "PIE-Bench source/target captions plus paired subject words, local GIDE adaptation",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_CONFIG), required=True)
    parser.add_argument("--benchmark", choices=tuple(REFERENCE_NAME), required=True)
    parser.add_argument("--stage", choices=["prepare", "generate", "finalize", "annotate"], required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="Initial N tasks for a smoke run; use a separate output directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.world_size < 1 or not (0 <= args.rank < args.world_size):
        parser.error("Require 0 <= rank < world-size and world-size >= 1")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.output_dir is None:
        suffix = f"_smoke{args.limit}" if args.limit else ""
        args.output_dir = OUTPUT / args.model / f"{args.benchmark}{suffix}"
    return args


def selected_reference(args):
    reference = REFERENCE / REFERENCE_NAME[args.benchmark]
    selection = json.loads((reference / "selection.json").read_text(encoding="utf-8"))
    if args.limit:
        selection = selection[:args.limit]
    info = json.loads((reference / "run.json").read_text(encoding="utf-8"))
    return selection, info


def prepare(args):
    selection, reference = selected_reference(args)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    generation = {
        "model_protocol": args.model,
        **MODEL_CONFIG[args.model],
        "mask_mode": "none",
        "seed": args.seed,
        "warmup": args.warmup,
        "timing_scope": "synchronized model pipeline; excludes image I/O, warmup, and metrics",
    }
    info = {
        "version": 1, "benchmark": reference["benchmark"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "benchmark": reference["benchmark"],
            "dataset": reference["settings"]["dataset"],
            "generation": generation,
            "selection_sha256": hashlib.sha256(
                json.dumps(selection, sort_keys=True).encode()
            ).hexdigest(),
        },
    }
    run_path = root / "run.json"
    if run_path.is_file():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        if not args.resume or existing["settings"] != info["settings"]:
            raise ValueError("Existing run differs; use --resume or a new output directory")
    else:
        atomic_json(root / "selection.json", selection)
        atomic_json(run_path, info)
    print(f"Prepared {len(selection)} {args.benchmark} tasks in {root}", flush=True)


def assigned(task, benchmark, rank, world_size):
    key = task["session_id"] if benchmark == "magicbrush" else task["key"]
    shard = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % world_size
    return shard == rank


def load_backend(name):
    if name == "dreamlite_mobile":
        from backends.dreamlite_mobile import Backend
    elif name == "chordedit":
        from backends.chordedit import Backend
    elif name == "flux2_klein":
        from backends.flux2_klein import Backend
    elif name == "turboedit":
        from backends.turboedit import Backend
    elif name == "swiftedit":
        from backends.swiftedit import Backend
    elif name == "sparse_lavida":
        from backends.sparse_lavida import Backend
    elif name == "spotedit_flux2_4b":
        from backends.spotedit_flux2_4b import Backend
    elif name == "ultraedit":
        from backends.ultraedit import Backend
    elif name == "lumina_dimoo":
        from backends.lumina_dimoo import Backend
    elif name == "mmada_parallel_a":
        from backends.mmada_parallel_a import Backend
    elif name == "nep_512":
        from backends.nep_512 import Backend
    elif name == "dice_paella":
        from backends.dice_paella import Backend
    elif name == "gide_pie":
        from backends.gide_pie import Backend
    else:
        raise ValueError(name)
    return Backend()


def generation(args):
    root = args.output_dir.resolve()
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    by_key = {task["key"]: task for task in selection}
    tasks = [task for task in selection if assigned(task, args.benchmark, args.rank, args.world_size)]
    resolver = BenchmarkImages(ROOT / "data/inter_edit_tar_index")
    backend = None
    warmed = False
    device = torch.device("cuda")
    try:
        for index, task in enumerate(tasks, 1):
            image_path = root / task["prediction"]
            record_path = root / "records" / f"{safe_name(task['key'])}.json"
            parent = task.get("input_prediction_key")
            if args.resume and image_path.is_file() and record_path.is_file():
                old = json.loads(record_path.read_text(encoding="utf-8"))
                if old.get("status") == "ok":
                    continue
            try:
                if parent:
                    upstream = by_key[parent]
                    with Image.open(root / upstream["prediction"]) as image:
                        source = image.convert("RGB")
                else:
                    source = resolver.image(task["source"], "RGB")
                native_size = source.size
                if backend is None:
                    backend = load_backend(args.model)
                    atomic_json(root / f"execution_rank{args.rank}.json", {
                        "model": args.model, "device_name": torch.cuda.get_device_name(device),
                        "torch_version": torch.__version__, "rank": args.rank,
                        "world_size": args.world_size,
                    })
                model_source = backend.prepare_source(source)
                if getattr(backend, "requires_edit_mask", False):
                    from src.mask_utils import resize_binary_edit_mask
                    edit_mask = resolver.edit_mask(task["mask"], args.benchmark)
                    edit_mask = resize_binary_edit_mask(edit_mask, (model_source.height, model_source.width))
                    task = {**task, "_edit_mask": edit_mask.convert("RGB")}
                seed = sample_seed(args.seed, task["seed_key"])
                if not warmed:
                    for _ in range(args.warmup):
                        backend.generate(model_source, task, seed)
                    warmed = True
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = backend.generate(model_source, task, seed)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                result = result.resize(native_size, Image.Resampling.BICUBIC)
                save_png(image_path, result)
                atomic_json(record_path, {
                    "status": "ok", "key": task["key"], "seed": seed,
                    "inference_seconds": elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "execution": {"device_name": torch.cuda.get_device_name(device)},
                })
            except Exception as error:
                atomic_json(record_path, {
                    "status": "error", "key": task["key"],
                    "error_type": type(error).__name__, "error": str(error),
                })
                raise
            if index % 25 == 0 or index == len(tasks):
                print(f"rank {args.rank}: {index}/{len(tasks)} {args.benchmark}", flush=True)
    finally:
        resolver.close()


def finalize(args):
    root = args.output_dir.resolve()
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))

    class Run:
        def __init__(self):
            self.root = root
            self.selection = selection

        def record(self, key):
            path = self.root / "records" / f"{safe_name(key)}.json"
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

        def image_path(self, task):
            return self.root / task["prediction"]

    result = write_generation_summary(Run())
    result["timing"]["scope"] = (
        "synchronized full model pipeline including text encoding and image decode; "
        "excludes source resize, output resize, warmup, image I/O, and metrics"
    )
    atomic_json(root / "generation_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["complete"]:
        raise SystemExit(1)


def main():
    args = parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "generate":
        generation(args)
    elif args.stage == "finalize":
        finalize(args)
    else:
        path = args.output_dir.resolve() / "metrics/summary.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["complete"], path
        result["timing"]["scope"] = (
            "synchronized full model pipeline including text encoding and image decode; "
            "excludes source resize, output resize, warmup, image I/O, and metrics"
        )
        atomic_json(path, result)


if __name__ == "__main__":
    main()
