#!/usr/bin/env python
"""Zero-shot smoke test for mask-region-aware inference with the released checkpoint."""
import argparse
import faulthandler
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.editmgt import init_edit_mgt  # noqa: E402
from src.v2_model import negative_prompt  # noqa: E402
from src.inter_edit_dataset import DEFAULT_INTER_EDIT_ROOT, InterEditReader  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the original EditMGT checkpoint zero-shot on held-out masked edit cases."
    )
    parser.add_argument("--checkpoint", default="checkpoint/EditMGT")
    parser.add_argument("--data-dir", default=DEFAULT_INTER_EDIT_ROOT)
    parser.add_argument("--split-manifest", default=None, help="Required only for the legacy parquet dataset.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/zero_shot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=18)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--sdpa-backend",
        choices=("math", "efficient", "auto"),
        default="math",
        help="Attention backend. 'math' is the safest smoke-test setting; 'auto' enables fused kernels.",
    )
    return parser.parse_args()


def decode_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        value = value.get("bytes") if value.get("bytes") is not None else value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    if isinstance(value, str):
        return Image.open(value)
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def iter_test_samples(data_dir: Path, manifest_path: Path, sample_index: int, num_cases: int):
    if sample_index < 0:
        raise ValueError("--sample-index must be non-negative")
    if num_cases < 1:
        raise ValueError("--num-cases must be positive")
    data_dir = Path(data_dir)
    if (data_dir / "test" / "metadata.json").is_file() or (data_dir / "metadata.json").is_file():
        yield from InterEditReader(data_dir, "test").iter_samples(sample_index, num_cases)
        return
    if manifest_path is None:
        raise ValueError("A parquet dataset requires --split-manifest")
    manifest = json.loads(manifest_path.read_text())
    current_index = 0
    yielded = 0
    for filename in manifest["test"]:
        table = pq.read_table(data_dir / filename)
        file_end = current_index + table.num_rows
        if sample_index < file_end and yielded < num_cases:
            first_row = max(0, sample_index - current_index)
            rows_to_take = min(table.num_rows - first_row, num_cases - yielded)
            for offset, row in enumerate(table.slice(first_row, rows_to_take).to_pylist()):
                row_index = first_row + offset
                yield current_index + row_index, filename, row_index, row
                yielded += 1
        if yielded == num_cases:
            return
        current_index += table.num_rows
    if yielded == 0:
        raise IndexError(f"sample-index {sample_index} exceeds the {current_index} held-out samples")
    raise IndexError(f"Requested {num_cases} cases but only found {yielded} from sample-index {sample_index}")


def get_first(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"None of the required fields are present: {keys}")


def configure_torch_runtime(args):
    faulthandler.enable()
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} was requested but CUDA is unavailable")
    device = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(device)
    if args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "--bf16 was requested but this GPU does not report native bfloat16 support; rerun without --bf16"
        )

    # H20 reports native bf16 support, but torch 2.1.0+cu121 has been observed
    # to terminate with SIGFPE in AdaLayerNormZero's first bf16 Linear GEMM.
    # A native SIGFPE cannot be caught and retried, so downgrade before loading
    # the checkpoint. Newer torch releases are left untouched.
    torch_major_minor = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if args.bf16 and "H20" in gpu_name and torch_major_minor <= (2, 1):
        print(
            "WARNING: disabling bf16 for NVIDIA H20 with torch<=2.1 because this "
            "combination crashes in torch.nn.Linear (SIGFPE). Using float32.",
            flush=True,
        )
        args.bf16 = False

    if args.sdpa_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif args.sdpa_backend == "efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(
        "Runtime: "
        f"torch={torch.__version__}, cuda={torch.version.cuda}, "
        f"gpu={gpu_name}, "
        f"bf16={args.bf16}, sdpa_backend={args.sdpa_backend}",
        flush=True,
    )


def main():
    args = parse_args()
    configure_torch_runtime(args)
    # Only the released editing checkpoint is loaded: no fine-tuned transformer
    # and no LoRA adapter are involved in this zero-shot baseline.
    pipe = init_edit_mgt(
        device=args.device,
        enable_bf16=args.bf16,
        base_model_path=args.checkpoint,
        transformer_subfolder="editmgt",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = []
    samples = iter_test_samples(
        Path(args.data_dir), Path(args.split_manifest) if args.split_manifest else None, args.sample_index, args.num_cases
    )
    for global_index, filename, row_index, row in samples:
        instruction = get_first(row, ("instruction", "edit_instruction"))
        source = decode_image(get_first(row, ("input_img", "input", "image_file"))).convert("RGB")
        target = decode_image(get_first(row, ("output_img", "output", "edited_file"))).convert("RGB")
        edit_mask = decode_image(get_first(row, ("mask_png", "edit_mask", "mask"))).convert("L")
        model_size = (args.resolution, args.resolution)
        source_model = source.resize(model_size, Image.Resampling.BILINEAR)
        target_model = target.resize(model_size, Image.Resampling.BILINEAR)
        mask_model = edit_mask.resize(model_size, Image.Resampling.NEAREST)

        case_seed = args.seed + global_index
        generator = torch.Generator(device=args.device).manual_seed(case_seed)
        result = pipe(
            prompt=[instruction],
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=negative_prompt or None,
            generator=generator,
            reference_image=[source_model],
            reference_strength=args.reference_strength,
            mask_image=mask_model,
            edit_only_diffusion=True,
            return_attention=False,
        )

        case_id = f"sample_{global_index:06d}_{Path(filename).stem}_row_{row_index}"
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        source_model.save(case_dir / "source.png")
        target_model.save(case_dir / "target.png")
        mask_model.save(case_dir / "gt_mask.png")
        result.images[0].save(case_dir / "generated.png")
        metadata = {
            "case_id": case_id,
            "sample_index": global_index,
            "dataset_file": filename,
            "row_index": row_index,
            "instruction": instruction,
            "checkpoint": args.checkpoint,
            "transformer_subfolder": "editmgt",
            "zero_shot": True,
            "edit_only_diffusion": True,
            "mask_convention": "1/white=edit, 0/black=preserve",
            "seed": case_seed,
            "resolution": args.resolution,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "reference_strength": args.reference_strength,
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        run_metadata.append(metadata)
        print(f"[{len(run_metadata)}/{args.num_cases}] Zero-shot inference completed: {case_dir}")

    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
