"""Visualize a manually selected EditMGT checkpoint on Inter-Edit training examples."""
import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-7000",
        help="Full checkpoint-N directory or Transformer directory; default: the maskfix checkpoint-7000.",
    )
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "checkpoint/EditMGT"),
                        help="Base model supplying the frozen text encoders, VQ model and scheduler.")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/mnt/geminihzceph/public/data/Inter-Edit"),
                        help="Inter-Edit root or its train/ directory; only the train split is read.")
    parser.add_argument("--tar-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: outputs/inter_edit_train/<checkpoint name>.")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-indices", nargs="+", type=int,
                        help="Zero-based indices in original train metadata order, before language filtering. "
                             "Preserves the given order and overrides --num-samples and --selection.")
    parser.add_argument("--language", choices=("en", "zh", "all"), default="en",
                        help="Filter original instructions without translation; default: en.")
    parser.add_argument("--selection", choices=("first", "random"), default="first",
                        help="First eligible examples, or shuffle metadata shards and rows with --selection-seed.")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42,
                        help="Generation seed for each case is seed + sample_id.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 instead of FP32.")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=18)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--sparse-target-halo", type=int, default=0,
                        help="Transformer-cell halo; default 0 disables dilation even for older checkpoints.")
    parser.add_argument("--preserve-unmasked-pixels", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt into copying source RGB outside the mask after decoding; default preserves only source tokens.")
    parser.add_argument("--disable-reference-kv-cache", action="store_true")
    parser.add_argument("--disable-condition-kv-cache", action="store_true")
    parser.add_argument("--reference-kv-cache-threshold", type=float, default=0.8)
    parser.add_argument("--condition-cache-threshold", type=float, default=0.8,
                        help="Runtime text/reference prefix phase; overrides the checkpoint's saved threshold.")
    parser.add_argument("--cache-first-step", action="store_true",
                        help="Ablation: compute text/reference KV on the first forward and reuse it on all "
                             "remaining forwards, including before the configured prefix phase.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Export selected inputs and train_inputs.png without loading model weights.")
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.resolution < 32 or args.resolution % 32:
        parser.error("--resolution must be a positive multiple of 32")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    if args.sparse_target_halo < 0:
        parser.error("--sparse-target-halo must be nonnegative")
    if not 0 <= args.reference_kv_cache_threshold <= 1:
        parser.error("--reference-kv-cache-threshold must be between 0 and 1")
    if not 0 < args.condition_cache_threshold <= 1:
        parser.error("--condition-cache-threshold must be in (0, 1]")
    if args.cache_first_step and (args.num_inference_steps < 2
                                 or args.disable_reference_kv_cache or args.disable_condition_kv_cache):
        parser.error("--cache-first-step requires at least two inference steps and enabled condition/reference caching")
    return args


def resolve_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    transformer = checkpoint / "transformer" if (checkpoint / "transformer/config.json").is_file() else checkpoint
    if not (transformer / "config.json").is_file():
        raise FileNotFoundError(
            f"No Transformer config under {checkpoint}. Pass checkpoint-N or its transformer/ directory."
        )
    run_checkpoint = transformer.parent if transformer.name == "transformer" else checkpoint
    label = run_checkpoint.name.removeprefix("checkpoint-")
    step = int(label) if label.isdigit() else 0
    return transformer, run_checkpoint.name, step


def iter_train_rows(reader, sample_indices=None, selection="first", selection_seed=42):
    """Select metadata before opening image archives, keeping official global indices."""
    if reader.split != "train":
        raise ValueError("This inference entry requires the Inter-Edit train split")
    if sample_indices is not None:
        indices = list(sample_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("--sample-indices must be nonempty and unique")
        if any(index < 0 or index >= reader.num_samples for index in indices):
            raise IndexError(f"Train sample indices must be in [0, {reader.num_samples})")
        shards = {}
        for index in indices:
            shard_index, row_index = divmod(index, reader.samples_per_shard)
            path = reader.metadata_files[shard_index]
            if shard_index not in shards:
                shards[shard_index] = reader.read_metadata(path)
            yield index, path.name, row_index, shards[shard_index][row_index]
        return
    rng = random.Random(selection_seed)
    shards = list(enumerate(reader.metadata_files))
    if selection == "random":
        rng.shuffle(shards)
    for shard_index, path in shards:
        rows = reader.read_metadata(path)
        row_indices = list(range(len(rows)))
        if selection == "random":
            rng.shuffle(row_indices)
        for row_index in row_indices:
            yield shard_index * reader.samples_per_shard + row_index, path.name, row_index, rows[row_index]


def prepare_cases(args, image_resample=None):
    from PIL import Image

    from src.inter_edit_dataset import (
        InterEditReader, TarAssetReader, inter_edit_prompt_language, matches_inter_edit_language,
    )
    from src.mask_utils import normalize_edit_mask_image, resize_binary_edit_mask
    from src.train_validation import _stack_panels, comparison_panel, mask_overlay

    image_resample = Image.Resampling.BILINEAR if image_resample is None else Image.Resampling(image_resample)
    reader = InterEditReader(args.data_dir.expanduser(), "train", args.tar_cache_dir)
    cases, manifest, skipped_empty = [], [], []
    count = len(args.sample_indices) if args.sample_indices is not None else args.num_samples
    if count > reader.num_samples:
        raise ValueError(f"Requested {count} examples from a train split with {reader.num_samples} rows")
    with TarAssetReader(reader.root, args.tar_cache_dir) as assets:
        for index, shard, row_index, row in iter_train_rows(
            reader, args.sample_indices, args.selection, args.selection_seed,
        ):
            if not matches_inter_edit_language(row, args.language):
                if args.sample_indices is not None:
                    raise ValueError(f"Train index {index} does not match --language {args.language}; "
                                     "use --language all to keep the original instruction")
                continue
            raw = reader.read_sample(row, assets)
            size = (args.resolution, args.resolution)
            source = raw["input"].convert("RGB").resize(size, image_resample)
            target = raw["output"].convert("RGB").resize(size, image_resample)
            mask = resize_binary_edit_mask(normalize_edit_mask_image(raw["edit_mask"]), size)
            if mask.getbbox() is None:
                if args.sample_indices is not None:
                    raise ValueError(f"Train index {index} has an empty edit mask")
                skipped_empty.append(index)
                continue
            overlay = mask_overlay(source, mask)
            case_dir = args.output_dir / "inputs" / f"sample_{raw['sample_id']:05d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            for name, image in (("source", source), ("target", target), ("mask", mask), ("mask_overlay", overlay)):
                image.save(case_dir / f"{name}.png")
            metadata = {
                **row, "split": "train", "train_index": index, "metadata_shard": shard, "row_index": row_index,
                "language": inter_edit_prompt_language(row), "language_filter": args.language,
                "resolution": args.resolution, "input_dir": str(case_dir.relative_to(args.output_dir)),
                "image_resampling": image_resample.name.lower(),
                "source_original_size": list(raw["input"].size),
                "target_original_size": list(raw["output"].size),
                "mask_original_size": list(raw["edit_mask"].size),
                "edit_area_ratio": mask.histogram()[255] / (mask.width * mask.height),
            }
            cases.append({**metadata, "source": source, "target": target, "mask": mask, "overlay": overlay})
            manifest.append(metadata)
            print(f"Train index {index} | sample_id={raw['sample_id']} | {raw['instruction']}", flush=True)
            if len(cases) == count:
                break
    if len(cases) != count:
        raise ValueError(f"Found only {len(cases)}/{count} nonempty train examples for --language {args.language}")
    selection = {
        "split": "train", "data_root": str(reader.root.resolve()), "language_filter": args.language,
        "selection": "indices" if args.sample_indices is not None else args.selection,
        "selection_seed": args.selection_seed, "total_train_samples": reader.num_samples,
        "selected_train_indices": [case["train_index"] for case in cases], "skipped_empty_indices": skipped_empty,
    }
    for name, contents in (("samples.json", manifest), ("selection.json", selection)):
        (args.output_dir / name).write_text(json.dumps(contents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _stack_panels([comparison_panel(case, None, 0, 0) for case in cases]).save(args.output_dir / "train_inputs.png")
    return cases


def main(args):
    transformer_path, checkpoint_name, step = resolve_checkpoint(args.checkpoint)
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "outputs/inter_edit_train" / checkpoint_name
        if args.cache_first_step:
            args.output_dir = args.output_dir.with_name(checkpoint_name + "_first_step_cache")
    args.output_dir = args.output_dir.expanduser().resolve()
    cases = prepare_cases(args)
    if args.prepare_only:
        print(f"Saved {len(cases)} train inputs to {args.output_dir / 'train_inputs.png'}", flush=True)
        return

    import torch

    from src.editmgt import init_edit_mgt
    from src.train_validation import run_inter_edit_validation

    print(f"Loading Transformer: {transformer_path}", flush=True)
    pipe = init_edit_mgt(device=args.device, enable_bf16=args.bf16,
                         base_model_path=args.model_path, transformer_path=str(transformer_path),
                         condition_cache_threshold=args.condition_cache_threshold)
    for module in (pipe.transformer, pipe.vqvae, pipe.text_encoder, pipe.text_encoder_t5):
        if module is not None:
            module.eval().requires_grad_(False)
    pipe.local_guidance = 0.0
    with torch.inference_mode():
        report = run_inter_edit_validation(
            pipe=pipe, cases=cases, output_dir=args.output_dir, step=step, split="train",
            num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
            reference_strength=args.reference_strength, seed=args.seed, negative_prompt=args.negative_prompt,
            edit_only_diffusion=True, sparse_target=bool(pipe.transformer.config.sparse_target),
            sparse_target_halo=args.sparse_target_halo,
            use_reference_kv_cache=not args.disable_reference_kv_cache,
            reference_kv_cache_threshold=args.reference_kv_cache_threshold,
            use_condition_kv_cache=False if args.disable_condition_kv_cache else None,
            preserve_unmasked_pixels=args.preserve_unmasked_pixels,
            condition_kv_cache_reuse_steps=args.num_inference_steps - 1 if args.cache_first_step else None,
            allow_early_condition_cache=args.cache_first_step,
        )
    metadata_path = report["step_dir"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "checkpoint": str(args.checkpoint.expanduser().resolve()), "transformer_path": str(transformer_path),
        "model_path": args.model_path, "data_root": str(args.data_dir.expanduser().resolve()),
        "samples_manifest": str(args.output_dir / "samples.json"),
        "selection": "indices" if args.sample_indices is not None else args.selection,
        "selection_seed": args.selection_seed, "seed": args.seed, "device": args.device, "bf16": args.bf16,
        "resolution": args.resolution, "mask_input": True,
        "cache_first_step": args.cache_first_step,
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(cases)} train results to {args.output_dir / 'index.html'}", flush=True)
    print(f"Comparison grid: {report['step_dir'] / 'contact_sheet.png'}", flush=True)


if __name__ == "__main__":
    main(parse_args())
