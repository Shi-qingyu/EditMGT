"""Fixed Inter-Edit test cases and image reports for training-time validation."""
import html
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.inter_edit_dataset import InterEditReader, TarAssetReader, matches_inter_edit_language
from src.mask_utils import normalize_edit_mask_image, resize_binary_edit_mask


def synchronized_main_process_call(accelerator, callback, description="Validation"):
    """Pause all ranks; propagate rank-zero errors before anyone resumes DDP."""
    accelerator.wait_for_everyone()
    result, error = None, None
    if accelerator.is_main_process:
        try:
            result = callback()
        except Exception as exc:
            error = exc
    failed = torch.tensor(int(error is not None), device=accelerator.device)
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(failed, src=0)
    if failed.item():
        raise RuntimeError(f"{description} failed on the main process") from error
    accelerator.wait_for_everyone()
    return result


def select_validation_indices(rows, num_images=5, sample_indices=None, language="all"):
    """Pick edit types within the requested language, preserving official row indices."""
    eligible_indices = [index for index, row in enumerate(rows)
                        if matches_inter_edit_language(row, language)]
    if sample_indices is not None:
        indices = list(sample_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Validation sample indices must be nonempty and unique")
        if any(index < 0 or index >= len(rows) for index in indices):
            raise IndexError("Validation sample index is outside the Inter-Edit test split")
        ineligible = [index for index in indices if not matches_inter_edit_language(rows[index], language)]
        if ineligible:
            raise ValueError(f"Validation indices {ineligible} do not match language={language!r}")
        return indices
    if not 1 <= num_images <= len(eligible_indices):
        raise ValueError(
            f"validation_num_images must be between 1 and {len(eligible_indices)} "
            f"for Inter-Edit test language={language!r}"
        )
    indices, seen_types = [], set()
    for index in eligible_indices:
        row = rows[index]
        if row["edit_type"] not in seen_types:
            indices.append(index)
            seen_types.add(row["edit_type"])
        if len(indices) == num_images:
            return indices
    for index in eligible_indices:
        if index not in indices:
            indices.append(index)
        if len(indices) == num_images:
            break
    return indices


def mask_overlay(source, mask, color=(255, 45, 45)):
    tinted = Image.blend(source, Image.new("RGB", source.size, color), 0.4)
    return Image.composite(tinted, source, mask)


def prepare_inter_edit_validation_cases(
    data_dir, output_dir, resolution, num_images=5, sample_indices=None, cache_dir=None,
    language="all", translations_dir=None,
):
    """Read/resize the test cases once; validation steps reuse these PIL images."""
    reader = InterEditReader(data_dir, "test", cache_dir, translations_dir=translations_dir)
    indices = select_validation_indices(reader.test_rows, num_images, sample_indices, language)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, manifest = [], []
    with TarAssetReader(reader.root, cache_dir) as assets:
        for index in indices:
            raw = reader.read_sample(reader.test_rows[index], assets)
            size = (resolution, resolution)
            source = raw["input"].convert("RGB").resize(size, Image.Resampling.BILINEAR)
            target = raw["output"].convert("RGB").resize(size, Image.Resampling.BILINEAR)
            mask = resize_binary_edit_mask(
                normalize_edit_mask_image(raw["edit_mask"]), (resolution, resolution)
            )
            if mask.getbbox() is None:
                raise ValueError(f"Inter-Edit validation case {raw['sample_id']} has an empty mask")
            overlay = mask_overlay(source, mask)
            case_dir = output_dir / "inputs" / f"sample_{raw['sample_id']:05d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            for name, image in (("source", source), ("target", target), ("mask", mask), ("mask_overlay", overlay)):
                image.save(case_dir / f"{name}.png")
            metadata = {
                "test_index": index, "sample_id": raw["sample_id"],
                "instruction": raw["instruction"], "edit_type": raw["edit_type"],
                "language": raw.get("language"), "language_filter": language, "resolution": resolution,
                "edit_area_ratio": float(np.asarray(mask).mean() / 255),
                "source_original_size": list(raw["input"].size),
                "target_original_size": list(raw["output"].size),
                "mask_original_size": list(raw["edit_mask"].size),
                "input_dir": str(case_dir.relative_to(output_dir)),
            }
            for field in ("instruction_original", "language_original"):
                if field in raw:
                    metadata[field] = raw[field]
            cases.append({**metadata, "source": source, "target": target, "mask": mask, "overlay": overlay})
            manifest.append(metadata)
    (output_dir / "samples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    language_summary = {
        "split": "test", "data_root": str(reader.root), "language_filter": language,
        "total_samples": reader.num_samples,
        "eligible_samples": sum(matches_inter_edit_language(row, language) for row in reader.test_rows),
        "selected_test_indices": indices,
    }
    if reader.translations is not None:
        language_summary.update(translations_dir=str(reader.translations.root), instruction_variant="translated_english")
    (output_dir / "selection.json").write_text(
        json.dumps(language_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    _stack_panels([comparison_panel(case, None, 0, 0) for case in cases]).save(output_dir / "test_inputs.png")
    return cases


@contextmanager
def validation_model_state(model, ema=None):
    """Temporarily use eval/EMA weights without changing training RNG streams."""
    parameters = tuple(model.parameters())
    device = parameters[0].device if parameters else torch.device("cpu")
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    was_training, ema_stored = model.training, False
    missing = object()
    reference_strengths = [(module, getattr(module, "reference_strength_factor", missing)) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=devices):
            try:
                if ema is not None:
                    ema.store(parameters)
                    ema_stored = True
                    ema.copy_to(parameters)
                model.eval()
                with torch.no_grad():
                    yield
            finally:
                try:
                    if ema_stored:
                        ema.restore(parameters)
                finally:
                    # Pipeline reference-strength overrides must not alter the
                    # attention used by subsequent training updates.
                    for module, strength in reference_strengths:
                        if strength is missing:
                            if hasattr(module, "reference_strength_factor"):
                                delattr(module, "reference_strength_factor")
                        else:
                            module.reference_strength_factor = strength
                    model.train(was_training)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def comparison_panel(case, generated, step, seed, generation_overlay=None):
    tile = min(512, case["source"].width)
    header, labels = 28, 24
    images = [case["source"], case["overlay"]]
    column_labels = ["Source", "Original edit mask"]
    if generation_overlay is not None:
        images.append(generation_overlay)
        column_labels.append("Generation token mask")
    images.append(case["target"])
    column_labels.append("Ground truth")
    if generated is not None:
        images.append(generated)
        column_labels.append("Generated")
    panel = Image.new("RGB", (tile * len(images), tile + header + labels), (245, 245, 245))
    draw, font = ImageDraw.Draw(panel), ImageFont.load_default()
    title = f"Sample {case['sample_id']} | {case['edit_type']}"
    if generated is not None:
        title = f"Step {step} | {title} | Seed {seed}"
    draw.text((8, 8), title, fill="black", font=font)
    for column, (label, image) in enumerate(zip(column_labels, images)):
        draw.text((column * tile + 8, header + 5), label, fill="black", font=font)
        panel.paste(image.resize((tile, tile), Image.Resampling.BILINEAR), (column * tile, header + labels))
    return panel


def _stack_panels(panels):
    sheet = Image.new("RGB", (max(panel.width for panel in panels), sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    return sheet


def _write_html_report(output_dir, step_dir, records, split="test"):
    rows = []
    for record in records:
        input_dir = output_dir / record["input_dir"]
        paths = [input_dir / "source.png", input_dir / "mask_overlay.png"]
        labels = ["Source", "Original edit mask"]
        if record.get("generation_mask_overlay_path"):
            paths.append(step_dir / record["generation_mask_overlay_path"])
            labels.append("Generation token mask")
        paths.extend((input_dir / "target.png", step_dir / record["generated_path"]))
        labels.extend(("Ground truth", "Generated"))
        images = "".join(
            f'<figure><figcaption>{label}</figcaption><img src="{html.escape(os.path.relpath(path, step_dir))}"></figure>'
            for label, path in zip(labels, paths)
        )
        original_instruction = (
            f'<p>Original instruction: {html.escape(record["original_instruction"])}</p>'
            if "original_instruction" in record else ""
        )
        rows.append(f'<section><h2>Sample {record["sample_id"]} · {html.escape(record["edit_type"])}</h2>'
                    f'{original_instruction}<p>{html.escape(record["instruction"])}</p>'
                    f'<div class="images">{images}</div></section>')
    style = "body{font-family:sans-serif;margin:24px} .images{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:12px} figure{margin:0} img{width:100%}"
    (step_dir / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{step_dir.name}</title><style>{style}</style>'
        f'<h1>Inter-Edit {split} · {step_dir.name}</h1>'
        '<p>Red: original edit mask. Orange: internal token generation region (halo is disabled by default). '
        'By default, the complete token grid is decoded without RGB compositing. '
        'When pixel preservation is explicitly enabled, only the red region is composited into the final image.</p>'
        + "".join(rows), encoding="utf-8",
    )
    steps = sorted(path for path in output_dir.glob("step_*") if (path / "index.html").is_file())
    links = "".join(f'<li><a href="{path.name}/index.html">{path.name}</a> · '
                    f'<a href="{path.name}/contact_sheet.png">comparison grid</a></li>' for path in steps)
    title = "Inter-Edit train inference" if split == "train" else "Inter-Edit training validation"
    (output_dir / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
        f'<h1>{title}</h1><ul>{links}</ul>', encoding="utf-8",
    )


@torch.no_grad()
def run_inter_edit_validation(
    pipe, cases, output_dir, step, num_inference_steps=18, guidance_scale=6.0,
    reference_strength=1.0, seed=42, edit_only_diffusion=True, sparse_target=True,
    sparse_target_halo=0, negative_prompt=None, use_ema=False,
    use_reference_kv_cache=True, reference_kv_cache_threshold=0.8,
    use_condition_kv_cache=None,
    inference_shift=1.0,
    preserve_unmasked_pixels=False,
    split="test",
    condition_kv_cache_reuse_steps=None,
    allow_early_condition_cache=False,
):
    """Generate one case at a time to keep validation memory bounded."""
    if split not in ("train", "test"):
        raise ValueError("Inter-Edit report split must be 'train' or 'test'")
    if not cases:
        raise ValueError("Inter-Edit validation needs at least one case")
    if sparse_target_halo is None:
        model_config = getattr(getattr(pipe, "transformer", None), "config", None)
        sparse_target_halo = getattr(model_config, "sparse_target_halo", 0)
    started = time.perf_counter()
    output_dir = Path(output_dir)
    step_dir = output_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    pipe.set_progress_bar_config(disable=True)
    records, panels, captions = [], [], []
    for case in cases:
        case_seed = seed + int(case["sample_id"])
        generator = torch.Generator(device=pipe._execution_device).manual_seed(case_seed)
        generated = pipe(
            prompt=[case["instruction"]], reference_image=[case["source"]], mask_image=case["mask"],
            height=case["resolution"], width=case["resolution"],
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
            reference_strength=reference_strength, generator=generator,
            negative_prompt=negative_prompt, edit_only_diffusion=edit_only_diffusion,
            sparse_target=sparse_target, sparse_target_halo=sparse_target_halo,
            lora_part_enable=True,
            use_reference_kv_cache=use_reference_kv_cache,
            reference_kv_cache_threshold=reference_kv_cache_threshold,
            use_condition_kv_cache=use_condition_kv_cache,
            inference_shift=inference_shift,
            preserve_unmasked_pixels=preserve_unmasked_pixels,
            condition_kv_cache_reuse_steps=condition_kv_cache_reuse_steps,
            allow_early_condition_cache=allow_early_condition_cache,
        ).images[0].convert("RGB")
        case_dir = step_dir / f"sample_{case['sample_id']:05d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        generated.save(case_dir / "generated.png")
        generation_overlay = None
        generation_paths = {}
        generation_mask = getattr(pipe, "last_generation_mask", None)
        if generation_mask is not None:
            generation_mask_image = Image.fromarray(
                generation_mask[0].numpy().astype(np.uint8) * 255,
            ).resize(case["source"].size, Image.Resampling.NEAREST)
            generation_mask_image.save(case_dir / "generation_mask.png")
            generation_overlay = mask_overlay(case["source"], generation_mask_image, color=(255, 165, 0))
            generation_overlay.save(case_dir / "generation_mask_overlay.png")
            generation_paths = {
                "generation_mask_path": str((case_dir / "generation_mask.png").relative_to(step_dir)),
                "generation_mask_overlay_path": str((case_dir / "generation_mask_overlay.png").relative_to(step_dir)),
            }
        panel = comparison_panel(case, generated, step, case_seed, generation_overlay=generation_overlay)
        panel.save(case_dir / "comparison.png")
        panels.append(panel)
        captions.append(f"sample={case['sample_id']} | {case['edit_type']} | {case['instruction']}")
        records.append({key: value for key, value in case.items() if key not in ("source", "target", "mask", "overlay")})
        records[-1].update({
            "seed": case_seed, "generated_path": str((case_dir / "generated.png").relative_to(step_dir)),
            "reference_kv_cache": getattr(pipe, "last_reference_cache_info", None),
            "condition_kv_cache": getattr(pipe, "last_condition_cache_info", None),
            "edit_region": getattr(pipe, "last_edit_region_info", None),
            **generation_paths,
        })
    contact_sheet = _stack_panels(panels)
    contact_sheet.save(step_dir / "contact_sheet.png")
    elapsed = time.perf_counter() - started
    model_config = getattr(getattr(pipe, "transformer", None), "config", None)
    metadata = {
        "step": step, "split": split, "use_ema": use_ema, "seconds": elapsed,
        "language_filter": cases[0].get("language_filter", "all"),
        "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale,
        "reference_strength": reference_strength, "negative_prompt": negative_prompt,
        "edit_only_diffusion": edit_only_diffusion, "sparse_target": sparse_target,
        "sparse_target_halo": sparse_target_halo, "cases": records,
        "supervise_target_halo": getattr(model_config, "supervise_target_halo", False),
        "preserve_unmasked_pixels": bool(preserve_unmasked_pixels and (edit_only_diffusion or sparse_target)),
        "output_compositing": (
            "original_pixel_mask" if preserve_unmasked_pixels and (edit_only_diffusion or sparse_target) else "none"
        ),
        "use_reference_kv_cache": use_reference_kv_cache,
        "reference_kv_cache_threshold": reference_kv_cache_threshold,
        "use_condition_kv_cache": use_condition_kv_cache,
        "condition_kv_cache_reuse_steps": condition_kv_cache_reuse_steps,
        "allow_early_condition_cache": allow_early_condition_cache,
        "cache_conditioning": getattr(model_config, "cache_conditioning", False),
        "condition_cache_threshold": getattr(model_config, "condition_cache_threshold", None),
        "time_conditioning": getattr(model_config, "time_conditioning", "legacy"),
        "masking_schedule": pipe.scheduler.config.masking_schedule,
        "inference_shift": pipe.scheduler.inference_shift,
        "inference_schedule": getattr(pipe, "last_inference_schedule_info", None),
    }
    (step_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_html_report(output_dir, step_dir, records, split=split)
    return {"step_dir": step_dir, "panels": panels, "captions": captions,
            "contact_sheet": contact_sheet, "seconds": elapsed}
