"""Build the rank-owned Inter-Edit, MagicBrush and MaskEdit training mixture."""
import hashlib
import json
import math
from pathlib import Path

from src.inter_edit_dataset import InterEditDataset
from src.magicbrush_dataset import MagicBrushDataset
from src.maskedit_dataset import MaskEditDataset
from src.mask_utils import MAGICBRUSH_MASK_DECODING
from src.mixed_edit_dataset import MixedEditDataset


def edit_data_probabilities(magicbrush_enabled, magicbrush_probability, maskedit_enabled, maskedit_probability):
    probabilities = {"Inter-Edit": 1.0}
    for name, enabled, probability in (("MagicBrush", magicbrush_enabled, magicbrush_probability),
                                       ("MaskEdit-10K", maskedit_enabled, maskedit_probability)):
        if not math.isfinite(probability) or not 0 <= probability < 1 or (enabled and probability == 0):
            raise ValueError(f"{name} probability must be finite and in (0,1) when enabled")
        if enabled:
            probabilities[name] = probability
    probabilities["Inter-Edit"] = 1.0 - math.fsum(value for name, value in probabilities.items() if name != "Inter-Edit")
    if probabilities["Inter-Edit"] <= 0:
        raise ValueError("MagicBrush + MaskEdit probabilities must sum to less than 1")
    return probabilities


def build_inter_edit_training_data(args, tokenizer, rank, world_size):
    probabilities = edit_data_probabilities(bool(args.magicbrush_data_dir), args.magicbrush_probability,
                                          bool(args.maskedit_data_dir), args.maskedit_probability)
    common = dict(tokenizer=tokenizer, size=args.resolution, text_encoder_architecture=args.text_encoder_architecture,
                  rank=rank, world_size=world_size, seed=args.seed if args.seed is not None else 42)
    inter_edit = InterEditDataset(root_dir=args.instance_data_dir, split=args.dataset_split,
                                 cache_dir=args.dataset_cache_dir, language=args.dataset_language,
                                 translations_dir=args.inter_edit_translations_dir, **common)
    datasets, names = [inter_edit], ["Inter-Edit"]
    sources = [{"name": "Inter-Edit", "root": str(inter_edit.reader.root.resolve()), "split": args.dataset_split,
                "sample_count": len(inter_edit), "probability": probabilities["Inter-Edit"]}]
    if inter_edit.reader.translations is not None:
        overlay = inter_edit.reader.translations
        sources[0].update(translations_dir=str(overlay.root), translation_model=overlay.config.get("model"),
                          translated_prompt_count_global=overlay.summary["translated"])
    if args.magicbrush_data_dir:
        magicbrush = MagicBrushDataset(root_dir=args.magicbrush_data_dir, split="train", language=args.dataset_language,
                                      cache_dir=Path(args.dataset_cache_dir) / "magicbrush" if args.dataset_cache_dir else None,
                                      read_batch_size=args.magicbrush_read_batch_size, **common)
        datasets.append(magicbrush)
        names.append("MagicBrush")
        sources.append({"name": "MagicBrush", "root": str(magicbrush.root.resolve()), "split": "train",
                        "sample_count": len(magicbrush), "probability": probabilities["MagicBrush"],
                        "edit_mask_decoding": MAGICBRUSH_MASK_DECODING})
    if args.maskedit_data_dir:
        maskedit = MaskEditDataset(root_dir=args.maskedit_data_dir, configs=args.maskedit_configs,
                                  prompt_field=args.maskedit_prompt_field,
                                  cache_dir=Path(args.dataset_cache_dir) / "maskedit" if args.dataset_cache_dir else None,
                                  read_batch_size=args.maskedit_read_batch_size, **common)
        datasets.append(maskedit)
        names.append("MaskEdit-10K")
        sources.append({"name": "MaskEdit-10K", "root": str(maskedit.root.resolve()), "split": "train",
                        "sample_count": len(maskedit), "probability": probabilities["MaskEdit-10K"],
                        "configs": list(maskedit.configs), "prompt_field": maskedit.prompt_field,
                        "mask_convention": "white=edit", "prompt_language_filter": "en_without_han",
                        "content_language_filter": "exclude_chinese"})
    dataset = inter_edit if len(datasets) == 1 else MixedEditDataset(
        datasets=datasets, names=names, weights=[probabilities[name] for name in names], rank=rank, seed=common["seed"],
    )
    info = {"rank": rank, "world_size": world_size, "language_filter": args.dataset_language,
            "count_scope": "rank-local, language-filtered, before empty-mask exclusions",
            "resume_from_checkpoint": args.resume_from_checkpoint, "sources": sources,
            "validation": {"dataset": "Inter-Edit", "split": "test", "language_filter": args.dataset_language}}
    return dataset, info


def write_data_mixture_info(output_dir, info):
    root = Path(output_dir) / "data_mixture"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"rank_{info['rank']:04d}.json"
    payload = json.dumps(info, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        previous = path.read_bytes()
        history = root / "history"
        history.mkdir(exist_ok=True)
        digest = hashlib.sha256(previous).hexdigest()[:16]
        (history / f"{path.stem}_{digest}.json").write_bytes(previous)
    path.write_text(payload, encoding="utf-8")
