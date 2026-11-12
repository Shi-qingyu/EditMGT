"""Multi-seed qualitative inference on the unused docs/cherry_pick cases.

Entry point: bash third_party/table_completion/run_cherry_pick.sh
Defaults: five models, seeds 42..49, outputs/qualitative_comparison.
The shell launcher selects each backend's existing Python environment.

Layout: OUTPUT/cases/TYPE/CASE/{source.png,ours.png,case.json,MODEL/seed_N.png}.
Each prediction has a matching JSON record containing the exact seed, prompts,
model protocol, dimensions, status and any GIDE grounding fallback. index.csv
lists all planned predictions and their final statuses. No metrics are run.

The prepare stage snapshots inputs and prompts before any GPU process starts.
GIDE/SwiftEdit receive the manually authored source/target captions in the
configuration; other models receive the editing instruction. Annotation masks,
target images and existing EditMGT predictions are saved only as references.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MARKER = "USED_IN_QUALITATIVE_RESULTS.txt"
MODEL_SPECS = {
    "spotedit": {
        "backend": "spotedit_flux2_4b",
        "label": "SpotEdit / FLUX.2 Klein 4B (local four-step adaptation)",
        "steps": 4, "resolution": 1024, "prompt_protocol": "instruction",
        "weights": "third_party/SpotEdit/models/FLUX.2-klein-4B",
        "settings": {"guidance_scale": 1.0, "threshold": 0.4,
                     "initial_steps": 1, "reset_steps": []},
    },
    "lumina_dimoo": {
        "backend": "lumina_dimoo", "label": "Lumina-DiMOO",
        "steps": 64, "resolution": 512, "prompt_protocol": "instruction",
        "weights": "third_party/Lumina-DiMOO/models/Lumina-DiMOO",
        "settings": {"cfg_scale": 2.5, "cfg_img": 4.0, "temperature": 1.0},
    },
    "gide": {
        "backend": "gide_pie", "label": "GIDE / Lumina-DiMOO + SAM 2 + SAM 3",
        "steps": "64 inversion + 64 editing/refinement; removal skips inversion",
        "resolution": 512, "prompt_protocol": "authored captions and subjects",
        "weights": ["third_party/Lumina-DiMOO/models/Lumina-DiMOO",
                    "third_party/GIDE/models/sam2-hiera-large/sam2_hiera_large.pt",
                    "third_party/GIDE/models/sam3/sam3.pt"],
        "settings": {"cfg_scale": 4.0, "temperature": 1.0, "lamda1_ratio": 0.2,
                     "grounding": "sam3_text; full-frame fallback if inversion grounding is empty"},
    },
    "dreamlite": {
        "backend": "dreamlite_mobile", "label": "DreamLite-mobile",
        "steps": 4, "resolution": 1024, "prompt_protocol": "instruction",
        "weights": "third_party/DreamLite/models/DreamLite-mobile",
    },
    "swiftedit": {
        "backend": "swiftedit", "label": "SwiftEdit v1.0",
        "steps": "1 inversion + 1 editing", "resolution": 512,
        "prompt_protocol": "authored source and target captions",
        "weights": ["third_party/SwiftEdit/swiftedit_weights",
                    "third_party/ChordEdit/models/sd-turbo",
                    "third_party/SwiftEdit/models/sd21base",
                    "third_party/SwiftEdit/models/ip_adapter"],
    },
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prediction_path(case, model, seed):
    return Path(case["output_dir"]) / model / f"seed_{seed:010d}.png"


def prepare(args):
    from PIL import Image

    cherry = args.cherry_pick.resolve()
    annotations = read_json(args.case_config)["cases"]
    train = (read_json(args.train_manifest)["samples"]
             if args.train_manifest.is_file() else [])
    train = {row["name"]: row for row in train}
    previously_used = set()
    usage = cherry / "qualitative_results_usage.csv"
    if usage.is_file():
        with usage.open(encoding="utf-8", newline="") as stream:
            previously_used = {row["case_path"] for row in csv.DictReader(stream)
                               if row["used_in_provided_figure"].lower() == "yes"}
    cases, excluded = [], []
    for category in ("add", "local", "remove", "texture"):
        for folder in sorted((cherry / category).iterdir()):
            if not folder.is_dir():
                continue
            key = folder.relative_to(cherry).as_posix()
            if (folder / MARKER).exists() or key in previously_used:
                excluded.append(key)
                continue
            if key not in annotations:
                raise ValueError(f"Add captions/subjects for {key} to {args.case_config}")
            extra = annotations[key]
            for field in ("source_caption", "target_caption", "source_subject", "target_subject"):
                if not extra.get(field):
                    raise ValueError(f"Missing {field} for {key}")
            instruction_file = folder / "instruction.txt"
            if instruction_file.is_file():
                original = instruction_file.read_text(encoding="utf-8").strip()
                origin = str(instruction_file)
            elif folder.name in train:
                original = train[folder.name]["instruction"]
                origin = f"{args.train_manifest}#{folder.name}"
            elif (folder / "metadata.json").is_file():
                original = read_json(folder / "metadata.json")["case"]["instruction"]
                origin = str(folder / "metadata.json")
            else:
                original = None
                origin = extra.get("instruction_origin", "unavailable")
            instruction = extra.get("instruction_override") or original or extra.get("instruction")
            if not instruction:
                raise ValueError(f"Missing instruction for {key}")
            source = folder / "source.png"
            crop = None
            if not source.is_file():
                # src/train_validation.py: five square tiles, 28px header + 24px labels.
                # Legacy names are not current test indices: do not look them up by ID.
                source = folder / "comparison.png"
                with Image.open(source) as panel:
                    tile = panel.width // 5
                    if panel.width % 5 or panel.height != tile + 52:
                        raise ValueError(f"Unrecognized comparison layout: {source}")
                    crop = [0, 52, tile, panel.height]
            references = {}
            for name in ("mask.png", "target.png", "comparison.png", "generation_mask.png"):
                if (folder / name).is_file():
                    references[name] = str(folder / name)
            for name in ("sft18.png", "generated.png", "edited.png"):
                if (folder / name).is_file():
                    references["ours.png"] = str(folder / name)
                    break
            if "ours.png" not in references:
                raise ValueError(f"Missing existing EditMGT output for {key}")
            case = {
                "key": key, "edit_type": category, "output_dir": f"cases/{key}",
                "original_instruction": original, "instruction": instruction,
                "instruction_origin": origin,
                "instruction_note": extra.get("instruction_note", ""),
                "source_input": str(source), "source_crop": crop,
                "source_sha256": file_hash(source),
                "source_note": ("First source tile from legacy comparison, 512px preview."
                                if crop else "Standalone source image from cherry_pick."),
                "source_caption": extra["source_caption"],
                "caption": extra["target_caption"],
                "caption_origin": "Authored supplemental captions in cherry_pick_cases.json",
                "gide": {
                    "original_prompt": extra["source_caption"],
                    "editing_prompt": extra["target_caption"],
                    "source_subject": extra["source_subject"],
                    "target_subject": extra["target_subject"],
                    "editing_type_id": {"add": "2", "remove": "3"}.get(category, "0"),
                    "editing_instruction": instruction,
                },
                "references": references,
                "reference_sha256": {name: file_hash(Path(path))
                                     for name, path in references.items()},
            }
            cases.append(case)
    if not cases:
        raise ValueError("No unused cherry-pick cases found")
    settings = {"version": 1, "models": {name: MODEL_SPECS[name] for name in args.models},
                "seeds": args.seeds, "cases": cases, "excluded": excluded,
                "annotation_mask_as_model_input": False,
                "target_image_as_model_input": False}
    run_id = fingerprint(settings)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file():
        old = read_json(manifest_path)
        if not args.resume or old["run_id"] != run_id:
            raise ValueError("Output already exists or settings changed; use --resume with the same settings, or a new OUTPUT_DIR")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        dest = args.output_dir / case["output_dir"]
        dest.mkdir(parents=True, exist_ok=True)
        with Image.open(case["source_input"]) as source:
            source = source.convert("RGB")
            if case["source_crop"]:
                source = source.crop(tuple(case["source_crop"]))
            source.save(dest / "source.png")
        for name, original_path in case["references"].items():
            shutil.copy2(original_path, dest / name)
        (dest / "instruction.txt").write_text(case["instruction"] + "\n", encoding="utf-8")
        atomic_json(dest / "case.json", case)
    atomic_json(manifest_path, {"run_id": run_id, "settings": settings})
    count = len(cases) * len(args.models) * len(args.seeds)
    print(f"Prepared {len(cases)} cases × {len(args.models)} models × {len(args.seeds)} seeds = {count} outputs", flush=True)


def completed(root, relative_path, run_id):
    image_path = root / relative_path
    record_path = image_path.with_suffix(".json")
    if not image_path.is_file() or not record_path.is_file() or image_path.stat().st_size == 0:
        return False
    try:
        record = read_json(record_path)
        return record.get("status") == "ok" and record.get("run_id") == run_id
    except (OSError, ValueError):
        return False


def generate(args):
    import numpy as np
    import torch
    from PIL import Image
    import src  # noqa: F401 -- initialize EditMGT's namespace before SwiftEdit extends it

    manifest = read_json(args.output_dir / "manifest.json")
    settings, run_id = manifest["settings"], manifest["run_id"]
    if args.model not in settings["models"]:
        raise ValueError(f"Model {args.model} was not included in prepare")
    spec = settings["models"][args.model]
    jobs = [(case, seed) for case in settings["cases"] for seed in settings["seeds"]]
    jobs = [job for index, job in enumerate(jobs) if index % args.world_size == args.rank]
    pending = []
    for case, seed in jobs:
        rel = prediction_path(case, args.model, seed)
        if args.resume and completed(args.output_dir, rel, run_id):
            continue
        if not args.resume and (args.output_dir / rel).exists():
            raise FileExistsError(f"Prediction exists: {rel}; use --resume")
        pending.append((case, seed))
    if not pending:
        print(f"{args.model} rank {args.rank}: nothing to generate", flush=True)
        return
    module = importlib.import_module(f"backends.{spec['backend']}")
    # Custom tasks supply their own caption/subject rows, so no PIE dataset is read.
    backend = module.Backend(metadata={}) if args.model == "gide" else module.Backend()
    execution = {"model": args.model, "protocol": spec, "rank": args.rank,
                 "world_size": args.world_size, "torch": torch.__version__,
                 "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                 "python": sys.executable, "run_id": run_id}
    atomic_json(args.output_dir / "execution" / args.model / f"rank_{args.rank}.json", execution)
    failures = 0
    for index, (case, seed) in enumerate(pending, 1):
        image_path = args.output_dir / prediction_path(case, args.model, seed)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"run_id": run_id, "case": case["key"], "model": args.model,
                  "seed": seed, "instruction": case["instruction"], "protocol": spec,
                  "source_caption": case["source_caption"], "target_caption": case["caption"],
                  "gide": case["gide"] if args.model == "gide" else None,
                  "created_at": datetime.now(timezone.utc).isoformat()}
        try:
            # Seed all global RNGs as well as backend-specific Generators. This
            # covers SwiftEdit VAE sampling and GIDE's inversion/refinement RNGs.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with Image.open(args.output_dir / case["output_dir"] / "source.png") as opened:
                source = opened.convert("RGB")
            model_source = backend.prepare_source(source)
            task = {"key": case["key"], "instruction": case["instruction"]}
            if args.model == "gide":
                task["gide"] = case["gide"]
            elif args.model == "swiftedit":
                task.update(source_caption=case["source_caption"], caption=case["caption"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                result = backend.generate(model_source, task, seed).convert("RGB")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            temporary = image_path.with_name(f".{image_path.name}.{os.getpid()}.tmp")
            result.save(temporary, format="PNG")
            temporary.replace(image_path)
            record.update(status="ok", seconds=elapsed, source_size=list(source.size),
                          model_input_size=list(model_source.size), output_size=list(result.size),
                          image=str(image_path.relative_to(args.output_dir)),
                          fallback_reason=getattr(backend, "last_fallback", None))
        except Exception as error:
            failures += 1
            record.update(status="error", error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
            traceback.print_exc()
            torch.cuda.empty_cache()
        atomic_json(image_path.with_suffix(".json"), record)
        print(f"{args.model} rank {args.rank}: {index}/{len(pending)} {case['key']} seed={seed} {record['status']}", flush=True)
    if failures:
        raise SystemExit(1)


def summarize(args):
    manifest = read_json(args.output_dir / "manifest.json")
    settings, run_id = manifest["settings"], manifest["run_id"]
    rows = []
    for case in settings["cases"]:
        for model in settings["models"]:
            for seed in settings["seeds"]:
                relative = prediction_path(case, model, seed)
                record_path = (args.output_dir / relative).with_suffix(".json")
                try:
                    record = read_json(record_path) if record_path.is_file() else {}
                except (OSError, ValueError):
                    record = {"error": "Unreadable prediction record"}
                status = "ok" if completed(args.output_dir, relative, run_id) else "missing_or_error"
                rows.append({"case": case["key"], "model": model, "seed": seed,
                             "status": status, "image": relative.as_posix(),
                             "record": relative.with_suffix(".json").as_posix(),
                             "instruction": case["instruction"], "error": record.get("error", "")})
    with (args.output_dir / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(row["status"] == "ok" for row in rows)
    atomic_json(args.output_dir / "summary.json", {"run_id": run_id, "expected": len(rows),
                                                  "ok": ok, "missing_or_error": len(rows) - ok})
    print(f"Saved {args.output_dir / 'index.csv'}: {ok}/{len(rows)} predictions", flush=True)
    if ok != len(rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("prepare", "generate", "summarize"), required=True)
    parser.add_argument("--cherry-pick", type=Path, default=ROOT / "docs/cherry_pick")
    parser.add_argument("--case-config", type=Path, default=HERE / "cherry_pick_cases.json")
    parser.add_argument("--train-manifest", type=Path,
                        default=ROOT / "outputs/inter_edit_train/sft_checkpoint_11000_large/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qualitative_comparison")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--model", choices=tuple(MODEL_SPECS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Models and seeds must not contain duplicates")
    if any(seed < 0 or seed >= 2 ** 32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32), matching all five backends")
    if not 0 <= args.rank < args.world_size:
        parser.error("Require 0 <= rank < world-size")
    if args.stage == "generate" and args.model is None:
        parser.error("--model is required for generate")
    {"prepare": prepare, "generate": generate, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
