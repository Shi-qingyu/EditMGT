"""Prepare input-only captions for DICE/GIDE on instruction-only test sets.

BLIP captions the provided source image. FLAN-T5 rewrites that caption using
the edit instruction. Neither model sees the target image or annotation mask.
MagicBrush multi-turn captions follow the preceding generated text caption;
standalone turns caption their own provided source image.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.benchmark_data import BenchmarkImages  # noqa: E402
from src.benchmark_utils import atomic_json  # noqa: E402

REFERENCE = ROOT / "outputs/dimo_step4000_sft8000_comparison/vanilla36"
OUTPUT = ROOT / "outputs/real_time_baselines/caption_protocol"
BLIP = ROOT / "third_party/table_completion/models/blip-image-captioning-base"
FLAN = ROOT / "third_party/NEP/pretrained_models/flan-t5-xl"
NAMES = {"magicbrush": "magicbrush_test", "inter_edit": "inter_edit"}


def load_selection(benchmark):
    return json.loads((REFERENCE / NAMES[benchmark] / "selection.json").read_text())


def is_multiturn(task):
    return bool(task.get("input_prediction_key"))


def source_captions(benchmark, selection, source_path, batch_size):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    captions = json.loads(source_path.read_text()) if source_path.exists() else {}
    tasks = [task for task in selection if not is_multiturn(task) and task["key"] not in captions]
    if not tasks:
        return captions
    processor = BlipProcessor.from_pretrained(str(BLIP), local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(
        str(BLIP), torch_dtype=torch.float16, local_files_only=True,
    ).to("cuda").eval()
    resolver = BenchmarkImages(ROOT / "data/inter_edit_tar_index")
    try:
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            images = [resolver.image(task["source"], "RGB") for task in batch]
            inputs = processor(images=images, return_tensors="pt").to("cuda", torch.float16)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=42)
            strings = processor.batch_decode(ids, skip_special_tokens=True)
            for task, caption in zip(batch, strings):
                caption = caption.strip().rstrip(".")
                if not caption:
                    raise ValueError(f"Empty BLIP caption for {task['key']}")
                captions[task["key"]] = caption
            if (start // batch_size) % 20 == 0 or start + len(batch) == len(tasks):
                atomic_json(source_path, captions)
                print(f"{benchmark}: source captions {start + len(batch)}/{len(tasks)}", flush=True)
    finally:
        resolver.close()
    atomic_json(source_path, captions)
    return captions


def edit_type(instruction):
    text = instruction.strip().lower()
    if re.match(r"^(remove|delete|erase|take away|get rid of)\b", text):
        return "3"
    if re.match(r"^(add|put|place|insert|include)\b", text):
        return "2"
    return "1"


def subject_words(source_caption, instruction):
    instruction = instruction.strip().rstrip(".?!")
    replacement = re.search(r"\breplace\s+(?:the|a|an)?\s*(.+?)\s+with\s+(?:the|a|an)?\s*(.+)$", instruction, re.I)
    if not replacement:
        replacement = re.search(r"(?:the|a|an)\s+(.+?)\s+be\s+replaced\s+by\s+(?:the|a|an)\s+(.+)$", instruction, re.I)
    if replacement:
        return replacement.group(1).strip(), replacement.group(2).strip()
    change = re.search(r"\bchange\s+(?:the|a|an)?\s*(.+?)\s+(?:to|into)\s+(.+)$", instruction, re.I)
    if change and len(change.group(1).split()) <= 5 and len(change.group(2).split()) <= 5:
        return change.group(1).strip(), change.group(2).strip()
    source = re.search(
        r"\b(?:a|an|the)\s+([a-z][a-z\s-]*?)(?=\s+(?:with|on|in|at|by|under|near|sitting|standing|walking|wearing|holding|lying|is|are)\b|[,.;]|$)",
        source_caption, re.I,
    )
    phrase = source.group(1).strip() if source else "object"
    word = phrase.split()[-1] if phrase.split() else "object"
    return word, word


def target_captions(benchmark, selection, source, manifest_path, batch_size, rebuild):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() and not rebuild else {})
    tokenizer = AutoTokenizer.from_pretrained(str(FLAN), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(FLAN), torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda").eval()
    by_key = {task["key"]: task for task in selection}
    waves = defaultdict(list)
    for task in selection:
        waves[task.get("turn", 1)].append(task)
    for turn in sorted(waves):
        tasks = [task for task in waves[turn] if task["key"] not in manifest]
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            inputs = []
            for task in batch:
                parent = task.get("input_prediction_key")
                if parent:
                    if parent not in by_key or parent not in manifest:
                        raise ValueError(f"Missing preceding turn for {task['key']}")
                    src = manifest[parent]["caption"]
                else:
                    src = source[task["key"]]
                inputs.append(src)
            prompts = [
                "Rewrite the image description to describe the image after this edit. "
                "Write one descriptive sentence, not an instruction.\n"
                f"Original description: {src}.\nEdit: {task['instruction']}\nEdited description:"
                for src, task in zip(inputs, batch)
            ]
            encoded = tokenizer(prompts, padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                ids = model.generate(**encoded, max_new_tokens=64, num_beams=1)
            targets = tokenizer.batch_decode(ids, skip_special_tokens=True)
            for task, src, target in zip(batch, inputs, targets):
                target = target.strip().replace("\n", " ").rstrip(".")
                if not target:
                    target = src
                # FLAN occasionally paraphrases the source without applying
                # the edit. Include the original instruction so the requested
                # change is always present in the model's target prompt.
                target = f"{target}. Edit: {task['instruction'].strip()}"
                src_subject, tgt_subject = subject_words(src, task["instruction"])
                manifest[task["key"]] = {
                    "source_caption": src,
                    "caption": target,
                    "gide": {
                        "original_prompt": src,
                        "editing_prompt": target,
                        "editing_instruction": task["instruction"],
                        "editing_type_id": edit_type(task["instruction"]),
                        "source_subject": src_subject,
                        "target_subject": tgt_subject,
                    },
                }
            atomic_json(manifest_path, manifest)
            print(f"{benchmark}: turn {turn}, target captions {start + len(batch)}/{len(tasks)}", flush=True)
    if len(manifest) != len(selection):
        raise ValueError(f"Expected {len(selection)} captions, found {len(manifest)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=tuple(NAMES))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--rebuild-targets", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = load_selection(args.benchmark)
    source = source_captions(args.benchmark, selection,
                             OUTPUT / f"{args.benchmark}_sources.json", args.batch_size)
    target_captions(args.benchmark, selection, source,
                    OUTPUT / f"{args.benchmark}.json", args.batch_size,
                    args.rebuild_targets)


if __name__ == "__main__":
    main()
