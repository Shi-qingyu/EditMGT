"""Compare saved DiMO EMA checkpoints on identical fixed training probes.

The run's fixed probes retain source tokens, masks and prompt embeddings.  This
script also arranges the previews emitted every 500 steps; those previews used
the training RNG and should not be read as a controlled checkpoint comparison.
"""

import argparse
import gc
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import VQModel

from src.dimo.core import generate_one_step
from src.dimo.infer import load_role


def font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size) if Path(path).is_file() else ImageFont.load_default()


def save_tensor_image(tensor, path):
    pixels = tensor[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((pixels * 255).round().astype(np.uint8)).save(path)


def decode(vq, tokens, path):
    b, h, w = tokens.shape
    image = vq.decode(tokens.flatten(), force_not_quantize=True,
                      shape=(b, h, w, vq.config.latent_channels)).sample
    save_tensor_image(image, path)


def grid(rows, columns, path, tile=384):
    """Each row is (label, {column label: image path})."""
    pad, top = 10, 52
    width = len(columns) * (tile + pad) + pad
    captions = [textwrap.wrap(label, width=max(40, (width - 2 * pad) // 11)) or ['']
                for label, _ in rows]
    heights = [tile + len(lines) * 28 + 16 + pad for lines in captions]
    height = top + sum(heights) + pad
    canvas = Image.new("RGB", (width, height), "#f7f7f7")
    draw = ImageDraw.Draw(canvas)
    heading, caption = font(22), font(19)
    for col, title in enumerate(columns):
        draw.text((pad + col * (tile + pad), 12), title, fill="#111111", font=heading)
    y = top
    for (label, images), lines, row_height in zip(rows, captions, heights):
        for col, title in enumerate(columns):
            image_path = images.get(title)
            if image_path is None:
                continue
            with Image.open(image_path) as original:
                thumb = original.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (pad + col * (tile + pad), y))
        for line_number, line in enumerate(lines):
            draw.text((pad, y + tile + 5 + line_number * 28), line, fill="#111111", font=caption)
        y += row_height
    canvas.save(path)


def selected_checkpoints(run, steps):
    if steps:
        candidates = [run / f"checkpoint-{step}" for step in steps]
    else:
        candidates = sorted(run.glob("checkpoint-*"),
                            key=lambda p: int(p.name.removeprefix("checkpoint-")))
    chosen = []
    for path in candidates:
        if not (path / "COMPLETE").is_file():
            raise FileNotFoundError(f"No complete checkpoint: {path}")
        chosen.append((int(path.name.removeprefix("checkpoint-")), path))
    if not chosen:
        raise ValueError(f"No complete checkpoints in {run}")
    return chosen


def preview_grid(run, output, ranks):
    paths = sorted((run / "validation").glob("step-*-rank0.png"),
                   key=lambda p: int(p.name.split("-")[1]))
    steps = [int(p.name.split("-")[1]) for p in paths]
    saved = {int(p.name.removeprefix("checkpoint-")) for p in run.glob("checkpoint-*")
             if (p / "COMPLETE").is_file()}
    columns = ["Source"] + [f"{step} {'checkpoint' if step in saved else 'preview'}" for step in steps]
    rows = []
    for rank in ranks:
        images = {label: run / "validation" / f"step-{step}-rank{rank}.png"
                  for step, label in zip(steps, columns[1:])}
        source = run / "validation" / f"fixed-source-rank{rank}.png"
        if source.is_file():
            images["Source"] = source
        context = probe_context(run, rank)
        rows.append((f"fixed probe rank {rank}; edit instruction: {context['instruction']}", images))
    if steps:
        grid(rows, columns, output / "training_previews.png", tile=320)
    return steps


def probe_context(run, rank):
    path = run / "validation" / f"fixed-probe-rank{rank}.json"
    return json.loads(path.read_text()) if path.is_file() else {"instruction": "Unavailable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="*")
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["ema", "student"], default="ema")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output is nonempty: {output}")
    checkpoints = selected_checkpoints(run, args.steps)
    output.mkdir(parents=True, exist_ok=True)
    previews = preview_grid(run, output, args.ranks)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Checkpoint rerendering requires a CUDA device")
    config = json.loads((run / "config_resolved.yaml").read_text())
    vq = VQModel.from_pretrained(config["base_model_dir"], subfolder="vqvae")
    vq.eval().requires_grad_(False).to(device)
    probes = {}
    for rank in args.ranks:
        path = run / "validation" / f"fixed-probe-rank{rank}.pt"
        probe = torch.load(path, map_location="cpu", weights_only=True)
        probes[rank] = {key: ({k: v.to(device) if torch.is_tensor(v) else v
                              for k, v in value.items()} if isinstance(value, dict)
                              else value.to(device)) for key, value in probe.items()}
        with torch.inference_mode():
            decode(vq, probes[rank]["source"], output / f"source-rank{rank}.png")
        mask = probes[rank]["original"][0].cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask).resize((config["resolution"], config["resolution"]),
                                         Image.Resampling.NEAREST).save(output / f"mask-rank{rank}.png")

    for step, checkpoint in checkpoints:
        print(f"Loading {checkpoint} ({args.model})", flush=True)
        model, metadata = load_role(checkpoint, args.model)
        model.eval().requires_grad_(False).to(device)
        for rank, probe in probes.items():
            with torch.inference_mode(), torch.random.fork_rng(devices=[device.index or 0]):
                torch.manual_seed(args.seed + rank)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = generate_one_step(model, probe["source"], probe["original"],
                                               model.config.vocab_size - 1, probe["cond"],
                                               r_init=config["student_initial_mask_ratio"])
                decode(vq, tokens, output / f"{args.model}-step{step}-rank{rank}.png")
            print(f"Rendered step {step}, rank {rank}", flush=True)
        del model, metadata
        gc.collect()
        torch.cuda.empty_cache()

    columns = ["Source", "edit mask"] + [f"{args.model} step {step}" for step, _ in checkpoints]
    rows = []
    for rank in args.ranks:
        source = run / "validation" / f"fixed-source-rank{rank}.png"
        images = {"Source": source if source.is_file() else output / f"source-rank{rank}.png",
                  "edit mask": output / f"mask-rank{rank}.png"}
        images.update({f"{args.model} step {step}": output / f"{args.model}-step{step}-rank{rank}.png"
                       for step, _ in checkpoints})
        rows.append((f"fixed probe rank {rank}; seed {args.seed + rank}; edit instruction: "
                     f"{probe_context(run,rank)['instruction']}", images))
    grid(rows, columns, output / "checkpoint_comparison.png")
    manifest = dict(run_dir=str(run), checkpoints=[str(path) for _, path in checkpoints],
                    steps=[step for step, _ in checkpoints], ranks=args.ranks, model=args.model,
                    seed=args.seed, previews=previews, output_compositing="none",
                    probe_context={str(rank):probe_context(run,rank) for rank in args.ranks},
                    sampling="one-step categorical; same fixed prompt embeddings and seed per rank")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {output / 'checkpoint_comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
