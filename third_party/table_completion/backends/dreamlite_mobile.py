"""Official DreamLite-mobile four-step editing pipeline."""

import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/DreamLite"
WEIGHTS = REPO / "models/DreamLite-mobile"


class Backend:
    name = "dreamlite_mobile"
    steps = 4
    resolution = 1024

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        from dreamlite import DreamLiteMobilePipeline

        if not (WEIGHTS / "model_index.json").is_file():
            raise FileNotFoundError(f"Download the official main-branch weights to {WEIGHTS}")
        self.device = torch.device(device)
        self.pipe = DreamLiteMobilePipeline.from_pretrained(
            str(WEIGHTS), torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        generator = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            prompt=task["instruction"], image=source,
            height=self.resolution, width=self.resolution,
            num_inference_steps=self.steps, generator=generator,
        ).images[0].convert("RGB")

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
