"""Official FLUX.2 klein 4B four-step Diffusers inference."""

from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "third_party/SpotEdit/models/FLUX.2-klein-4B"


class Backend:
    name = "flux2_klein"
    steps = 4
    resolution = 1024

    def __init__(self, device: str = "cuda"):
        from diffusers import Flux2KleinPipeline

        if not (WEIGHTS / "model_index.json").is_file():
            raise FileNotFoundError(f"Download official FLUX.2 klein weights to {WEIGHTS}")
        self.device = torch.device(device)
        self.pipe = Flux2KleinPipeline.from_pretrained(
            str(WEIGHTS), torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(
            prompt=task["instruction"], image=source,
            height=self.resolution, width=self.resolution,
            guidance_scale=1.0, num_inference_steps=self.steps,
            generator=generator,
        ).images[0].convert("RGB")

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
