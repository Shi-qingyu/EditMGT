"""UltraEdit's released SD3 mask-conditioned 50-step editing pipeline."""

import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
FORK = ROOT / "third_party/UltraEdit/diffusers/src"
WEIGHTS = ROOT / "third_party/UltraEdit/models/SD3_UltraEdit_w_mask"


class Backend:
    name = "ultraedit"
    steps = 50
    resolution = 512
    requires_edit_mask = True

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(FORK))
        from diffusers import StableDiffusion3InstructPix2PixPipeline

        if not (WEIGHTS / "model_index.json").is_file():
            raise FileNotFoundError(f"Download official UltraEdit weights to {WEIGHTS}")
        self.device = torch.device(device)
        self.pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            str(WEIGHTS), torch_dtype=torch.float16, local_files_only=True,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(
            task["instruction"], image=source, mask_img=task["_edit_mask"],
            negative_prompt="", num_inference_steps=self.steps,
            image_guidance_scale=1.5, guidance_scale=7.5,
            generator=generator,
        ).images[0].convert("RGB")

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
