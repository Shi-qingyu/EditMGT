"""Official TurboEdit four-step SDXL-Turbo code with in-memory image input."""

import os
import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/turbo-edit"
WEIGHTS = REPO / "models/sdxl-turbo"


class Backend:
    name = "turboedit"
    steps = 4
    resolution = 512

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        import main
        from diffusers import AutoPipelineForImage2Image, DDPMScheduler

        if not (WEIGHTS / "model_index.json").is_file():
            raise FileNotFoundError(f"Download stabilityai/sdxl-turbo fp16 to {WEIGHTS}")
        self.device = torch.device(device)
        self.main = main
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            str(WEIGHTS), torch_dtype=torch.float16, variant="fp16",
            safety_checker=None, local_files_only=True,
        ).to(self.device)
        self.pipe.scheduler = DDPMScheduler.from_pretrained(
            str(WEIGHTS), subfolder="scheduler", local_files_only=True,
        )
        self.pipe.set_progress_bar_config(disable=True)
        # The official set_pipeline() resolves run_configs relative to CWD.
        os.chdir(REPO)

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        from utils import SAMPLING_DEVICE, create_xts, get_ddpm_inversion_scheduler

        main = self.main
        generator = torch.Generator(device=SAMPLING_DEVICE).manual_seed(seed)
        timesteps, config = main.set_pipeline(self.pipe, self.steps, generator)
        x0 = main.encode_image(source, self.pipe, generator)
        x_ts = create_xts(
            config.noise_shift_delta, config.noise_timesteps,
            generator, self.pipe.scheduler, timesteps, x0,
        )
        x_ts = [x.to(dtype=x0.dtype) for x in x_ts]
        latents = [x_ts[0]]
        self.pipe.scheduler = get_ddpm_inversion_scheduler(
            self.pipe.scheduler, config, timesteps, latents, x_ts,
        )
        self.pipe.scheduler.w1 = 1.5  # official main.py default
        source_prompt = task.get("source_caption") or ""
        target_prompt = task.get("caption") or task["instruction"]
        image = self.pipe.__call__(
            image=latents[0].expand(3, -1, -1, -1),
            prompt=[source_prompt, source_prompt, target_prompt],
        ).images[2]
        return image.convert("RGB")

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((self.resolution, self.resolution), self.main.RESIZE_TYPE)
