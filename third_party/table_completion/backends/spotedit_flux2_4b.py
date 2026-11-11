"""SpotEdit FLUX.2 code adapted to the ungated four-step Klein 4B backbone."""

import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/SpotEdit"
WEIGHTS = REPO / "models/FLUX.2-klein-4B"


class Backend:
    name = "spotedit_flux2_4b"
    steps = 4
    resolution = 1024

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        from diffusers import Flux2KleinPipeline
        from FLUX2.flux2klein_spotedit import generate
        from FLUX2.flux2_spot_ultis import SpotEditConfig

        if not (WEIGHTS / "model_index.json").is_file():
            raise FileNotFoundError(WEIGHTS)
        self.device = torch.device(device)
        self.pipe = Flux2KleinPipeline.from_pretrained(
            str(WEIGHTS), torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        unpack = self.pipe._unpack_latents_with_ids
        self.pipe._unpack_latents_with_ids = lambda latents, ids, *_shape: unpack(latents, ids)
        self.generate_spot = generate
        # The released SpotEdit defaults to four full warm-up steps for its
        # 50-step Klein 9B example. The public Klein 4B backbone needs four
        # total steps, so one warm-up step enables selective token reuse.
        self.config = SpotEditConfig(threshold=0.4, initial_steps=1, reset_steps=[])

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        # SpotEdit's notebook holds one image per pipeline; its token metric
        # cache must be cleared when the pipeline is reused for a dataset.
        self.pipe.metric = None
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.generate_spot(
            self.pipe, image=source, prompt=task["instruction"],
            config=self.config, num_inference_steps=self.steps,
            guidance_scale=1.0, height=self.resolution, width=self.resolution,
            generator=generator,
        ).images[0].convert("RGB")

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
