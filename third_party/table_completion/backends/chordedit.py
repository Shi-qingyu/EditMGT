"""Official ChordEdit with its published one-step default configuration."""

import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/ChordEdit"
WEIGHTS = REPO / "models/sd-turbo"


class Backend:
    name = "chordedit"
    steps = 1
    resolution = 512

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        from pipeline_chord import ChordEditPipeline
        from run_pie_bench import DEFAULT_EDIT_CONFIG

        components = {
            name: str(WEIGHTS / dirname)
            for name, dirname in {
                "unet_path": "unet", "scheduler_path": "scheduler",
                "text_encoder_path": "text_encoder", "tokenizer_path": "tokenizer",
                "vae_path": "vae",
            }.items()
        }
        if not (WEIGHTS / "unet/config.json").is_file():
            raise FileNotFoundError(f"Download stabilityai/sd-turbo to {WEIGHTS}")
        self.device = torch.device(device)
        self.pipe = ChordEditPipeline.from_local_weights(
            component_paths=components,
            default_edit_config=dict(DEFAULT_EDIT_CONFIG),
            device=self.device,
            torch_dtype=torch.float32,
            compute_dtype=torch.float32,
            image_size=self.resolution,
            use_center_crop=True,
            use_attention_mask=False,
            use_safety_checker=False,
        )

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        # PIE supplies both descriptive captions. The two instruction datasets
        # do not; use an empty source prompt and the provided edit instruction.
        source_prompt = task.get("source_caption") or ""
        target_prompt = task.get("caption") or task["instruction"]
        result = self.pipe(
            image=source, source_prompt=source_prompt,
            target_prompt=target_prompt, seed=seed, output_type="pil",
        )
        return result.images[0].convert("RGB")

    @staticmethod
    def prepare_source(source: Image.Image) -> Image.Image:
        # Official ChordEdit center-crops and resizes inside its pipeline.
        return source
