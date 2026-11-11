"""Sparse-LaViDa official 1024px image-edit demo configuration with cache."""

import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/Sparse-LaViDa"
WEIGHTS = REPO / "models/sparse-lavida"
VISION = REPO / "models/siglip-so400m-patch14-384"
VQVAE = REPO / "models/Meissonic"


class Backend:
    name = "sparse_lavida"
    steps = 64
    resolution = 1024

    def __init__(self, device: str = "cuda"):
        os.environ["DEBUG_FIX_PADDING"] = "1"
        os.environ["NOT_ALWASY_DO_2DPOOL"] = "1"
        sys.path.insert(0, str(REPO))
        from llava.model.builder import load_pretrained_model
        from llava.model.language_model.llava_llada import LlavaLladaConfig
        from llava.eval.predict_t2i_edit import text_to_image

        for path in (WEIGHTS / "config.json", VISION / "config.json",
                     VQVAE / "vqvae/config.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.device = torch.device(device)
        config = LlavaLladaConfig.from_pretrained(str(WEIGHTS))
        config.mm_vqvae = str(VQVAE)
        vision_kwargs = {
            "mm_vision_tower": str(VISION), "mm_resampler_type": None,
            "mm_projector_type": "mlp2x_gelu", "mm_hidden_size": 1152,
            "use_mm_proj": True,
        }
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            str(WEIGHTS), None, "llava_llada", device_map=device,
            vision_kwargs=vision_kwargs, torch_dtype="bfloat16",
            customized_config=config,
        )
        self.model.eval()
        self.model.tie_weights()
        self.model.to(torch.bfloat16)
        self.model.requires_grad_(False)
        self.text_to_image = text_to_image
        self.quiet = open(os.devnull, "w", encoding="utf-8")
        self.config = yaml.safe_load((REPO / "llava/eval/1024_eval_edit.yaml").read_text())["config"]
        self.config.update(
            use_cache=True, cache_vq=True, truncate_vq=True,
            cache_prompt=True, cache_rounds=2, add_register=True,
            cache_buffer_round=3, cache_buffer_round_end=3,
            return_compute_positions=True, block_policy=2,
        )

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        with redirect_stdout(self.quiet), redirect_stderr(self.quiet):
            result = self.text_to_image(
                self.model, task["instruction"], edit_image=source,
                image_processor=self.image_processor, tokenizer=self.tokenizer,
                **self.config, image_resolution=self.resolution,
                n_tokens=4096, is_legacy=False,
            )
        return result.convert("RGB")

    def prepare_source(self, source: Image.Image) -> Image.Image:
        from llava.mm_utils import resize_and_center_crop
        return resize_and_center_crop(source, self.resolution)
