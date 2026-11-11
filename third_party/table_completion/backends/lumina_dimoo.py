"""Lumina-DiMOO official 64-step image editing inference, in memory."""

import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/Lumina-DiMOO"
WEIGHTS = REPO / "models/Lumina-DiMOO"


class Backend:
    steps = 64
    resolution = 512

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        from transformers import AutoTokenizer
        from diffusers import VQModel
        from model import LLaDAForMultiModalGeneration
        from generators.image_to_image_generator import generate_i2i
        from utils.image_utils import (
            add_break_line, calculate_vq_params, decode_vq_to_image,
            encode_img_with_breaks,
        )
        from utils.prompt_utils import create_prompt_templates, generate_image_to_image_prompt

        if not (WEIGHTS / "config.json").is_file():
            raise FileNotFoundError(f"Download Lumina-DiMOO weights to {WEIGHTS}")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS), trust_remote_code=True, local_files_only=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(
            str(WEIGHTS), torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True,
        ).eval()
        self.vqvae = VQModel.from_pretrained(str(WEIGHTS), subfolder="vqvae", local_files_only=True).to(self.device).eval()
        self.vae_scale = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.generate_i2i = generate_i2i
        self.add_break_line = add_break_line
        self.calculate_vq_params = calculate_vq_params
        self.decode_vq_to_image = decode_vq_to_image
        self.encode_img_with_breaks = encode_img_with_breaks
        self.prompt = generate_image_to_image_prompt
        self.templates = create_prompt_templates()

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((512, 512), Image.Resampling.BICUBIC)

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        torch.manual_seed(seed)
        prompt, uncon, _ = self.prompt(task["instruction"], "edit_replace", self.templates)
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        uncon_ids = self.tokenizer(uncon)["input_ids"]
        image_tokens = self.encode_img_with_breaks(source, self.vqvae)
        conditional = prompt_ids[:-1] + image_tokens + prompt_ids[-1:]
        uncond_text = uncon_ids[:-1] + image_tokens + uncon_ids[-1:]
        seq_len, newline_every, grid_h, grid_w = self.calculate_vq_params(512, 512, self.vae_scale)
        masked = self.add_break_line([126336] * seq_len, grid_h, grid_w, new_number=126084)
        suffix = [126354, 126349] + masked + [126350, 126355]
        cond_tensor = torch.tensor(conditional + suffix, device=self.device).unsqueeze(0)
        text_tensor = torch.tensor(uncond_text, device=self.device).unsqueeze(0)
        image_tensor = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        codes = self.generate_i2i(
            self.model, cond_tensor, seq_len=seq_len, newline_every=newline_every,
            timesteps=self.steps, temperature=1.0, cfg_scale=2.5, cfg_img=4.0,
            uncon_text=text_tensor, uncon_image=image_tensor,
            code_start=len(conditional) + 2, generator=generator,
        )
        return self.decode_vq_to_image(
            codes, "", str(WEIGHTS), 512, 512, self.vqvae,
        ).convert("RGB")
