"""MMaDA-Parallel-A official TI2TI editing inference, in memory."""

import math
import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/MMaDA-Parallel/MMaDA-Parallel-A"
WEIGHTS = ROOT / "third_party/MMaDA-Parallel/models/MMaDA-Parallel-A"
SYSTEM_PROMPT = "Generate an image applying the following editing instruction based on the original image."


class Backend:
    steps = 64
    resolution = 512
    text_steps = 128
    text_gen_length = 256

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        from transformers import AutoTokenizer
        from diffusers import VQModel
        from model import LLaDAForMultiModalGeneration
        from generators.parallel_generator import generate_ti2ti
        from utils.image_utils import (
            add_break_line, calculate_vq_params, decode_vq_to_image,
            encode_img_with_breaks,
        )
        from utils.prompt_utils import generate_text_image_to_text_image_prompt

        if not (WEIGHTS / "config.json").is_file():
            raise FileNotFoundError(f"Download MMaDA-Parallel-A weights to {WEIGHTS}")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS), trust_remote_code=True, local_files_only=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(
            str(WEIGHTS), torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True,
        ).eval()
        self.vqvae = VQModel.from_pretrained(str(WEIGHTS), subfolder="vqvae", local_files_only=True).to(self.device).eval()
        self.vae_scale = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.generate_ti2ti = generate_ti2ti
        self.add_break_line = add_break_line
        self.calculate_vq_params = calculate_vq_params
        self.decode_vq_to_image = decode_vq_to_image
        self.encode_img_with_breaks = encode_img_with_breaks
        self.prompt = generate_text_image_to_text_image_prompt
        self.text_vocab_size = getattr(self.model.config, "text_vocab_size", 126356)
        self.codebook_size = getattr(self.model.config, "codebook_size", 8192)

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((512, 512), Image.Resampling.BICUBIC)

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        torch.manual_seed(seed)
        prompt, uncon = self.prompt(task["instruction"], SYSTEM_PROMPT)
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        uncon_ids = self.tokenizer(uncon)["input_ids"]
        image_tokens = self.encode_img_with_breaks(source, self.vqvae)
        conditional = prompt_ids[:-1] + image_tokens + prompt_ids[-1:]
        uncond_text = uncon_ids[:-1] + image_tokens + uncon_ids[-1:]
        seq_len, newline_every, grid_h, grid_w = self.calculate_vq_params(512, 512, self.vae_scale)
        masked = self.add_break_line([126336] * seq_len, grid_h, grid_w, new_number=126084)
        end_ids = self.tokenizer("</answer>", add_special_tokens=False).input_ids
        suffix = [126354, 126349] + masked + [126350] + [126336] * self.text_gen_length + end_ids
        image_start = len(conditional) + 2
        text_start = image_start + len(masked) + 1
        generator = torch.Generator(device=self.device).manual_seed(seed)
        codes, _ = self.generate_ti2ti(
            model=self.model,
            input_ids=torch.tensor(conditional + suffix, device=self.device).unsqueeze(0),
            image_start=image_start, text_start=text_start,
            text_end=text_start + self.text_gen_length,
            seq_len=seq_len, newline_every=newline_every,
            text_steps=self.text_steps, text_gen_length=self.text_gen_length,
            text_block_length=32, timesteps=self.steps,
            temperature=1.0, text_temperature=0.0, cfg_scale=0.0, cfg_img=4.0,
            uncon_text=torch.tensor(uncond_text, device=self.device).unsqueeze(0),
            uncon_image=torch.tensor(prompt_ids, device=self.device).unsqueeze(0),
            tokenizer=self.tokenizer, remasking="low_confidence",
            noise_schedule=lambda t: torch.cos(t * math.pi / 2), generator=generator,
            text_vocab_size=self.text_vocab_size, codebook_size=self.codebook_size,
        )
        codes_tensor = torch.tensor(codes, dtype=torch.long, device=self.device).unsqueeze(0)
        return self.decode_vq_to_image(
            codes_tensor, None, str(WEIGHTS), 512, 512, self.vqvae,
        ).convert("RGB")
