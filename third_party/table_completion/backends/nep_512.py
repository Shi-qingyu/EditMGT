"""NEP-512 official next-edit-token generator with benchmark edit masks."""

import os
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/NEP"
WEIGHTS = REPO / "pretrained_models"


class Backend:
    resolution = 512
    requires_edit_mask = True

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        from autoregressive.models.gpt import GPT_models
        from autoregressive.models.generate import generate_magicbrush_by_next_editing_token_prediction
        from tokenizer.tokenizer_image.vq_model import VQ_models
        from language.t5 import T5Embedder

        self.device = torch.device(device)
        checkpoint = torch.load(WEIGHTS / "NEP_512.pt", map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        self.image_size = args.image_size
        self.latent_size = args.image_size // args.downsample_size
        self.patch_size = args.downsample_size
        self.codebook_embed_dim = args.codebook_embed_dim
        # The released checkpoint says GPT-XL-RAR, while this repository names
        # the matching RTransformer constructor R-GPT-XL.
        self.model = GPT_models["R-GPT-XL"](
            vocab_size=args.vocab_size, block_size=self.latent_size ** 2,
            num_classes=args.num_classes, cls_token_num=args.cls_token_num,
            model_type=args.gpt_type, resid_dropout_p=args.dropout_p,
            ffn_dropout_p=args.dropout_p, token_dropout_p=args.token_dropout_p,
            set_max_len=args.set_max_len, mask_in_context=args.mask_in_context,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        del checkpoint
        self.vq_model = VQ_models["VQ-16"](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim,
        ).to(self.device).eval()
        vq_checkpoint = torch.load(WEIGHTS / "vq_ds16_t2i.pt", map_location="cpu", weights_only=False)
        self.vq_model.load_state_dict(vq_checkpoint["model"])
        del vq_checkpoint
        self.t5 = T5Embedder(
            str(self.device), dir_or_name="flan-t5-xl", local_cache=True,
            cache_dir=str(WEIGHTS), model_max_length=args.t5_feature_max_len,
        )
        self.generate_nep = generate_magicbrush_by_next_editing_token_prediction
        self.cfg_scale = args.cfg_scale
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.mask_in_context = args.mask_in_context
        self.scratch = tempfile.TemporaryDirectory(prefix=f"nep-{os.getpid()}-", dir="/tmp")
        self.scratch_path = Path(self.scratch.name)

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((512, 512), Image.Resampling.BILINEAR)

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        mask = task["_edit_mask"]
        if mask.getbbox() is None:
            return source.copy()
        torch.manual_seed(seed)
        embeddings, attention_mask = self.t5.get_text_embeddings([task["instruction"]])
        # Mirror the repository's default left-padding path.
        lengths = attention_mask.sum(dim=-1)
        shifted = [torch.cat((embedding[int(n):], embedding[:int(n)]))
                   for embedding, n in zip(embeddings, lengths)]
        attention_mask = attention_mask.flip(dims=[-1])
        condition = torch.stack(shifted) * attention_mask[:, :, None]
        white_path = self.scratch_path / "edit_white.png"
        gray_path = self.scratch_path / "edit.png"
        result_path = self.scratch_path / "result.png"
        mask.save(white_path)
        Image.new("RGB", mask.size, (0, 0, 0)).save(gray_path)
        self.generate_nep(
            self.model, condition, self.latent_size ** 2, attention_mask,
            cfg_scale=self.cfg_scale, temperature=self.temperature,
            top_k=self.top_k, top_p=self.top_p, sample_logits=True,
            vq_model=self.vq_model,
            qzshape=[1, self.codebook_embed_dim, self.latent_size, self.latent_size],
            image_size=self.image_size, patch_size=self.patch_size,
            source_img_path=[source], edit_region_mask_path=str(white_path),
            save_output_img_path=str(result_path), mask_in_context=self.mask_in_context,
        )
        with Image.open(result_path) as result:
            return result.convert("RGB")
