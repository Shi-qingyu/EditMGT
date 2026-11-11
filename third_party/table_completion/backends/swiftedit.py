"""SwiftEdit official one-step inversion and editing with in-memory input."""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/SwiftEdit"
WEIGHTS = REPO / "swiftedit_weights"
SD_TURBO = ROOT / "third_party/ChordEdit/models/sd-turbo"
SD21 = REPO / "models/sd21base"
IP_ADAPTER = REPO / "models/ip_adapter"


class Backend:
    name = "swiftedit"
    steps = 1
    resolution = 512

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        # The shared benchmark runner already imports EditMGT's `src`
        # namespace. Let the unmodified SwiftEdit `models.py` resolve its
        # own `src.*` modules from the cloned repository as well.
        import src
        src.__path__.append(str(REPO / "src"))
        from models import AuxiliaryModel, IPSBV2Model, InverseModel
        from models import MaskController, tokenize_captions

        for path in (WEIGHTS / "inverse_ckpt-120k", WEIGHTS / "sbv2_0.5",
                     WEIGHTS / "ip_adapter_ckpt-90k/ip_adapter.bin",
                     SD_TURBO / "unet/config.json", SD21 / "vae/config.json",
                     IP_ADAPTER / "models/image_encoder/config.json"):
            if not path.exists():
                raise FileNotFoundError(path)

        self.device = torch.device(device)
        self.tokenize_captions = tokenize_captions
        self.MaskController = MaskController
        self.inverse = InverseModel(
            str(WEIGHTS / "inverse_ckpt-120k"), model_name=str(SD_TURBO),
            device=str(self.device),
        )
        self.aux = AuxiliaryModel(
            model_name=str(SD21), image_encoder_path=str(IP_ADAPTER),
            device=str(self.device),
        )
        self.ip = IPSBV2Model(
            str(WEIGHTS / "sbv2_0.5"),
            str(WEIGHTS / "ip_adapter_ckpt-90k/ip_adapter.bin"),
            self.aux, device=str(self.device), with_ip_mask_controller=True,
        )

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        # In-memory adaptation of the official infer.py edit_image() function.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        source_prompt = task.get("source_caption") or ""
        target_prompt = task.get("caption") or task["instruction"]
        mid_timestep = torch.tensor([500], dtype=torch.int64, device=self.device)
        image = to_tensor(source).unsqueeze(0).to(self.device) * 2 - 1
        latents = self.inverse.vae.encode(
            image.to(self.inverse.weight_dtype)
        ).latent_dist.sample()
        latents = latents * self.inverse.vae.config.scaling_factor
        doubled = torch.cat([latents] * 2, dim=0)
        input_ids = self.tokenize_captions(
            self.inverse.tokenizer, [source_prompt, target_prompt]
        ).to(self.device)
        hidden = self.inverse.text_encoder(input_ids)[0].to(
            dtype=self.inverse.weight_dtype
        )
        predicted = self.inverse.unet_inverse(
            doubled, mid_timestep, hidden
        ).sample.to(self.device, dtype=self.inverse.weight_dtype)
        inverted_source, inverted_target = predicted.chunk(2)
        diff = (inverted_source - inverted_target).abs_().mean(dim=[0, 1])
        maximum = (diff.mean() * 3.0).item()
        mask = (diff.clamp(0, maximum) / maximum > 0.5).to(diff.dtype)
        noise = self.ip.alpha_t * latents + self.ip.sigma_t * inverted_source
        controller = self.MaskController(
            mask, scale_text_hiddenstate=1, scale_ip_fg=0.2,
            scale_ip_bg=1,
        )
        self.ip.set_controller(controller, where=["mid_blocks", "up_blocks"])
        generated, _ = self.ip.gen_img(
            pil_image=source, prompts=[source_prompt, target_prompt], noise=noise,
        )
        return to_pil_image(generated[0].float().clamp(0, 1).cpu()).convert("RGB")

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((self.resolution, self.resolution))
