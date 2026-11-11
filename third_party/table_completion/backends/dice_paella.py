"""DICE's released Paella inversion/editing path for captioned PIE-Bench cases.

The DICE implementation requires separate source and target *descriptions*.
The local MagicBrush and Inter-Edit test releases provide instructions only,
so this backend intentionally rejects those tasks rather than using target GT.
"""

import importlib.util
import sys
from pathlib import Path

import open_clip
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/DICE/image_editing/paella"
CHECKPOINTS = REPO / "checkpoints"
NEGATIVE = "low quality, low resolution, bad image, blurry, blur"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Backend:
    steps = 32
    resolution = 512

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(ROOT / "third_party/DICE/pytorch-tools"))
        self.device = torch.device(device)
        paella_mod = _module("dice_paella_model", REPO / "models/paella.py")
        vq_mod = _module("dice_paella_vq", REPO / "src/vqgan.py")
        attention_mod = _module("dice_paella_attention", REPO / "utils/alter_attention.py")
        sampling_mod = _module("dice_paella_sampling", REPO / "utils/sampling.py")
        self.sample_gumbel = sampling_mod.sample_gumbel
        self.generate_mask_schedule = sampling_mod.generate_mask_schedule
        self.sample_noised_tokens = sampling_mod.sample_noised_tokens
        self.get_lambda_schedule = sampling_mod.get_lambda_schedule

        self.vq = vq_mod.VQModel().to(self.device).eval()
        self.vq.load_state_dict(torch.load(CHECKPOINTS / "paella/vqgan_f4.pt", map_location="cpu", weights_only=False))
        self.model = paella_mod.Paella(byt5_embd=2560).to(self.device).eval()
        self.model.load_state_dict(torch.load(CHECKPOINTS / "paella/paella_v3.pt", map_location="cpu", weights_only=False))
        attention_mod.replace_attention_layers(self.model)
        self.model.to(self.device).eval()

        self.t5_tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINTS / "byt5-xl"), local_files_only=True)
        self.t5 = T5EncoderModel.from_pretrained(
            str(CHECKPOINTS / "byt5-xl"), local_files_only=True,
        ).to(self.device).eval()
        self.clip, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=str(CHECKPOINTS / "clip-h/open_clip_model.safetensors"),
        )
        self.clip = self.clip.to(self.device).eval()
        for module in (self.vq, self.model, self.t5, self.clip):
            module.requires_grad_(False)
        self.preprocess = T.Compose([T.Resize(512), T.CenterCrop(512)])
        self.latent_shape = (1, 128, 128)

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return self.preprocess(source)

    @torch.inference_mode()
    def _condition(self, prompt):
        tokens = self.t5_tokenizer([prompt], padding="longest", max_length=768,
                                   truncation=True, return_tensors="pt").input_ids.to(self.device)
        byt5 = self.t5(input_ids=tokens).last_hidden_state
        clip = self.clip.encode_text(open_clip.tokenize([prompt]).to(self.device))
        return {"byt5": byt5, "clip": clip, "clip_image": None}

    @torch.inference_mode()
    def _invert(self, x0, condition, uncond):
        dev = self.device
        steps, shape = self.steps, self.latent_shape
        t_list = torch.linspace(0, 0.9, steps + 1)
        temperatures = torch.linspace(0.2, 1.2, steps)
        mask_schedule = self.generate_mask_schedule(steps, shape, max_t=0.9, device=dev)
        noise = torch.randint(0, 8192, shape, device=dev)
        noised = self.sample_noised_tokens(x0, mask_schedule, shape, init_noise=noise, device=dev)
        weights = torch.ones(condition["byt5"].shape[1], device=dev)
        weights[-4:] = 0.4
        weights[:-4] = 1.2
        clean_logits = self.model(x0, torch.zeros(shape[0], device=dev), **condition, attn_weights=weights)
        residuals = []
        current = None
        for i, tv in enumerate(t_list[1:steps], 1):
            t = torch.ones(shape[0], device=dev) * tv
            current, _ = noised[i]
            current = current.to(dev).long()
            predicted = self.model(current, t, **condition, attn_weights=weights)
            predicted = predicted * 10.0 + self.model(current, t, **uncond) * -9.0
            residuals.append(clean_logits - predicted.div(temperatures[i]))
        return current, residuals, mask_schedule, noise

    @torch.inference_mode()
    def _edit(self, current, residuals, mask_schedule, noise, condition, uncond):
        dev = self.device
        steps, shape = self.steps, self.latent_shape
        t_list = torch.linspace(0.9, 0, steps + 1)
        temperatures = torch.linspace(1.2, 0.2, steps)
        lambdas = self.get_lambda_schedule(0.7, steps, "None")
        weights = torch.ones(condition["byt5"].shape[1], device=dev)
        weights[-4:] = 0.4
        weights[:-4] = 1.2
        for i, tv in enumerate(t_list[1:steps]):
            t = torch.ones(shape[0], device=dev) * tv
            logits = self.model(current, t, **condition, attn_weights=weights)
            logits = logits * 10.0 + self.model(current, t, **uncond) * -9.0
            logits = (logits.div(temperatures[i]) + residuals[i] * lambdas[i]
                      + self.sample_gumbel(logits.shape, device=dev) * (1 - lambdas[i]))
            current = logits.argmax(dim=1)
            if i + 2 < steps:
                current = (current * (1 - mask_schedule[i + 2])
                           + noise * mask_schedule[i + 2]).long()
        return current

    @torch.inference_mode()
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        if not task.get("source_caption") or not task.get("caption"):
            raise ValueError("DICE Paella requires source and target captions")
        torch.manual_seed(seed)
        source_tensor = T.ToTensor()(source).unsqueeze(0).to(self.device)[:, :3]
        src_cond = self._condition(task["source_caption"])
        tgt_cond = self._condition(task["caption"])
        uncond = self._condition(NEGATIVE)
        with torch.autocast(device_type="cuda"), torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True,
        ):
            latents = self.vq.encode(source_tensor, quantize=True)[2]
            current, residuals, mask_schedule, noise = self._invert(latents, src_cond, uncond)
            edited = self._edit(current.long(), residuals[::-1], mask_schedule[::-1],
                                noise, tgt_cond, uncond)
            result = self.vq.decode_indices(edited)
        return T.ToPILImage()(result[0].float().clamp(0, 1)).convert("RGB")
