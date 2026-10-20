"""PIE-Bench preservation and text alignment metrics, scored per image.

Definitions follow the published evaluator, including its blacked-out regions,
SqueezeNet LPIPS, CLIP ViT-L/14 scores in [0,100], and last-layer DINO ViT-B/8
key self-similarity distance. No edited ground-truth images are assumed.
https://github.com/cure-lab/PnPInversion/blob/main/evaluation/matrics_calculator.py
"""
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


DEFAULT_PIE_METRICS = (
    "structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
    "ssim_unedit_part", "clip_similarity_source_image", "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
)
PIE_METRICS = tuple(name + suffix for name in ("psnr", "lpips", "mse", "ssim", "structure_distance")
                    for suffix in ("", "_unedit_part", "_edit_part")) + DEFAULT_PIE_METRICS[-3:]
PIE_DIRECTIONS = {name: ("higher" if name.startswith(("psnr", "ssim", "clip_")) else "lower")
                  for name in PIE_METRICS}


def pie_metric_summary(rows):
    """Preserve exact-match infinite PSNR and exclude undefined empty regions."""
    values = defaultdict(list)
    for row in rows:
        for name, value in row["metrics"].items():
            values[name].append(value)
    result = {}
    for name, raw in sorted(values.items()):
        valid = [float(value) for value in raw if value is not None]
        infinite = any(math.isinf(value) for value in valid)
        result[name] = {
            "mean": "inf" if infinite else float(np.mean(valid)) if valid else None,
            "std": None if infinite or not valid else float(np.std(valid)),
            "count": len(valid), "undefined_count": len(raw) - len(valid),
        }
    return result


class PIEBenchMetrics:
    def __init__(self, names, device="cuda:0", model_cache_dir=None,
                 dino_repo=None, dino_weights=None, metric_resolution=None):
        self.names, self.device = tuple(names), torch.device(device)
        self.resolution = metric_resolution or 512
        self.lpips = self.clip = self.dino = self.ssim = None
        cache = Path(model_cache_dir) if model_cache_dir is not None else None
        if cache:
            torch.hub.set_dir(str(cache / "torch_hub"))
        if any(name.startswith("lpips") for name in names):
            import lpips

            self.lpips = lpips.LPIPS(net="squeeze").to(self.device).eval().requires_grad_(False)
        if any(name.startswith("ssim") for name in names):
            from torchmetrics.functional.image import structural_similarity_index_measure

            self.ssim = structural_similarity_index_measure
        if any(name.startswith("clip_") for name in names):
            from transformers import CLIPModel, CLIPProcessor

            clip_cache = str(cache / "huggingface") if cache else None
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=clip_cache)
            self.clip.to(self.device).eval().requires_grad_(False)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=clip_cache)
        if any(name.startswith("structure_distance") for name in names):
            local = dino_repo is not None
            self.dino = torch.hub.load(
                str(dino_repo) if local else "facebookresearch/dino:main", "dino_vitb8",
                source="local" if local else "github", pretrained=dino_weights is None,
                **({} if local else {"trust_repo": True, "skip_validation": True}),
            )
            if dino_weights is not None:
                state = torch.load(dino_weights, map_location="cpu", weights_only=True)
                state = state.get("teacher", state.get("state_dict", state))
                state = {key.removeprefix("module.").removeprefix("backbone."): value for key, value in state.items()}
                self.dino.load_state_dict(state, strict=True)
            self.dino.to(self.device).eval().requires_grad_(False)
            self.dino_transform = transforms.Compose([
                transforms.Resize(224, max_size=480, antialias=True),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.protocol = {
            "requested": list(names), "directions": PIE_DIRECTIONS,
            "reference_metric_set": list(DEFAULT_PIE_METRICS),
            "includes_all_reference_metrics": set(DEFAULT_PIE_METRICS) <= set(names),
            "reference_evaluator": "https://github.com/cure-lab/PnPInversion/tree/main/evaluation",
            "preservation_reference": "source image, not an edited GT", "metric_resolution": self.resolution,
            "resize": "bicubic RGB, nearest-neighbor annotation mask",
            "mask": "official 512x512 zero-based RLE with edited outer border",
            "regional_metrics": "zero excluded pixels, then compute over the whole image; no area renormalization",
            "lpips_backbone": "squeeze", "lpips_input_range": [-1, 1],
            "pixel_input_range": [0, 1], "psnr_data_range": 1.0,
            "clip_model": "openai/clip-vit-large-patch14", "clip_score": "100 * max(cosine(image,text),0)",
            "clip_edit_region": "black outside annotation; target global caption with [] removed",
            "structure_model": "DINO v1 ViT-B/8, last-block key self-similarity including CLS",
            "structure_input_range": "float [0,255] before Resize(224,max_size=480) and ImageNet normalization, as upstream",
            "structure_resize_antialias": True,
            "dino_repo": str(dino_repo or "facebookresearch/dino:main"),
            "dino_weights": str(dino_weights) if dino_weights else "official pretrained",
            "aggregation": "arithmetic mean of per-image raw scores; no paper/table scaling",
            "empty_region": "JSON null, excluded from mean with undefined_count recorded",
            "perfect_match_psnr": "JSON/CSV string inf; mean inf and std null when any valid PSNR is infinite",
        }

    def _tensor255(self, image):
        pixels = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _clip_score(self, image, caption):
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("PIE CLIP scores require the corresponding global image caption")
        inputs = self.clip_processor(text=[caption], images=[image], return_tensors="pt",
                                     padding=True, truncation=True, max_length=77).to(self.device)
        image_features = self.clip.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return float((100 * F.cosine_similarity(image_features.float(), text_features.float())).clamp(min=0).item())

    def _structure(self, pixels):
        captured = []
        handle = self.dino.blocks[-1].attn.qkv.register_forward_hook(
            lambda module, inputs, output: captured.append(output),
        )
        try:
            self.dino(self.dino_transform(pixels))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("DINO structure metric did not capture last-block Q/K/V")
        qkv = captured[0]
        keys = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1)[:, :, 1].float()
        products = keys @ keys.transpose(-2, -1)
        norms = keys.norm(dim=-1, keepdim=True)
        return products / (norms @ norms.transpose(-2, -1)).clamp_min(1e-8)

    @torch.inference_mode()
    def score(self, prediction, source, mask, source_caption=None, target_caption=None):
        size = (self.resolution, self.resolution)
        source = source.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        prediction = prediction.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        mask = mask.convert("L").resize(size, Image.Resampling.NEAREST)
        region = np.asarray(mask) > 0
        source255, prediction255 = self._tensor255(source), self._tensor255(prediction)
        masks = {"": None, "_edit_part": torch.from_numpy(region.copy()).to(self.device)[None, None],
                 "_unedit_part": torch.from_numpy((~region).copy()).to(self.device)[None, None]}
        scores = {}
        for name in self.names:
            if name == "clip_similarity_source_image":
                scores[name] = self._clip_score(source, source_caption)
                continue
            if name.startswith("clip_similarity_target_image"):
                image = prediction
                if name.endswith("_edit_part"):
                    if not region.any():
                        scores[name] = None
                        continue
                    image = Image.fromarray(np.where(region[..., None], np.asarray(prediction), 0).astype(np.uint8))
                scores[name] = self._clip_score(image, target_caption)
                continue
            suffix = next((suffix for suffix in ("_unedit_part", "_edit_part") if name.endswith(suffix)), "")
            base = name[:-len(suffix)] if suffix else name
            selected = masks[suffix]
            if selected is not None and not bool(selected.any()):
                scores[name] = None
                continue
            left255 = source255 if selected is None else source255 * selected
            right255 = prediction255 if selected is None else prediction255 * selected
            left, right = left255 / 255.0, right255 / 255.0
            if base in ("mse", "psnr"):
                error = float(F.mse_loss(left, right).item())
                scores[name] = error if base == "mse" else (-10 * math.log10(error) if error > 0 else "inf")
            elif base == "lpips":
                scores[name] = float(self.lpips(left * 2 - 1, right * 2 - 1).item())
            elif base == "ssim":
                scores[name] = float(self.ssim(left, right, data_range=1.0).item())
            elif base == "structure_distance":
                scores[name] = float(F.mse_loss(self._structure(left255), self._structure(right255)).item())
            else:
                raise ValueError(f"Unsupported PIE metric: {name}")
        if any(value is not None and value != "inf" and not math.isfinite(value) for value in scores.values()):
            raise ValueError(f"Non-finite PIE metric: {scores}")
        return scores
