"""Offline image metrics for Inter-Edit and MagicBrush benchmark outputs.

Protocols follow the authors' public implementations:
https://github.com/Delong-liu-bupt/Inter-Edit/blob/main/evaluation/objective_metrics.py
https://github.com/OSU-NLP-Group/MagicBrush/blob/main/evaluation/image_eval.py
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.mask_utils import normalize_edit_mask_image
from src.pie_bench_metrics import DEFAULT_PIE_METRICS, PIE_METRICS


DEFAULT_METRICS = {
    "inter_edit": ("lpips", "alpha-in", "alpha-out", "boundary"),
    "magicbrush": ("l1", "l2", "clip-i", "dino", "clip-t"),
    "pie_bench": DEFAULT_PIE_METRICS,
}
DIRECTIONS = {
    "l1": "lower", "l2": "lower", "clip-i": "higher", "dino": "higher",
    "clip-t": "higher", "clip-t-oracle": "higher", "S_global_LPIPS": "lower",
    "S_in_AlphaCLIP": "higher", "S_out_AlphaCLIP": "higher", "BSS_Laplacian": "lower",
}


def parse_metrics(value, benchmark):
    names = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = set(PIE_METRICS) if benchmark == "pie_bench" else set(DEFAULT_METRICS[benchmark]) | {"l1", "l2"}
    if not names or len(set(names)) != len(names) or set(names) - allowed:
        raise ValueError(f"Choose distinct metrics from {sorted(allowed)}")
    return names


class BenchmarkMetrics:
    def __init__(self, benchmark, names, device="cuda:0", alpha_clip_ckpt=None,
                 alpha_clip_model="ViT-L/14", model_cache_dir=None,
                 dino_repo=None, dino_weights=None, metric_resolution=None):
        self.benchmark, self.names = benchmark, tuple(names)
        self.device = torch.device(device)
        self.metric_resolution = metric_resolution
        self.clip = self.dino = self.alpha = self.lpips = None
        cache = str(model_cache_dir) if model_cache_dir else None
        if model_cache_dir:
            torch.hub.set_dir(str(Path(model_cache_dir) / "torch_hub"))
        self.protocol = {
            "requested": list(names), "directions": DIRECTIONS,
            "reference_metric_set": list(DEFAULT_METRICS[benchmark]),
            "includes_all_reference_metrics": set(DEFAULT_METRICS[benchmark]) <= set(names),
            "aggregation": "arithmetic mean of per-image scores",
            "metric_resolution": metric_resolution or "native",
            "mask": "original benchmark annotation, never the dilated generation mask",
            "pixel_range": [0, 1], "l2_definition": "mean squared error, without square root",
        }
        if "lpips" in names:
            try:
                import lpips
            except ImportError as error:
                raise ImportError("LPIPS requires lpips; see requirements-benchmark.txt") from error
            self.lpips = lpips.LPIPS(net="alex").to(self.device).eval()
            self.protocol["lpips_backbone"] = "alex"
        if {"alpha-in", "alpha-out"} & set(names):
            if alpha_clip_ckpt is None or not Path(alpha_clip_ckpt).is_file():
                raise FileNotFoundError("AlphaCLIP metrics require --alpha-clip-ckpt with the authors' visual weights")
            try:
                import alpha_clip
            except ImportError as error:
                raise ImportError("Install AlphaCLIP from https://github.com/SunzeY/AlphaCLIP for regional metrics") from error
            self.alpha, self.alpha_transform = alpha_clip.load(
                alpha_clip_model, alpha_vision_ckpt_pth=str(alpha_clip_ckpt),
                device=str(self.device), download_root=cache,
            )
            self.alpha.eval()
            if self.device.type == "cpu":
                self.alpha.float()
            size = 336 if "336" in alpha_clip_model else 224
            self.alpha_mask_transform = transforms.Compose([
                transforms.ToTensor(), transforms.Resize((size, size)),
                transforms.Normalize(0.5, 0.26),
            ])
            self.protocol.update(alpha_clip_model=alpha_clip_model, alpha_clip_ckpt=str(Path(alpha_clip_ckpt).resolve()))
        if {"clip-i", "clip-t"} & set(names):
            try:
                import clip
            except ImportError as error:
                raise ImportError("CLIP metrics require OpenAI CLIP; see requirements-benchmark.txt") from error
            self.clip_api = clip
            self.clip, self.clip_transform = clip.load("ViT-B/32", device=self.device, download_root=cache)
            self.clip.eval()
            self.protocol.update(clip_model="ViT-B/32", clip_t_text="target global caption; never the edit instruction")
        if "dino" in names:
            local = dino_repo is not None
            # The upstream repo/ref is fixed here. Avoid Torch Hub's GitHub
            # branches/tags API validation, which can hit anonymous rate limits
            # (and trigger an Authorization KeyError in some Torch versions).
            self.dino = torch.hub.load(
                str(dino_repo) if local else "facebookresearch/dino:main", "dino_vits16",
                source="local" if local else "github", pretrained=dino_weights is None,
                **({} if local else {"trust_repo": True, "skip_validation": True}),
            )
            if dino_weights:
                state = torch.load(dino_weights, map_location="cpu", weights_only=True)
                state = state.get("teacher", state.get("state_dict", state))
                state = {key.removeprefix("module.").removeprefix("backbone."): value
                         for key, value in state.items()}
                self.dino.load_state_dict(state, strict=True)
            self.dino.to(self.device).eval()
            self.dino_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.protocol.update(dino_model="DINO v1 ViT-S/16", dino_repo=str(dino_repo or "facebookresearch/dino:main"),
                                 dino_weights=str(dino_weights) if dino_weights else "official pretrained")
        if benchmark == "inter_edit":
            self.protocol.update(
                unavailable_metrics={"S_ref_AlphaCLIP": "public source/GT/mask release has no extra reference-object image"},
                boundary_definition="absolute difference of mean Sobel gradient magnitudes in inner/outer mask rings",
                boundary_kernel=11, empty_boundary_policy="official zero",
                # Preserve the published script's naming, despite its Sobel implementation.
                boundary_output_name="BSS_Laplacian",
            )

    @staticmethod
    def _tensor(image):
        return transforms.ToTensor()(image.convert("RGB")).unsqueeze(0)

    @staticmethod
    def _cosine(left, right):
        return float(F.cosine_similarity(left.float().reshape(1, -1), right.float().reshape(1, -1)).item())

    def _alpha_feature(self, image, mask):
        dtype = next(self.alpha.parameters()).dtype
        pixels = self.alpha_transform(image).unsqueeze(0).to(self.device, dtype=dtype)
        alpha = self.alpha_mask_transform(mask.astype(np.uint8) * 255).unsqueeze(0).to(self.device, dtype=dtype)
        return self.alpha.visual(pixels, alpha)

    @torch.inference_mode()
    def score(self, prediction, target, source=None, mask=None, caption=None):
        prediction, target = prediction.convert("RGB"), target.convert("RGB")
        if self.benchmark == "inter_edit":
            size = (self.metric_resolution, self.metric_resolution) if self.metric_resolution else source.size
            source = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            target = target.resize(size, Image.Resampling.LANCZOS)
            prediction = prediction.resize(size, Image.Resampling.LANCZOS)
            mask = normalize_edit_mask_image(mask).resize(size, Image.Resampling.NEAREST)
        else:
            size = (self.metric_resolution, self.metric_resolution) if self.metric_resolution else target.size
            # The official MagicBrush evaluator resizes predictions to GT size.
            target = target.resize(size, Image.Resampling.BICUBIC)
            prediction = prediction.resize(size, Image.Resampling.BICUBIC)
        scores = {}
        if {"l1", "l2"} & set(self.names):
            pred_tensor, gt_tensor = self._tensor(prediction), self._tensor(target)
            if "l1" in self.names:
                scores["l1"] = float(F.l1_loss(pred_tensor, gt_tensor).item())
            if "l2" in self.names:
                scores["l2"] = float(F.mse_loss(pred_tensor, gt_tensor).item())
        if self.lpips is not None:
            scores["S_global_LPIPS"] = float(self.lpips(
                self._tensor(prediction).to(self.device) * 2 - 1,
                self._tensor(target).to(self.device) * 2 - 1,
            ).item())
        if self.alpha is not None:
            region = np.array(mask, dtype=np.uint8) > 128
            if "alpha-in" in self.names:
                scores["S_in_AlphaCLIP"] = self._cosine(
                    self._alpha_feature(prediction, region), self._alpha_feature(target, region))
            if "alpha-out" in self.names:
                scores["S_out_AlphaCLIP"] = self._cosine(
                    self._alpha_feature(prediction, ~region), self._alpha_feature(source, ~region))
        if "boundary" in self.names:
            import cv2

            region = (np.array(mask, dtype=np.uint8) > 128).astype(np.uint8)
            kernel = np.ones((11, 11), dtype=np.uint8)
            inside = (region - cv2.erode(region, kernel)) > 0
            outside = (cv2.dilate(region, kernel) - region) > 0
            gray = np.array(prediction.convert("L"))
            gradient = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
                                cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
            scores["BSS_Laplacian"] = (float(abs(gradient[inside].mean() - gradient[outside].mean()))
                                       if inside.any() and outside.any() else 0.0)
        if self.clip is not None:
            pred_feature = self.clip.encode_image(self.clip_transform(prediction).unsqueeze(0).to(self.device))
            gt_feature = self.clip.encode_image(self.clip_transform(target).unsqueeze(0).to(self.device))
            if "clip-i" in self.names:
                scores["clip-i"] = self._cosine(pred_feature, gt_feature)
            if "clip-t" in self.names:
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError("CLIP-T requires this target image's global caption")
                text_feature = self.clip.encode_text(self.clip_api.tokenize(caption).to(self.device))
                scores["clip-t"] = self._cosine(pred_feature, text_feature)
                scores["clip-t-oracle"] = self._cosine(gt_feature, text_feature)
        if self.dino is not None:
            pred_feature = self.dino(self.dino_transform(prediction).unsqueeze(0).to(self.device))
            gt_feature = self.dino(self.dino_transform(target).unsqueeze(0).to(self.device))
            scores["dino"] = self._cosine(pred_feature, gt_feature)
        if not all(np.isfinite(value) for value in scores.values()):
            raise ValueError(f"Non-finite metric result: {scores}")
        return scores
