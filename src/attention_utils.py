"""EditMGT instruction-to-generation attention analysis and visualization."""
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def num_attention_layers(transformer) -> int:
    return len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)


def _processor_for_layer(transformer, layer: int):
    double_count = len(transformer.transformer_blocks)
    if layer < 0 or layer >= num_attention_layers(transformer):
        raise IndexError(f"Attention layer {layer} is out of range")
    if layer < double_count:
        return transformer.transformer_blocks[layer].attn.processor
    return transformer.single_transformer_blocks[layer - double_count].attn.processor


def _conditioning_tokenizer(tokenizer, architecture: str):
    if not isinstance(tokenizer, (list, tuple)):
        return tokenizer
    # encode_prompt uses the second encoder's hidden states for CLIP+T5/Gemma.
    if "T5" in architecture or "Gemma" in architecture:
        return tokenizer[1]
    return tokenizer[0]


def _token_metadata(tokenizer, prompt: str, text_len: int):
    encoded = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=text_len,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    ids = encoded.input_ids[0]
    attention_mask = encoded.attention_mask[0].bool()
    special_mask = encoded.get("special_tokens_mask", torch.zeros_like(ids)).bool()
    valid = attention_mask & ~special_mask
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
    return ids, tokens, valid


def _keyword_positions(tokenizer, full_ids, valid_mask, keywords: Iterable[str]) -> List[int]:
    positions = []
    full_ids = full_ids.tolist()
    for keyword in keywords:
        keyword_ids = tokenizer(keyword, add_special_tokens=False).input_ids
        if keyword_ids and isinstance(keyword_ids[0], list):
            keyword_ids = keyword_ids[0]
        for start in range(len(full_ids) - len(keyword_ids) + 1):
            if full_ids[start:start + len(keyword_ids)] == keyword_ids:
                positions.extend(
                    i for i in range(start, start + len(keyword_ids)) if valid_mask[i]
                )
    return sorted(set(positions))


def extract_image_to_text_attention(
    transformer,
    tokenizer,
    prompt: str,
    layers: Optional[Iterable[int]] = None,
    keywords: Optional[Iterable[str]] = None,
    guidance_enabled: bool = True,
    include_per_head: bool = False,
) -> Dict:
    """Extract generation-image-query -> instruction-text-key spatial maps.

    Raw softmax weights are used for metrics. Head, token, and layer averaging is
    performed only after selecting the requested slices.
    """
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != 1:
            raise ValueError("Attention analysis currently supports one prompt at a time")
        prompt = prompt[0]
    layers = list(layers) if layers is not None else list(range(num_attention_layers(transformer)))
    keywords = list(keywords or [])
    working_tokenizer = _conditioning_tokenizer(tokenizer, transformer.text_encoder_architecture)

    result = {
        "direction": "generation_image_query_to_instruction_text_key",
        "layers": layers,
        "per_layer": {},
    }
    all_layer_maps = []
    keyword_layer_maps = []
    sequence_layout = None
    for layer in layers:
        processor = _processor_for_layer(transformer, layer)
        weights = processor.attention_weights
        if not weights or weights.get("image_to_text") is None:
            raise RuntimeError(f"Layer {layer} has no captured image_to_text attention")
        attention = weights["image_to_text"].float()  # [B,H,image query,text key], on CPU
        if sequence_layout is None:
            text_length = int(weights["text_len"])
            image_length = int(weights["image_len"])
            reference_length = int(weights.get("reference_len", 0))
            sequence_layout = {
                "text": {"start": 0, "end": text_length},
                "generation_image": {
                    "start": text_length,
                    "end": text_length + image_length,
                },
                "reference_image": {
                    "start": text_length + image_length,
                    "end": text_length + image_length + reference_length,
                },
                "reference_excluded_from_analysis_slice": True,
            }
        if guidance_enabled:
            if attention.shape[0] % 2:
                raise ValueError("CFG attention batch must contain unconditional and conditional halves")
            attention = attention[attention.shape[0] // 2:]
        if attention.shape[0] != 1:
            raise ValueError("Attention visualization currently supports batch size 1")

        target_grid_size = weights.get("target_grid_size")
        sparse_indices = weights.get("target_token_indices")
        if sparse_indices is not None:
            sparse_valid = weights["target_token_valid"]
            if guidance_enabled:
                sparse_indices = sparse_indices[sparse_indices.shape[0] // 2:]
                sparse_valid = sparse_valid[sparse_valid.shape[0] // 2:]
            grid_length = target_grid_size[0] * target_grid_size[1]
            spatial_attention = attention.new_zeros(1, attention.shape[1], grid_length, attention.shape[-1])
            spatial_attention[0].index_copy_(
                1, sparse_indices[0, sparse_valid[0]], attention[0, :, sparse_valid[0]]
            )
            attention = spatial_attention
            sequence_layout["generation_full_grid_size"] = list(target_grid_size)
            sequence_layout["generation_active_indices"] = sparse_indices[0, sparse_valid[0]].tolist()

        text_len = attention.shape[-1]
        ids, tokens, valid = _token_metadata(working_tokenizer, prompt, text_len)
        valid_positions = valid.nonzero(as_tuple=False).flatten().tolist()
        if not valid_positions:
            raise ValueError("Instruction contains no valid non-special tokens")
        keyword_positions = _keyword_positions(
            working_tokenizer, ids, valid, keywords
        ) if keywords else []

        head_averaged = attention[0].mean(dim=0)  # [image query,text key]
        all_map = head_averaged[:, valid_positions].mean(dim=-1)
        keyword_map = (
            head_averaged[:, keyword_positions].mean(dim=-1)
            if keyword_positions else None
        )
        per_token = {
            f"{position}:{tokens[position]}": head_averaged[:, position]
            for position in valid_positions
        }
        image_len = attention.shape[-2]
        grid_height, grid_width = target_grid_size or (math.isqrt(image_len), math.isqrt(image_len))
        if grid_height * grid_width != image_len:
            raise ValueError(f"Generation attention has non-square image length {image_len}")

        layer_data = {
            "all_tokens": all_map,
            "keyword": keyword_map,
            "per_token": per_token,
            "valid_token_positions": valid_positions,
            "keyword_token_positions": keyword_positions,
        }
        if include_per_head:
            layer_data["per_head"] = attention[0]
        result["per_layer"][str(layer)] = layer_data
        all_layer_maps.append(all_map)
        if keyword_map is not None:
            keyword_layer_maps.append(keyword_map)

    result.update({
        "token_grid_size": [grid_height, grid_width],
        "sequence_layout": sequence_layout,
        "tokens": tokens,
        "valid_token_positions": valid_positions,
        "keyword_token_positions": keyword_positions,
        "all_tokens": torch.stack(all_layer_maps).mean(dim=0),
        "keyword": torch.stack(keyword_layer_maps).mean(dim=0) if keyword_layer_maps else None,
    })
    return result


def attention_enrichment(attention: torch.Tensor, edit_mask: torch.Tensor) -> Dict[str, float]:
    attention = attention.detach().float().flatten().cpu()
    mask = edit_mask.detach().float()
    while mask.ndim > 2:
        mask = mask[0]
    side = math.isqrt(attention.numel())
    mask = F.interpolate(mask[None, None], size=(side, side), mode="nearest")[0, 0].flatten().cpu()
    area = mask.mean()
    mass = (attention * mask).sum() / attention.sum().clamp(min=1e-12)
    enrichment = mass / area.clamp(min=1e-12)
    return {
        "edit_area_ratio": area.item(),
        "attention_mass": mass.item(),
        "attention_enrichment": enrichment.item(),
    }


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    tensor = image.detach().float().cpu()
    while tensor.ndim > 3:
        tensor = tensor[0]
    if tensor.min() < 0:
        tensor = tensor.mul(0.5).add(0.5)
    array = tensor[:3].clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def save_attention_case(
    output_dir: str,
    sample_id: str,
    instruction: str,
    source,
    target,
    generated,
    pixel_edit_mask: torch.Tensor,
    analysis: Dict,
) -> Dict:
    """Write raw case assets, heatmaps/overlays, per-layer/token maps, and metadata."""
    import matplotlib.pyplot as plt

    case_dir = Path(output_dir) / sample_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_pil, target_pil, generated_pil = map(_as_pil, (source, target, generated))
    source_pil.save(case_dir / "source.png")
    target_pil.save(case_dir / "target.png")
    generated_pil.save(case_dir / "generated.png")
    mask = pixel_edit_mask.detach().float().cpu()
    while mask.ndim > 2:
        mask = mask[0]
    Image.fromarray(mask.ge(0.5).byte().mul(255).numpy()).save(case_dir / "gt_mask.png")

    def save_map(flat_map, name, directory=case_dir):
        if flat_map is None:
            return
        side = math.isqrt(flat_map.numel())
        raw = flat_map.detach().float().reshape(side, side).cpu().numpy()
        normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        plt.imsave(directory / f"attention_{name}.png", normalized, cmap="jet", vmin=0, vmax=1)
        source_array = np.asarray(source_pil.resize((mask.shape[1], mask.shape[0]))).astype(np.float32) / 255
        heat = plt.get_cmap("jet")(np.asarray(Image.fromarray((normalized * 255).astype(np.uint8)).resize(
            (source_array.shape[1], source_array.shape[0]), Image.Resampling.BILINEAR
        )) / 255.0)[..., :3]
        mask_array = mask.ge(0.5).numpy()
        eroded = F.max_pool2d((~torch.from_numpy(mask_array))[None, None].float(), 3, 1, 1)[0, 0].eq(0).numpy()
        boundary = mask_array ^ eroded
        overlay = source_array * 0.55 + heat * 0.45
        overlay[boundary] = np.array([1.0, 1.0, 1.0])
        plt.imsave(directory / f"overlay_source_{name}.png", overlay.clip(0, 1))

    save_map(analysis["all_tokens"], "all_tokens")
    save_map(analysis.get("keyword"), "keyword")
    per_layer_dir = case_dir / "attention_per_layer"
    per_layer_dir.mkdir(exist_ok=True)
    layer_metrics = {}
    for layer, layer_data in analysis["per_layer"].items():
        save_map(layer_data["all_tokens"], f"layer_{layer}", per_layer_dir)
        layer_metrics[layer] = attention_enrichment(layer_data["all_tokens"], mask)
    per_token_dir = case_dir / "attention_per_token"
    per_token_dir.mkdir(exist_ok=True)
    # Per-token maps from the layer-averaged representation.
    token_keys = list(next(iter(analysis["per_layer"].values()))["per_token"])
    for token_key in token_keys:
        token_map = torch.stack([
            data["per_token"][token_key] for data in analysis["per_layer"].values()
        ]).mean(0)
        safe_name = token_key.replace("/", "_").replace(" ", "_")
        save_map(token_map, safe_name, per_token_dir)

    metrics = attention_enrichment(analysis["all_tokens"], mask)
    metadata = {
        "sample_id": sample_id,
        "instruction": instruction,
        "attention_direction": analysis["direction"],
        "sequence_layout": analysis["sequence_layout"],
        "attention_tensor_shape": "[batch, heads, generation_queries, text_keys]",
        "tokens": analysis["tokens"],
        "valid_token_positions": analysis["valid_token_positions"],
        "selected_keyword_tokens": analysis["keyword_token_positions"],
        **metrics,
        "layers": layer_metrics,
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
