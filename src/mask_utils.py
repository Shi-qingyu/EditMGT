"""Utilities shared by mask-aware EditMGT training and inference."""
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps


def normalize_edit_mask_image(mask: Image.Image) -> Image.Image:
    """Return a binary L mask, accepting both 0/1 and 0/255 annotations."""
    mask = ImageOps.exif_transpose(mask).convert("L")
    _, maximum = mask.getextrema()
    threshold = 1 if maximum <= 1 else 128
    return mask.point(lambda value: 255 if value >= threshold else 0)


def resize_binary_edit_mask(mask: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize a normalized binary mask to (height, width), keeping coverage.

    Downsampling uses max pooling so thin/subpixel edit regions survive. Upsampling
    uses nearest neighbours. A positive output cell represents existing positive
    input pixels; an originally empty annotation stays empty.
    """
    height, width = size
    if height < 1 or width < 1:
        raise ValueError("Edit mask dimensions must be positive")
    if mask.size == (width, height):
        return mask
    pooled_height = min(mask.height, height)
    pooled_width = min(mask.width, width)
    if (pooled_width, pooled_height) != mask.size:
        pixels = torch.from_numpy(np.array(mask, dtype=np.uint8, copy=True)).float()[None, None]
        pixels = F.adaptive_max_pool2d(pixels, (pooled_height, pooled_width))
        mask = Image.fromarray(pixels[0, 0].to(torch.uint8).numpy())
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return mask


def prepare_edit_only_corruption(
    target_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    token_edit_mask: torch.Tensor,
    mask_probability: torch.Tensor,
    mask_token_id: int,
    random_scores: Optional[torch.Tensor] = None,
):
    """Apply the original mask-ratio schedule only to editable target tokens."""
    if target_tokens.shape != source_tokens.shape or target_tokens.shape != token_edit_mask.shape:
        raise ValueError("target, source, and token edit mask must have identical [B,N] shapes")
    token_edit_mask = token_edit_mask.bool()
    edit_count = token_edit_mask.sum(dim=-1)
    if (edit_count == 0).any():
        raise ValueError("Every sample must contain at least one editable token")
    if random_scores is None:
        random_scores = torch.rand(target_tokens.shape, device=target_tokens.device)
    random_scores = random_scores.masked_fill(~token_edit_mask, 2.0)
    edit_rank = random_scores.argsort(dim=-1).argsort(dim=-1)
    masked_count = (edit_count * mask_probability).round().long().clamp(min=1)
    corruption_mask = token_edit_mask & (edit_rank < masked_count.unsqueeze(-1))
    clean_composite = torch.where(token_edit_mask, target_tokens, source_tokens)
    input_tokens = torch.where(corruption_mask, mask_token_id, clean_composite)
    loss_mask = token_edit_mask & corruption_mask
    labels = torch.where(loss_mask, target_tokens, -100)
    if (loss_mask & ~token_edit_mask).any():
        raise AssertionError("Preserve-region token entered the generation loss")
    return input_tokens, labels, corruption_mask, loss_mask


def resize_edit_mask_to_tokens(
    pixel_edit_mask: torch.Tensor,
    token_size: Tuple[int, int],
) -> torch.Tensor:
    """Convert a pixel mask (1=edit) to a flattened boolean token mask.

    Nearest-neighbour interpolation is intentional: edit/preserve labels must not
    be blended at region boundaries.
    """
    if pixel_edit_mask.ndim == 2:
        pixel_edit_mask = pixel_edit_mask[None, None]
    elif pixel_edit_mask.ndim == 3:
        pixel_edit_mask = pixel_edit_mask[:, None]
    elif pixel_edit_mask.ndim != 4 or pixel_edit_mask.shape[1] != 1:
        raise ValueError(
            "pixel_edit_mask must have shape [H,W], [B,H,W], or [B,1,H,W]; "
            f"got {tuple(pixel_edit_mask.shape)}"
        )

    pixel_edit_mask = pixel_edit_mask.float()
    token_mask = F.interpolate(pixel_edit_mask, size=token_size, mode="nearest")[:, 0].ge(0.5)

    # A sub-token-sized positive region can fall between nearest-neighbour sample
    # points. Preserve it as one token at the edit-mask centroid instead of
    # silently turning the sample into an all-preserve example.
    pixel_positive = pixel_edit_mask[:, 0].ge(0.5)
    for batch_index in range(token_mask.shape[0]):
        if pixel_positive[batch_index].any() and not token_mask[batch_index].any():
            coordinates = pixel_positive[batch_index].nonzero(as_tuple=False).float().mean(dim=0)
            row = min(token_size[0] - 1, int(coordinates[0] * token_size[0] / pixel_positive.shape[-2]))
            col = min(token_size[1] - 1, int(coordinates[1] * token_size[1] / pixel_positive.shape[-1]))
            token_mask[batch_index, row, col] = True
    return token_mask.flatten(1)


def save_mask_debug_visualization(
    source_image: torch.Tensor,
    pixel_edit_mask: torch.Tensor,
    token_edit_mask: torch.Tensor,
    token_size: Tuple[int, int],
    output_path: str,
) -> None:
    """Save source, pixel mask, token mask, and token-mask overlay side by side."""
    source = source_image.detach().float().cpu()
    if source.ndim == 4:
        source = source[0]
    if source.min() < 0:
        source = source.mul(0.5).add(0.5)
    source = source.clamp(0, 1)

    pixel_mask = pixel_edit_mask.detach().float().cpu()
    while pixel_mask.ndim > 2:
        pixel_mask = pixel_mask[0]
    pixel_mask = pixel_mask.ge(0.5).float()

    flat_token_mask = token_edit_mask.detach().float().cpu()
    if flat_token_mask.ndim == 2:
        flat_token_mask = flat_token_mask[0]
    token_mask = flat_token_mask.reshape(token_size)
    token_overlay_mask = F.interpolate(
        token_mask[None, None], size=source.shape[-2:], mode="nearest"
    )[0, 0]
    pixel_mask = F.interpolate(
        pixel_mask[None, None], size=source.shape[-2:], mode="nearest"
    )[0, 0]

    source_rgb = source[:3]
    pixel_rgb = pixel_mask.expand(3, -1, -1)
    token_rgb = token_overlay_mask.expand(3, -1, -1)
    red = torch.tensor([1.0, 0.0, 0.0])[:, None, None]
    overlay = source_rgb * (1.0 - 0.45 * token_overlay_mask) + red * (0.45 * token_overlay_mask)
    panel = torch.cat([source_rgb, pixel_rgb, token_rgb, overlay], dim=-1)
    array = panel.mul(255).round().byte().permute(1, 2, 0).numpy()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
