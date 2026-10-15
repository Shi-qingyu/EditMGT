"""Gather/scatter metadata for full-reference, sparse-target attention."""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class SparseTargetLayout:
    indices: torch.Tensor  # [B, max_active], original row-major grid indices
    valid: torch.Tensor  # [B, max_active], excludes batch padding
    shape: torch.Tensor  # [B, 2], active bounding-box (h, w), including halo
    grid_size: Tuple[int, int]

    def gather(self, sequence: torch.Tensor) -> torch.Tensor:
        """Gather a [B,H*W,C] sequence; padding is zero and never a valid key."""
        values = sequence.gather(1, self.indices[..., None].expand(-1, -1, sequence.shape[-1]))
        return values.masked_fill(~self.valid[..., None], 0)

    @torch.compiler.disable
    def scatter(self, sequence: torch.Tensor, full_sequence: torch.Tensor) -> torch.Tensor:
        """Replace active positions once; retain the pre-Transformer background."""
        batch, length, channels = full_sequence.shape
        offsets = torch.arange(batch, device=self.indices.device)[:, None] * length
        flat_indices = (self.indices + offsets)[self.valid]
        return full_sequence.reshape(-1, channels).index_copy(
            0, flat_indices, sequence[self.valid]
        ).reshape(batch, length, channels)


# Active-set sizing and boolean-index scatter are data dependent. Keep these
# small topology operations eager while torch.compile optimizes the blocks.
@torch.compiler.disable
def prepare_sparse_target_layout(
    edit_mask: torch.Tensor,
    input_grid_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    batch_size: int,
    device: torch.device,
    halo: int = 1,
) -> SparseTargetLayout:
    """Project VQ-grid edit regions onto the stem grid, then dilate a small halo.

    Pooling uses OR/max, so an edit anywhere in a downsampled patch is retained.
    Samples keep their own active sets; only batch padding uses max_active.
    """
    if not isinstance(halo, int) or halo < 0:
        raise ValueError("sparse_target_halo must be a non-negative integer")
    edit_mask = edit_mask.to(device=device, dtype=torch.bool)
    if edit_mask.ndim == 2:
        if edit_mask.shape[1] != input_grid_size[0] * input_grid_size[1]:
            raise ValueError("Flattened target_edit_mask must match the input VQ grid")
        edit_mask = edit_mask.reshape(-1, *input_grid_size)
    elif edit_mask.ndim == 4 and edit_mask.shape[1] == 1:
        edit_mask = edit_mask[:, 0]
    if edit_mask.ndim != 3 or tuple(edit_mask.shape[-2:]) != tuple(input_grid_size):
        raise ValueError("target_edit_mask must be [B,H,W], [B,1,H,W], or [B,H*W] on the VQ grid")
    if edit_mask.shape[0] == 1:
        edit_mask = edit_mask.expand(batch_size, -1, -1)
    if edit_mask.shape[0] != batch_size:
        raise ValueError("target_edit_mask batch must match the target batch")

    active = F.adaptive_max_pool2d(edit_mask[:, None].float(), grid_size)
    if halo:
        active = F.max_pool2d(active, 2 * halo + 1, stride=1, padding=halo)
    active = active[:, 0].bool()
    flat_active = active.flatten(1)
    counts = flat_active.sum(dim=1)
    max_active = int(counts.max().item())
    # Sorting original indices preserves spatial order; no spatial re-numbering.
    length = flat_active.shape[1]
    all_indices = torch.arange(length, device=device).expand(batch_size, -1)
    indices = all_indices.masked_fill(~flat_active, length).sort(dim=1).values[:, :max_active]
    valid = torch.arange(max_active, device=device)[None] < counts[:, None]
    indices = indices.masked_fill(~valid, 0)

    rows = active.any(dim=2)
    cols = active.any(dim=1)
    row_ids = torch.arange(grid_size[0], device=device)[None].expand(batch_size, -1)
    col_ids = torch.arange(grid_size[1], device=device)[None].expand(batch_size, -1)
    shape_h = row_ids.masked_fill(~rows, -1).amax(1) - row_ids.masked_fill(~rows, grid_size[0]).amin(1) + 1
    shape_w = col_ids.masked_fill(~cols, -1).amax(1) - col_ids.masked_fill(~cols, grid_size[1]).amin(1) + 1
    shape = torch.stack((shape_h, shape_w), dim=1).clamp(min=0)
    return SparseTargetLayout(indices, valid, shape, tuple(grid_size))


def prepare_branch_ids(ids: torch.Tensor, batch_size: int, length: int, token_type: int = None):
    """Keep each sample's original (row,col); optionally assign the RoPE type axis."""
    if ids is None or ids.ndim not in (2, 3) or ids.shape[-2:] != (length, 3):
        raise ValueError(f"Branch position IDs must be [N,3] or [B,N,3] with N={length}")
    if ids.ndim == 2:
        ids = ids[None]
    if ids.shape[0] == 1:
        ids = ids.expand(batch_size, -1, -1)
    if ids.shape[0] != batch_size:
        raise ValueError("Branch position ID batch must match the hidden-state batch")
    if token_type is not None:
        ids = ids.clone()
        ids[..., 0] = token_type
    return ids


def batched_rotary_embedding(pos_embed, ids: torch.Tensor):
    """FluxPosEmbed takes [N,3]; restore a batch axis for per-sample positions."""
    batch, length, _ = ids.shape
    return tuple(value.reshape(batch, length, -1) for value in pos_embed(ids.flatten(0, 1)))
