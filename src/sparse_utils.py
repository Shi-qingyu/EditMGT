"""Gather/scatter metadata for full-reference, sparse-target attention."""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def _active_target_mask(edit_mask: torch.Tensor, grid_size: Tuple[int, int], halo: int):
    """Shared OR projection and dilation for attention and supervision."""
    if not isinstance(halo, int) or halo < 0:
        raise ValueError("sparse_target_halo must be a non-negative integer")
    active = F.adaptive_max_pool2d(edit_mask[:, None].float(), grid_size)
    if halo:
        active = F.max_pool2d(active, 2 * halo + 1, stride=1, padding=halo)
    return active[:, 0].bool()


def expand_edit_mask_for_halo(
    edit_mask: torch.Tensor,
    input_grid_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    halo: int,
) -> torch.Tensor:
    """Expand active Transformer cells back to their covered VQ tokens.

    The returned mask has the input shape. Use it for corruption, masked-token
    CE and inference remasking. Keep the ORIGINAL mask for Transformer gather,
    which performs the same dilation exactly once on its own grid.
    With halo=0, return the original VQ mask without filling whole Transformer
    cells. Only the attention gather needs the coarser grid alignment.
    """
    if not isinstance(halo, int) or halo < 0:
        raise ValueError("sparse_target_halo must be a non-negative integer")
    if edit_mask.ndim == 2:
        if edit_mask.shape[1] != input_grid_size[0] * input_grid_size[1]:
            raise ValueError("Flattened edit mask must match the VQ grid")
    elif edit_mask.ndim == 3 or (edit_mask.ndim == 4 and edit_mask.shape[1] == 1):
        if tuple(edit_mask.shape[-2:]) != tuple(input_grid_size):
            raise ValueError("Edit mask must match the VQ grid")
    else:
        raise ValueError("Edit mask must be [B,N], [B,H,W], or [B,1,H,W]")
    if any(size < 1 for size in (*input_grid_size, *grid_size)) or any(
        source % target for source, target in zip(input_grid_size, grid_size)
    ):
        raise ValueError("The VQ grid must be an integer multiple of the Transformer grid")
    if halo == 0:
        # A downsample->upsample round trip would otherwise activate neighbouring
        # VQ tokens even with dilation disabled. Keep the original generation
        # and supervision region; the Transformer can still gather its cell.
        return edit_mask.bool()
    mask_grid = edit_mask.reshape(-1, *input_grid_size).bool()
    active = _active_target_mask(mask_grid, grid_size, halo)
    expanded = F.interpolate(active[:, None].float(), size=input_grid_size, mode="nearest")[:, 0].bool()
    return expanded.reshape(edit_mask.shape)


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
    halo: int = 0,
) -> SparseTargetLayout:
    """Project VQ-grid edit regions onto the stem grid; optional halo defaults off.

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

    active = _active_target_mask(edit_mask, grid_size, halo)
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
