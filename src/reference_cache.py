"""Per-generation, full-reference KV snapshots for low-noise inference."""

from typing import Dict, Sequence, Tuple

import torch


class ReferenceKVCache:
    """Cache each layer's reference K after QK normalization and RoPE, plus V.

    A normal joint forward records all layers at the low-noise transition.
    After ``freeze``, subsequent forwards consume only the cached reference
    keys/values, without reference queries or hidden-state updates. CFG halves
    remain separate along the batch dimension. No target tensors are stored.
    Create a fresh instance for every generation, prompt, mask and model state.
    """

    def __init__(self):
        self._layers: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.reference_length = 0
        self.ready = False

    def store(self, layer: str, key: torch.Tensor, value: torch.Tensor):
        if self.ready:
            raise ValueError("A frozen reference cache cannot be overwritten")
        if layer is None or layer in self._layers:
            raise ValueError("Reference cache capture requires a unique name for each layer")
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("Reference K/V must have matching [batch, heads, tokens, dim] shapes")
        if self._layers and key.shape[2] != self.reference_length:
            raise ValueError("All layers must cache the complete reference sequence")
        self.reference_length = key.shape[2]
        self._layers[layer] = (key.detach(), value.detach())

    def freeze(self, expected_layers: Sequence[str]):
        if not expected_layers or set(self._layers) != set(expected_layers):
            raise ValueError("Reference cache is missing Transformer layers; use ConFluxAttnProcessor2_0")
        self.ready = True

    def get(self, layer: str, query: torch.Tensor):
        if not self.ready:
            raise ValueError("Capture a complete reference forward before reusing its KV cache")
        if layer not in self._layers:
            raise ValueError(f"Reference cache has no entry for {layer}")
        key, value = self._layers[layer]
        if (key.shape[0], key.shape[1], key.shape[3]) != (query.shape[0], query.shape[1], query.shape[3]):
            raise ValueError("Reference cache batch/head dimensions differ from the current queries")
        if key.device != query.device or key.dtype != query.dtype:
            raise ValueError("Reference cache device/dtype differs from the current queries")
        return key, value

    @property
    def num_layers(self):
        return len(self._layers)

    def clear(self):
        self._layers.clear()
        self.reference_length = 0
        self.ready = False


def scheduled_mask_ratio(step: int, num_steps: int, schedule: str) -> float:
    """Scheduled unresolved fraction BEFORE a model forward (no GPU sync).

    Step zero starts fully masked. Step i receives the sample produced by
    scheduler step i-1, whose mask schedule uses i / num_steps. The ratio is
    relative to editable tokens, rather than the area of the entire image.
    Rounding and the scheduler's minimum-one-token rule can change actual counts.
    """
    import math

    progress = step / num_steps
    if schedule == "cosine":
        return math.cos(progress * math.pi / 2)
    if schedule == "linear":
        return 1.0 - progress
    raise ValueError(f"Unknown masking schedule {schedule}")
