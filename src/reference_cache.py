"""Per-generation reference snapshots and cache-consistent text/reference KV."""

from typing import Dict, Sequence, Tuple

import torch

# Keep this import available for callers of the original cache-local helper.
from src.mask_schedule import scheduled_mask_ratio


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
        for name, tensor in (("key", key), ("value", value)):
            if (tensor.shape[0], tensor.shape[1], tensor.shape[3]) != (
                query.shape[0], query.shape[1], query.shape[3]
            ) or tensor.device != query.device or tensor.dtype != query.dtype:
                raise ValueError(
                    f"Reference cache and query batch/head/device/dtype differ at {layer} ({name}): "
                    f"cached shape={tuple(tensor.shape)}, device={tensor.device}, dtype={tensor.dtype}; "
                    f"query shape={tuple(query.shape)}, device={query.device}, dtype={query.dtype}"
                )
        return key, value

    @property
    def num_layers(self):
        return len(self._layers)

    def clear(self):
        self._layers.clear()
        self.reference_length = 0
        self.ready = False


class ConditionKVCache:
    """Text/reference K/V, normally from the low-noise, target-independent prefix.

    A complete low-noise forward prefills every layer. Subsequent forwards omit
    both context branches, including their queries, normalization and FFNs.
    Keys are stored after QK normalization and RoPE. CFG halves stay separate.
    This object belongs to one generation; target tensors are never retained.
    ``allow_early_capture`` explicitly enables approximate high-noise snapshots
    for reuse-length ablations. The trained phase policy remains unchanged;
    the captured context is frozen even when the low-noise phase starts.
    """

    def __init__(self, allow_early_capture=False):
        self._layers = {}
        self.allow_early_capture = bool(allow_early_capture)
        self.text_length = 0
        self.reference_length = 0
        self.pooled_projections = None
        self.ready = False

    def store(self, layer, text_key, text_value, reference_key, reference_value):
        if self.ready or layer is None or layer in self._layers:
            raise ValueError("Condition cache capture requires a fresh, unique layer entry")
        pairs = ((text_key, text_value), (reference_key, reference_value))
        for key, value in pairs:
            if key.ndim != 4 or key.shape != value.shape or key.shape[2] == 0:
                raise ValueError("Condition K/V must contain complete nonempty [B,H,N,D] sequences")
        if (text_key.shape[0], text_key.shape[1], text_key.shape[3]) != (
            reference_key.shape[0], reference_key.shape[1], reference_key.shape[3]
        ):
            raise ValueError("Text and reference K/V dimensions differ")
        if self._layers and (text_key.shape[2], reference_key.shape[2]) != (
            self.text_length, self.reference_length
        ):
            raise ValueError("All layers must cache the same complete text/reference sequences")
        self.text_length, self.reference_length = text_key.shape[2], reference_key.shape[2]
        # Text slices come from concatenated [text,target] storage. Clone them
        # so retaining the cache cannot retain target K/V through a tensor view.
        self._layers[layer] = (
            text_key.detach().clone(), text_value.detach().clone(),
            reference_key.detach(), reference_value.detach(),
        )

    def freeze(self, expected_layers):
        if not expected_layers or set(self._layers) != set(expected_layers):
            raise ValueError("Condition cache is missing Transformer layers")
        if self.pooled_projections is None:
            raise ValueError("Condition cache is missing static pooled conditioning")
        self.ready = True

    def get(self, layer, query):
        if not self.ready or layer not in self._layers:
            raise ValueError("Prefill the complete condition cache before reuse")
        tensors = self._layers[layer]
        for name, tensor in zip(("text_key", "text_value", "reference_key", "reference_value"), tensors):
            if (tensor.shape[0], tensor.shape[1], tensor.shape[3]) != (
                query.shape[0], query.shape[1], query.shape[3]
            ) or tensor.device != query.device or tensor.dtype != query.dtype:
                raise ValueError(
                    f"Condition cache and target query batch/head/device/dtype differ at {layer} ({name}): "
                    f"cached shape={tuple(tensor.shape)}, device={tensor.device}, dtype={tensor.dtype}; "
                    f"query shape={tuple(query.shape)}, device={query.device}, dtype={query.dtype}"
                )
        return tensors

    @property
    def num_layers(self):
        return len(self._layers)

    def clear(self):
        self._layers.clear()
        self.text_length = self.reference_length = 0
        self.pooled_projections = None
        self.ready = False


