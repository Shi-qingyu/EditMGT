"""Cosine training noise sampler for discrete, edit-only mask diffusion.

All distributions operate on the unresolved fraction of EDITABLE VQ tokens:
zero is clean and one is fully masked. This is distinct from the descending
cosine schedule's progress variable and from the scheduler's integer step IDs.
"""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class MaskRatioSampler:
    # Retain the historical config fields so existing cosine checkpoints have
    # the same metadata schema. They cannot enable a different sampling law.
    distribution: str = "cosine"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    shift: float = 1.0
    cosine_mix: float = 0.0
    low_noise_mix: float = 0.0
    low_noise_threshold: float = 0.3
    min_mask_ratio: float = 0.0

    def __post_init__(self):
        if self.distribution != "cosine":
            raise ValueError("Training uses cosine sampling only")
        if self.logit_mean != 0.0 or self.logit_std != 1.0:
            raise ValueError("Logit-normal sampling is disabled; retain the compatibility defaults 0 and 1")
        if self.shift != 1.0:
            raise ValueError("Training uses unshifted cosine sampling; timestep_shift must be 1")
        if self.cosine_mix != 0.0 or self.low_noise_mix != 0.0:
            raise ValueError("Training uses pure cosine sampling without mixture components")
        if not 0 <= self.min_mask_ratio < 1 or not 0 < self.low_noise_threshold <= 1:
            raise ValueError("min_mask_ratio must be in [0,1), low_noise_threshold in (0,1]")

    def to_dict(self):
        return asdict(self)

    def sample(self, batch_size, device, generator=None):
        """Sample r=cos(pi*u/2), u~Uniform(0,1), independently per example."""
        draws = torch.rand(batch_size, device=device, generator=generator, dtype=torch.float32)
        ratios = torch.cos(draws * math.pi * 0.5)
        return ratios.clamp(min=self.min_mask_ratio, max=1.0)
