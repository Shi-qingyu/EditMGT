"""Training noise samplers for discrete, edit-only mask diffusion.

All distributions operate on the unresolved fraction of EDITABLE VQ tokens:
zero is clean and one is fully masked. This is distinct from the descending
cosine schedule's progress variable and from the scheduler's integer step IDs.
"""

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class MaskRatioSampler:
    distribution: str = "cosine"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    shift: float = 1.0
    cosine_mix: float = 0.0
    low_noise_mix: float = 0.0
    low_noise_threshold: float = 0.3
    min_mask_ratio: float = 0.0

    def __post_init__(self):
        if self.distribution not in ("cosine", "uniform", "logit_normal"):
            raise ValueError("Unknown mask-ratio sampling distribution")
        if not math.isfinite(self.logit_mean) or not math.isfinite(self.logit_std) or self.logit_std <= 0:
            raise ValueError("Logit-normal mean must be finite and std must be positive and finite")
        if not math.isfinite(self.shift) or self.shift <= 0:
            raise ValueError("Timestep shift must be positive and finite")
        if not 0 <= self.cosine_mix <= 1 or not 0 <= self.low_noise_mix <= 1:
            raise ValueError("Timestep mixture probabilities must be in [0,1]")
        if self.cosine_mix + self.low_noise_mix > 1:
            raise ValueError("Cosine and low-noise mixture probabilities must sum to at most one")
        if not 0 <= self.min_mask_ratio < 1 or not 0 < self.low_noise_threshold <= 1:
            raise ValueError("min_mask_ratio must be in [0,1), low_noise_threshold in (0,1]")
        if self.low_noise_mix and self.min_mask_ratio >= self.low_noise_threshold:
            raise ValueError("The low-noise component requires min_mask_ratio < low_noise_threshold")

    def to_dict(self):
        return asdict(self)

    def sample(self, batch_size, device, generator=None):
        """Return float32 mask ratios, without CPU transfers or scalar GPU reads.

        Shift applies to the main component only. The unshifted cosine coverage
        component retains the previous sampler's high-noise coverage; the final
        low-noise component stays inside the trained cache-conditioning phase.
        Mixture probabilities describe examples, rather than supervised tokens.
        """
        if self.distribution == "logit_normal":
            logits = torch.randn(batch_size, device=device, generator=generator, dtype=torch.float32)
            # A rational shift of sigmoid(z) is exactly sigmoid(z + log(shift)).
            ratios = torch.sigmoid(logits * self.logit_std + self.logit_mean + math.log(self.shift))
        else:
            draws = torch.rand(batch_size, device=device, generator=generator, dtype=torch.float32)
            ratios = torch.cos(draws * math.pi * 0.5) if self.distribution == "cosine" else draws
            if self.shift != 1.0:
                scaled = self.shift * ratios
                ratios = scaled / (1.0 - ratios + scaled)

        if self.cosine_mix or self.low_noise_mix:
            component = torch.rand(batch_size, device=device, generator=generator, dtype=torch.float32)
            if self.cosine_mix:
                coverage = torch.cos(
                    torch.rand(batch_size, device=device, generator=generator, dtype=torch.float32)
                    * math.pi * 0.5
                )
                use_cosine = (component >= self.low_noise_mix) & (
                    component < self.low_noise_mix + self.cosine_mix
                )
                ratios = torch.where(use_cosine, coverage, ratios)
            if self.low_noise_mix:
                low_ratios = self.min_mask_ratio + (
                    self.low_noise_threshold - self.min_mask_ratio
                ) * torch.rand(batch_size, device=device, generator=generator, dtype=torch.float32)
                ratios = torch.where(component < self.low_noise_mix, low_ratios, ratios)

        return ratios.clamp(min=self.min_mask_ratio, max=1.0)
