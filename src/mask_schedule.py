"""Host-side mask schedules shared by remasking, time conditioning and caching."""

import math


def validate_inference_shift(shift: float) -> float:
    shift = float(shift)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("inference_shift must be positive and finite")
    return shift


def scheduled_mask_ratio(step: int, num_steps: int, schedule: str, shift: float = 1.0) -> float:
    """Unresolved fraction BEFORE forward ``step``, including the terminal state.

    Ratios are relative to the generation region (including supervised halo),
    rather than the full image. Integer rounding and the scheduler's minimum
    progress/minimum-one-token rules can change actual masked token counts.
    ``step=0`` is fully masked; ``step=num_steps`` is exactly clean.
    """
    shift = validate_inference_shift(shift)
    if num_steps < 1 or not 0 <= step <= num_steps:
        raise ValueError("Expected num_steps > 0 and step in [0,num_steps]")
    if schedule not in ("cosine", "linear"):
        raise ValueError(f"Unknown masking schedule {schedule}")
    if step == 0:
        return 1.0
    if step == num_steps:
        return 0.0
    progress = step / num_steps
    ratio = math.cos(progress * math.pi / 2) if schedule == "cosine" else 1.0 - progress
    if shift == 1.0:
        return ratio
    scaled = shift * ratio
    return scaled / (1.0 - ratio + scaled)


def legacy_timestep_for_mask_ratio(mask_ratio: float, num_steps: int, schedule: str) -> float:
    """Map shifted noise to the equivalent position in the old step-ID units.

    Original forward i used time N-1-i and mask schedule r(i/N). Inverting
    that schedule preserves this coordinate system for legacy checkpoints.
    Values beyond the last old forward's noise level use its zero time.
    """
    ratio = min(1.0, max(0.0, mask_ratio))
    if schedule == "cosine":
        progress = 2.0 * math.acos(ratio) / math.pi
    elif schedule == "linear":
        progress = 1.0 - ratio
    else:
        raise ValueError(f"Unknown masking schedule {schedule}")
    return max(0.0, min(float(num_steps - 1), num_steps - 1 - num_steps * progress))
