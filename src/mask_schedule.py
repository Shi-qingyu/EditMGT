"""Host-side mask schedules shared by remasking, time conditioning and caching."""

import math


def validate_inference_shift(shift: float) -> float:
    """Accept the old identity argument without enabling shifted schedules."""
    shift = float(shift)
    if not math.isfinite(shift) or shift != 1.0:
        raise ValueError("Inference uses unshifted cosine only; inference_shift must be 1")
    return 1.0


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
    if schedule != "cosine":
        raise ValueError("Inference uses cosine masking only")
    if step == 0:
        return 1.0
    if step == num_steps:
        return 0.0
    progress = step / num_steps
    return math.cos(progress * math.pi / 2)
