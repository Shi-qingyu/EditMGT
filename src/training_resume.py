"""Compatibility checks for deliberate changes to a resumed training recipe."""
import hashlib
import json
import math
from pathlib import Path


def condition_cache_transition(saved_config, current_config, allow_threshold_change=False):
    """Allow an explicit threshold migration, keeping the cache mode unchanged."""
    saved_enabled = bool(saved_config.get("cache_conditioning", False))
    current_enabled = bool(current_config.get("cache_conditioning", False))
    if saved_enabled != current_enabled:
        raise ValueError("Checkpoint cache-conditioning mode differs from this run; --cache_conditioning must match")
    if not current_enabled:
        return None
    # Missing fields in historical checkpoints denote their original 0.3
    # phase; current runs use 0.8 and record that migration explicitly.
    previous = float(saved_config.get("condition_cache_threshold", 0.3))
    current = float(current_config.get("condition_cache_threshold", 0.8))
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in (previous, current)):
        raise ValueError("Checkpoint and current condition-cache thresholds must be finite and in (0,1]")
    if previous == current:
        return None
    if not allow_threshold_change:
        raise ValueError(
            f"Checkpoint condition-cache threshold {previous} differs from requested {current}. "
            "Use --allow_condition_cache_threshold_change_on_resume for intentional continued training "
            "with the new attention phase."
        )
    return {"previous_threshold": previous, "current_threshold": current}


def record_condition_cache_transition(output_dir, checkpoint, transition):
    """Keep distinct transitions without modifying the source checkpoint."""
    record = {"resume_checkpoint": str(Path(checkpoint).resolve()), **transition}
    payload = json.dumps(record, indent=2) + "\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    directory = Path(output_dir) / "condition_cache_transitions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{Path(checkpoint).name}_{digest}.json"
    path.write_text(payload, encoding="utf-8")
    return path
