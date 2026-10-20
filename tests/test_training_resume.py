"""Guard explicit condition-cache threshold migrations during full-state resume."""
import json

import pytest

from src.training_resume import condition_cache_transition, record_condition_cache_transition


def cache_config(threshold=0.3, enabled=True):
    return {"cache_conditioning": enabled, "condition_cache_threshold": threshold}


def test_unchanged_threshold_needs_no_migration():
    assert condition_cache_transition(cache_config(), cache_config()) is None


def test_threshold_change_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="--allow_condition_cache_threshold_change_on_resume"):
        condition_cache_transition(cache_config(), cache_config(0.8))


def test_migration_retains_both_original_configs():
    saved, current = cache_config(), cache_config(0.8)
    transition = condition_cache_transition(saved, current, allow_threshold_change=True)
    assert transition == {"previous_threshold": 0.3, "current_threshold": 0.8}
    assert saved == cache_config() and current == cache_config(0.8)


@pytest.mark.parametrize("saved_enabled,current_enabled", [(True, False), (False, True)])
def test_migration_cannot_change_cache_conditioning_mode(saved_enabled, current_enabled):
    with pytest.raises(ValueError, match="mode differs"):
        condition_cache_transition(cache_config(enabled=saved_enabled), cache_config(enabled=current_enabled),
                                   allow_threshold_change=True)


@pytest.mark.parametrize("threshold", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_checkpoint_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="finite and in"):
        condition_cache_transition(cache_config(threshold), cache_config(0.8), allow_threshold_change=True)


def test_transition_history_keeps_source_checkpoint_and_previous_migrations(tmp_path):
    checkpoint = tmp_path / "checkpoint-6800"
    checkpoint.mkdir()
    config = checkpoint / "config.json"
    config.write_text('{"condition_cache_threshold": 0.3}\n')
    output = tmp_path / "run"
    first = {"previous_threshold": 0.3, "current_threshold": 0.8}
    first_path = record_condition_cache_transition(output, checkpoint, first)
    assert record_condition_cache_transition(output, checkpoint, first) == first_path
    second_path = record_condition_cache_transition(
        output, checkpoint, {"previous_threshold": 0.3, "current_threshold": 0.7},
    )
    assert second_path != first_path
    assert json.loads(first_path.read_text()) == {"resume_checkpoint": str(checkpoint.resolve()), **first}
    assert config.read_text() == '{"condition_cache_threshold": 0.3}\n'
