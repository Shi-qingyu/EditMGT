import pytest
import torch

from src.mask_utils import (
    prepare_edit_only_corruption,
    random_mask_to_bbox,
    resize_edit_mask_to_tokens,
)
from src.sparse_utils import prepare_sparse_target_layout


def test_disabled_bbox_keeps_masks_and_rng_unchanged():
    mask = torch.zeros(2, 1, 5, 7)
    mask[:, :, 1, 2] = 1
    mask[:, :, 3, 5] = 1
    rng_state = torch.get_rng_state().clone()
    result, selected = random_mask_to_bbox(mask, 0.0)
    assert torch.equal(result, mask)
    assert not selected.any()
    assert torch.equal(torch.get_rng_state(), rng_state)


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_bbox_fills_disconnected_regions_and_includes_all_extreme_pixels(dtype):
    mask = torch.zeros(4, 1, 5, 7, dtype=dtype)
    mask[0, 0, 1, 2] = 1
    mask[0, 0, 3, 5] = 1
    mask[1, 0, 4, 6] = 1  # A single boundary pixel must stay a single pixel.
    mask[3] = 1  # A full mask stays full; sample 2 remains empty.
    original = mask.clone()
    result, selected = random_mask_to_bbox(mask, 1.0)
    expected = original.clone()
    expected[0, 0, 1:4, 2:6] = 1
    assert torch.equal(result, expected)
    assert torch.equal(mask, original)
    assert selected.tolist() == [True, True, False, True]
    assert result.dtype == mask.dtype and result.device == mask.device


def test_bbox_selection_is_per_example_and_respects_probability(monkeypatch):
    mask = torch.zeros(4, 1, 5, 7)
    mask[:, :, 1, 2] = 1
    mask[:, :, 3, 5] = 1
    original = mask.clone()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.1, 0.7, 0.29, 0.3]))
    result, selected = random_mask_to_bbox(mask, 0.3)
    assert selected.tolist() == [True, False, True, False]
    expected = original.clone()
    expected[[0, 2], :, 1:4, 2:6] = 1
    assert torch.equal(result, expected)
    assert torch.equal(mask, original)


def test_bbox_sampling_is_reproducible_from_torch_rng_state():
    mask = torch.ones(32, 1, 3, 4)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        rng_state = torch.get_rng_state().clone()
        result, selected = random_mask_to_bbox(mask, 0.5)
        torch.set_rng_state(rng_state)
        repeated_result, repeated_selected = random_mask_to_bbox(mask, 0.5)
    assert selected.any() and (~selected).any()
    assert torch.equal(result, repeated_result)
    assert torch.equal(selected, repeated_selected)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_bbox_probability_is_rejected(probability):
    with pytest.raises(ValueError, match="probability must be between 0 and 1"):
        random_mask_to_bbox(torch.zeros(1, 1, 5, 7), probability)


def test_bbox_generation_supervises_filled_gaps_and_preserves_outside_tokens():
    mask = torch.zeros(1, 1, 8, 12)
    mask[0, 0, 2, 2] = mask[0, 0, 5, 9] = 1
    augmented, _ = random_mask_to_bbox(mask, 1.0)
    token_mask = resize_edit_mask_to_tokens(augmented, (4, 6))
    original_token_mask = resize_edit_mask_to_tokens(mask, (4, 6))
    filled_gap = 1 * 6 + 2
    assert token_mask[0, filled_gap] and not original_token_mask[0, filled_gap]

    source = torch.arange(24).reshape(1, 24)
    target = source + 30
    inputs, labels, corruption, loss_mask = prepare_edit_only_corruption(
        target, source, token_mask, torch.ones(1), 99,
    )
    assert torch.equal(corruption, token_mask)
    assert torch.equal(loss_mask, token_mask)
    assert labels[0, filled_gap] == target[0, filled_gap]
    assert inputs[0, filled_gap] == 99
    assert torch.equal(inputs[~token_mask], source[~token_mask])
    assert labels[~token_mask].eq(-100).all()

    layout = prepare_sparse_target_layout(token_mask, (4, 6), (4, 6), 1, token_mask.device, halo=0)
    assert set(layout.indices[0, layout.valid[0]].tolist()) == set(token_mask[0].nonzero().flatten().tolist())
