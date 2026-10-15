import torch

from src.mask_utils import prepare_edit_only_corruption, resize_edit_mask_to_tokens
from src.scheduler import Scheduler


def test_token_mask_uses_nearest_and_keeps_tiny_edit():
    pixel_mask = torch.zeros(1, 1, 16, 16)
    pixel_mask[:, :, 7, 7] = 1
    token_mask = resize_edit_mask_to_tokens(pixel_mask, (4, 4))
    assert token_mask.dtype == torch.bool
    assert token_mask.shape == (1, 16)
    assert token_mask.sum().item() == 1


def test_training_corruption_and_loss_never_touch_preserve_tokens():
    source = torch.tensor([[1, 2, 3, 4, 5, 6]])
    target = torch.tensor([[10, 11, 12, 13, 14, 15]])
    edit_mask = torch.tensor([[False, True, True, False, False, True]])
    scores = torch.tensor([[0.0, 0.3, 0.1, 0.0, 0.0, 0.2]])
    inputs, labels, corruption, loss_mask = prepare_edit_only_corruption(
        target, source, edit_mask, torch.tensor([2 / 3]), 99, scores
    )
    assert torch.equal(inputs[~edit_mask], source[~edit_mask])
    assert (labels[~edit_mask] == -100).all()
    assert torch.equal(loss_mask, edit_mask & corruption)
    assert corruption.sum().item() == 2


def test_scheduler_preserves_source_tokens_at_every_step():
    scheduler = Scheduler(mask_token_id=9)
    scheduler.set_timesteps(2, temperature=0.0, device="cpu")
    source = torch.tensor([[[1, 2], [3, 4]]])
    edit_mask = torch.tensor([[[False, True], [False, True]]])
    sample = source.clone()
    sample[edit_mask] = 9
    logits = torch.zeros(1, 8, 2, 2)
    logits[:, 7] = 10
    output = scheduler.step(
        logits, scheduler.timesteps[0], sample, edit_mask=edit_mask,
        source_tokens=source, generator=torch.Generator().manual_seed(0),
    ).prev_sample
    assert torch.equal(output[~edit_mask], source[~edit_mask])
