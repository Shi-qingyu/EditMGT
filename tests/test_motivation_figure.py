"""Measurement validity and non-invasive trajectory collection for paper figures."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.motivation_layout import _load_measurements, apply_unresolved_gray, render_figure
from scripts.plot_mgm_motivation import DecodingTrace, display_steps, select_rows


def test_selection_depends_on_ids_not_quality_or_input_order():
    rows = [dict(key=f'sample_{i}', instruction='Turn it blue.', score=i) for i in range(20)]
    expected = [row['key'] for row in select_rows(rows, 8, 2026)]
    reversed_rows = [dict(row, score=-1000 * row['score']) for row in reversed(rows)]
    assert [row['key'] for row in select_rows(reversed_rows, 8, 2026)] == expected
    assert [row['key'] for row in select_rows(rows, 8, 2027)] != expected
    assert [row['key'] for row in select_rows(rows, 1, 2026, ['sample_3', 'sample_1'])] == ['sample_3', 'sample_1']


def test_equal_image_weight_and_raw_predictions_do_not_follow_frozen_state(tmp_path):
    # Synthetic measurements exercise the estimator only; they are not model evidence.
    samples = []
    for key, mask, first_entropy in [
            ('small_roi', np.array([[True, False], [False, False]]), 2.),
            ('large_roi', np.array([[True, True], [True, False]]), 6.)]:
        entropy = np.stack([np.full((2, 2), first_entropy), np.ones((2, 2))])
        argmax = np.stack([np.zeros((2, 2), dtype=int), np.ones((2, 2), dtype=int)])
        unresolved = np.zeros_like(argmax, dtype=bool)  # frozen state differs from raw predictions
        np.savez(tmp_path / f'{key}.npz', entropy=entropy, argmax=argmax,
                 unresolved=unresolved, token_mask=mask, commitment_step=np.ones((2, 2)))
        samples.append(dict(key=key, trajectory=f'{key}.npz'))
    total, curves, _ = _load_measurements(tmp_path, samples)
    assert total == 2
    for region in ['edited', 'context']:
        assert curves[region]['entropy_nats']['mean'] == [4., 1.]
        assert curves[region]['argmax_disagreement']['mean'] == [1., 0.]
        assert curves[region]['unresolved_fraction']['mean'] == [0., 0.]
    assert display_steps(36) == [6, 12, 18, 24, 30, 36]
    assert display_steps(35) == [6, 12, 18, 24, 30, 35]
    with pytest.raises(ValueError, match='Image steps'):
        display_steps(36, [0, 5])


def test_unknown_token_cells_are_uniform_gray_and_known_cells_are_untouched():
    from PIL import Image

    pixels = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)
    unresolved = np.array([[True, False], [False, True]])
    result = np.asarray(apply_unresolved_gray(Image.fromarray(pixels), unresolved))
    assert (result[:4, :3] == 166).all()
    assert (result[4:, 3:] == 166).all()
    np.testing.assert_array_equal(result[:4, 3:], pixels[:4, 3:])
    np.testing.assert_array_equal(result[4:, :3], pixels[4:, :3])
    final = apply_unresolved_gray(Image.fromarray(pixels), np.zeros((2, 2), dtype=bool))
    np.testing.assert_array_equal(np.asarray(final), pixels)


def test_observer_preserves_outputs_rng_and_restores_scheduler():
    torch = pytest.importorskip('torch', exc_type=ImportError)
    from types import SimpleNamespace

    class RandomScheduler:
        def step(self, model_output, timestep, sample, **kwargs):
            prediction = torch.multinomial(model_output.softmax(1).movedim(1, -1).reshape(-1, 3), 1)
            prediction = prediction.reshape_as(sample)
            return SimpleNamespace(prev_sample=prediction, pred_original_sample=prediction)

    scheduler = RandomScheduler()
    logits = torch.tensor([[[[2., 1.]], [[1., 3.]], [[0., 1.]]]])
    sample = torch.full((1, 1, 2), 3, dtype=torch.long)
    torch.manual_seed(25)
    before = torch.get_rng_state()
    expected = scheduler.step(logits, 0, sample)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(before)
    trace = DecodingTrace(mask_token_id=3, chunk_size=1)
    with trace.capture(scheduler):
        actual = scheduler.step(model_output=logits, timestep=0, sample=sample)
    assert torch.equal(actual.prev_sample, expected.prev_sample)
    assert torch.equal(trace.states[0], actual.prev_sample)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert 'step' not in scheduler.__dict__
    log_p = logits.float().log_softmax(1)
    np.testing.assert_allclose(trace.entropy[0], (-(log_p.exp() * log_p).sum(1))[0].numpy())
    with pytest.raises(RuntimeError, match='forced'):
        with trace.capture(scheduler):
            raise RuntimeError('forced')
    assert 'step' not in scheduler.__dict__


def test_renderer_writes_publication_files_from_measured_arrays(tmp_path):
    pytest.importorskip('matplotlib')
    from PIL import Image

    # This deliberately labeled fixture only checks output/layout under tmp_path.
    Image.new('RGB', (64, 64), 'gray').save(tmp_path / 'source.png')
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    for step in [1, 2]:
        Image.new('RGB', (64, 64), (70 * step, 80, 120)).save(tmp_path / f'step_{step}.png')
    token_mask = np.zeros((4, 4), dtype=bool)
    token_mask[1:3, 1:3] = True
    np.savez(tmp_path / 'trajectory.npz', entropy=np.ones((2, 4, 4)),
             argmax=np.zeros((2, 4, 4)), unresolved=np.zeros((2, 4, 4), dtype=bool),
             token_mask=token_mask, commitment_step=np.ones((4, 4)))
    metadata = dict(protocol=dict(model_label='SYNTHETIC TEST FIXTURE'), samples=[dict(
        key='fixture', instruction='Synthetic fixture for renderer validation only.',
        source='source.png', mask='mask.png', trajectory='trajectory.npz',
        frames={'1': 'step_1.png', '2': 'step_2.png'})])
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    paths = render_figure(tmp_path, tmp_path / 'fixture', dpi=72)
    for path in paths.values():
        assert Path(path).stat().st_size > 0
    assert json.loads((tmp_path / 'fixture.figure_data.json').read_text())['sample_count'] == 1
