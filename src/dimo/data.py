"""Deterministic data replay, independent of model sampling RNG.

The existing readers are infinite iterable streams without seek/state APIs.
Zero DataLoader workers plus isolated RNG permits exact replay, at O(batches)
resume cost, preserving their preprocessing and rank-local sampling.
"""
import random
from types import SimpleNamespace
import numpy as np
import torch


def local_stream_args(config):
    """Adapt DiMO settings to the same dataset factory used by SFT."""
    from src.maskedit_dataset import MASKEDIT_CONFIGS

    return SimpleNamespace(
        instance_data_dir=config['inter_edit_root'], dataset_split=config['dataset_split'],
        dataset_language=config['language'], dataset_cache_dir=config['cache_dir'],
        resolution=config['resolution'], text_encoder_architecture='CLIP_Gemma2', seed=config['seed'],
        inter_edit_translations_dir=config.get('inter_edit_translations_dir') or None,
        exclude_inter_edit_text_editing=config.get('exclude_inter_edit_text_editing', False),
        magicbrush_data_dir=config.get('magicbrush_root') or None,
        magicbrush_probability=config.get('magicbrush_probability', 0.2),
        magicbrush_read_batch_size=config.get('magicbrush_read_batch_size', 4),
        maskedit_data_dir=config.get('maskedit_root') or None,
        maskedit_probability=config.get('maskedit_probability', 0.1),
        maskedit_configs=config.get('maskedit_configs', list(MASKEDIT_CONFIGS)),
        maskedit_prompt_field=config.get('maskedit_prompt_field', 'edit_instruction'),
        maskedit_read_batch_size=config.get('maskedit_read_batch_size', 4),
        # DiMO restores its own stream offset, never the SFT optimizer/data state.
        resume_from_checkpoint=None,
    )


def capture_rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


class ReplayStream:
    def __init__(self, loader, seed, consumed=0):
        current = capture_rng()
        try:
            random.seed(seed)
            np.random.seed(seed)
            # Data replay must not reset CUDA's model-sampling RNG.
            torch.random.default_generator.manual_seed(seed)
            self.iterator = iter(loader)
            self.rng = capture_rng()
        finally:
            restore_rng(current)
        self.consumed = 0
        for _ in range(consumed):
            next(self)

    def __iter__(self):
        return self

    def __next__(self):
        current = capture_rng()
        try:
            restore_rng(self.rng)
            batch = next(self.iterator)
            self.rng = capture_rng()
            self.consumed += 1
            return batch
        finally:
            restore_rng(current)
