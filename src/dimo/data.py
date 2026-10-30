"""Deterministic data replay, independent of model sampling RNG.

The existing readers are infinite iterable streams without seek/state APIs.
Zero DataLoader workers plus isolated RNG permits exact replay, at O(batches)
resume cost, preserving their preprocessing and rank-local sampling.
"""
import random
import numpy as np
import torch


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
