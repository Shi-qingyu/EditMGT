"""Weighted sampling of editing streams that already own their rank/worker shards."""
import math
import random

from torch.utils.data import IterableDataset, get_worker_info


class MixedEditDataset(IterableDataset):
    """Infinite per-example mixture; optimizer steps determine training length."""

    def __init__(self, datasets, names, weights, rank=0, seed=42):
        super().__init__()
        if not datasets or len(datasets) != len(names) or len(datasets) != len(weights):
            raise ValueError("Mixture datasets, names and weights must have the same nonzero length")
        if len(set(names)) != len(names) or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("Mixture names must be unique and weights must be positive and finite")
        self.datasets, self.names, self.weights = tuple(datasets), tuple(names), tuple(weights)
        self.rank, self.seed = rank, seed

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(self.seed + 104729 + 1000003 * self.rank + worker_id)
        # Child iterators execute in this same worker and perform their own
        # disjoint sharding. Do not shard or Accelerator.prepare this stream again.
        iterators = [iter(dataset) for dataset in self.datasets]
        choices = range(len(self.datasets))
        while True:
            index = rng.choices(choices, weights=self.weights, k=1)[0]
            try:
                sample = next(iterators[index])
            except StopIteration:
                iterators[index] = iter(self.datasets[index])
                try:
                    sample = next(iterators[index])
                except StopIteration as exc:
                    raise ValueError(f"Mixture source {self.names[index]} has no samples in worker {worker_id}") from exc
            sample = dict(sample)
            sample["dataset_name"] = self.names[index]
            sample["sample_key"] = f"{self.names[index]}:{sample['sample_id']}"
            yield sample
