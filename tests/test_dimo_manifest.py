import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.dimo.checkpoint import CHECKPOINT_KIND, UPDATE_RULE, state_digest, validate_progress
from src.dimo.manifest_data import manifest_stream


class Tokenizer:
    def __call__(self, text, max_length, **kwargs):
        return SimpleNamespace(input_ids=torch.full((1, max_length), len(text), dtype=torch.long))


@pytest.fixture
def manifest_config(tmp_path):
    for name, image in [('source.png', Image.new('RGB', (32, 20), 'gray')),
                        ('target.png', Image.new('RGB', (32, 20), 'red')),
                        ('mask.png', Image.new('L', (32, 20), 255))]:
        image.save(tmp_path / name)
    rows = []
    for i in range(8):
        row = dict(sample_uid=str(i), dataset_name='magicbrush', instruction_en='Edit sample ' + str(i))
        for key, name in [('source_locator', 'source.png'), ('target_locator', 'target.png'),
                          ('region_locator', 'mask.png')]:
            row[key] = dict(backend='file', relative_path=name)
        rows.append(row)
    path = tmp_path / 'train.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return dict(data_manifest=str(path), data_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_dataset_counts={'magicbrush': 8}, data_roots={'magicbrush': str(tmp_path)},
                micro_batch=1, resolution=16, seed=42)


def test_manifest_exact_seek_across_epochs_and_rank_disjointness(manifest_config):
    config = manifest_config
    tokenizer = [Tokenizer(), Tokenizer()]
    rank_first = []
    for rank in range(4):
        acc = SimpleNamespace(process_index=rank, num_processes=4)
        stream = manifest_stream(config, tokenizer, acc, 0)
        first = [next(stream), next(stream)]
        rank_first.extend(batch['sample_id'][0] for batch in first)
        for consumed in (1, 2, 5):
            uninterrupted = manifest_stream(config, tokenizer, acc, 0)
            for _ in range(consumed):
                next(uninterrupted)
            expected = state_digest(next(uninterrupted))
            rng = torch.get_rng_state().clone()
            resumed = manifest_stream(config, tokenizer, acc, consumed)
            assert resumed.consumed == consumed
            assert state_digest(next(resumed)) == expected
            assert torch.equal(torch.get_rng_state(), rng)
    assert len(set(rank_first)) == 8


def test_manifest_rejects_content_change(manifest_config):
    config = manifest_config
    Path(config['data_manifest']).write_text('[]\n')
    with pytest.raises(ValueError, match='changed'):
        manifest_stream(config, [Tokenizer(), Tokenizer()],
                        SimpleNamespace(process_index=0, num_processes=1), 0)


def test_resume_allows_only_budget_extension():
    config = dict(max_steps=3125, gradient_accumulation=8, fake_rounds=1, seed=42)
    progress = dict(kind=CHECKPOINT_KIND, update_rule=UPDATE_RULE, config=config,
                    world_size=4, global_step=3125, consumed_batches=50000,
                    microbatches_per_update=16)
    validate_progress(progress, dict(config, max_steps=6250), 4)
    with pytest.raises(ValueError, match='shorten'):
        validate_progress(progress, dict(config, max_steps=3000), 4)
    with pytest.raises(ValueError, match='identical'):
        validate_progress(progress, dict(config, max_steps=6250, seed=43), 4)
