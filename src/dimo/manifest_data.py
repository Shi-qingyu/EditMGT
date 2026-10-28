"""Read a pinned, already materialized editing manifest without altering it."""
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import IterableDataset

from src.dataset_utils import process_edit_sample, tokenize_prompt
from src.dimo.data import ReplayStream


class ManifestDataset(IterableDataset):
    def __init__(self, config, tokenizer, rank, world_size, consumed=0):
        super().__init__()
        self.config, self.tokenizer = config, tokenizer
        self.rank, self.world_size = rank, world_size
        self.path = Path(config['data_manifest']).resolve()
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != config['data_manifest_sha256']:
            raise ValueError('Fixed training manifest changed')
        self.rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        counts = dict(Counter(row['dataset_name'] for row in self.rows))
        if counts != config['expected_dataset_counts']:
            raise ValueError(f'Unexpected fixed manifest composition: {counts}')
        if len(self.rows) % world_size:
            raise ValueError('Fixed manifest must divide evenly across ranks')
        if len({row['sample_uid'] for row in self.rows}) != len(self.rows):
            raise ValueError('Duplicate sample IDs in training manifest')
        self.local_length = len(self.rows) // world_size
        self.start = consumed * config['micro_batch']
        self.roots = {key: Path(value).resolve() for key, value in config['data_roots'].items()}
        self.tokens = {}
        if not self.rows:raise ValueError('Empty training manifest')
        for row in self.rows:
            if not row.get('instruction_en','').strip():raise ValueError('Missing English instruction')
            for key in ('source_locator','target_locator','region_locator'):
                path=self.asset_path(row,key)
                if not path.is_file():raise FileNotFoundError(path)

    def __len__(self):
        return self.local_length

    def asset_path(self, row, key):
        locator = row[key]
        if locator['backend'] != 'file':
            raise ValueError('This run requires already materialized file assets')
        root = self.roots[row['dataset_name']]
        path = (root / locator['relative_path']).resolve()
        path.relative_to(root)
        return path

    def process(self, index, epoch):
        row = self.rows[index]
        instruction = row.get('instruction_en', '').strip()
        if not instruction:
            raise ValueError(f'Missing English instruction: {row["sample_uid"]}')
        # These region.png assets are already binary, white=edit masks. In
        # particular, never invert MagicBrush alpha a second time here.
        def read_image(key, digest_key):
            payload = self.asset_path(row, key).read_bytes()
            if self.config.get('verify_asset_hashes', False):
                if hashlib.sha256(payload).hexdigest() != row[digest_key]:
                    raise ValueError(f'Pinned asset checksum changed: {row["sample_uid"]}, {key}')
            return Image.open(io.BytesIO(payload))

        with read_image('source_locator', 'source_sha256') as source, \
                read_image('target_locator', 'target_sha256') as target, \
                read_image('region_locator', 'region_sha256') as mask:
            sample_seed = int.from_bytes(hashlib.sha256(
                f'{self.config["seed"]}:{epoch}:{row["sample_uid"]}'.encode()).digest()[:8], 'big') % (2**63)
            # Each sample owns its crop RNG, making a saved data offset seekable
            # without decoding all previously consumed images or touching CUDA RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(sample_seed)
                sample = process_edit_sample(target, source, mask,
                    self.config['resolution'], ensure_edit_in_crop=True)
        if not sample['edit_mask'].any():
            raise ValueError(f'Empty mask in pinned training sample: {row["sample_uid"]}')
        if instruction not in self.tokens:
            self.tokens[instruction] = [value[0] for value in
                tokenize_prompt(self.tokenizer, instruction, 'CLIP_Gemma2')]
        sample.update(prompt_input_ids=self.tokens[instruction], instruction=instruction,
                      sample_id=row['sample_uid'], dataset_name=row['dataset_name'])
        return sample

    def __iter__(self):
        epoch, offset = divmod(self.start, self.local_length)
        while True:
            indices = list(range(len(self.rows)))
            random.Random(self.config['seed'] + epoch).shuffle(indices)
            local = indices[self.rank::self.world_size]
            for index in local[offset:]:
                yield self.process(index, epoch)
            epoch, offset = epoch + 1, 0


def manifest_stream(config, tokenizer, acc, consumed):
    dataset = ManifestDataset(config, tokenizer, acc.process_index, acc.num_processes, consumed)
    generator = torch.Generator().manual_seed(config['seed'] + acc.process_index)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['micro_batch'],
                                       num_workers=0, generator=generator)
    stream = ReplayStream(loader, config['seed'] + acc.process_index)
    stream.consumed = consumed
    return stream
