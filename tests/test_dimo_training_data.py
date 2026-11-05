"""Exercise the SFT-compatible mixture and exact DiMO stream continuation."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from src.dimo.checkpoint import state_digest
from src.dimo.config import resolve_config, validate_config
from src.dimo.train import dataset_stream
from test_inter_edit_dataset import inter_edit, write_test_translations
from test_magicbrush_masks import magicbrush_cutout_data
from test_training_data import maskedit_data

ROOT = Path(__file__).resolve().parents[1]
PRESET = ROOT / 'configs/dimo/region_dimo_inter_brush_maskedit.json'


@pytest.fixture
def stream_config(inter_edit, magicbrush_cutout_data, maskedit_data, tmp_path):
    inter, cache = inter_edit
    magic = magicbrush_cutout_data[0]
    base = tmp_path / 'base'
    base.mkdir()
    c = json.loads(PRESET.read_text())
    c.update(base_model_dir=str(base), teacher_transformer_dir=str(tmp_path / 'teacher'),
             output_dir=str(tmp_path / 'dimo'), resolution=16, micro_batch=2,
             inter_edit_root=str(inter), magicbrush_root=str(magic), maskedit_root=str(maskedit_data),
             cache_dir=str(cache),
             inter_edit_translations_dir=str(write_test_translations(inter, tmp_path / 'translations')))
    return c


class Tokenizer:
    def __call__(self, instruction, max_length, **kwargs):
        return SimpleNamespace(input_ids=torch.full((1, max_length), len(instruction), dtype=torch.long))


def test_three_datasets_translate_tokenize_and_replay_exactly(stream_config):
    c = stream_config
    validate_config(c)
    acc = SimpleNamespace(process_index=0, num_processes=1)
    tokenizer = [Tokenizer(), Tokenizer()]
    stream = dataset_stream(c, tokenizer, acc, 0)
    observed = Counter()
    for _ in range(80):
        batch = next(stream)
        observed.update(batch['dataset_name'])
        assert batch['edit_mask'].shape == (2, 1, 16, 16)
        assert [tokens.shape for tokens in batch['prompt_input_ids']] == [(2, 77), (2, 256)]
        for name, instruction in zip(batch['dataset_name'], batch['instruction']):
            if name == 'Inter-Edit':
                assert instruction.startswith('Add object ')
    assert set(observed) == {'Inter-Edit', 'MagicBrush', 'MaskEdit-10K'}
    expected = state_digest(next(stream))
    rng = torch.get_rng_state().clone()
    replay = dataset_stream(c, tokenizer, acc, 80)
    assert torch.equal(torch.get_rng_state(), rng)
    assert state_digest(next(replay)) == expected
    assert torch.equal(torch.get_rng_state(), rng)
    info = json.loads((Path(c['output_dir']) / 'data_mixture/rank_0000.json').read_text())
    assert [source['probability'] for source in info['sources']] == pytest.approx([.9, .05, .05])
    assert info['sources'][0]['exclude_text_editing'] is True
    assert info['sources'][0]['translated_prompt_count_global'] == 8
    assert info['sources'][2]['configs'] == ['scene', 'infographics_en']


def test_historical_two_dataset_stream_still_works(stream_config):
    c = stream_config
    for key in ('maskedit_root', 'maskedit_probability', 'inter_edit_translations_dir',
                'exclude_inter_edit_text_editing'):
        c.pop(key)
    c.update(language='all', magicbrush_probability=.2)
    validate_config(c)
    next(dataset_stream(c, None, SimpleNamespace(process_index=0, num_processes=1), 0))
    info = json.loads((Path(c['output_dir']) / 'data_mixture/rank_0000.json').read_text())
    assert [source['probability'] for source in info['sources']] == pytest.approx([.8, .2])


@pytest.mark.parametrize('settings,match', [
    ({'maskedit_probability': .95}, 'sum to less than 1'),
    ({'maskedit_probability': float('nan')}, 'probability'),
    ({'maskedit_probability': 0}, 'probability'),
    ({'maskedit_root': '/missing-maskedit'}, 'missing-maskedit'),
    ({'inter_edit_translations_dir': '/missing-translations'}, 'missing-translations'),
    ({'exclude_inter_edit_text_editing': 'yes'}, 'must be boolean'),
    ({'maskedit_configs': ['infographics_cn']}, 'non-Chinese'),
    ({'maskedit_read_batch_size': 0}, 'positive integer'),
])
def test_invalid_local_mixture_fails_before_models_load(stream_config, settings, match):
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        validate_config(dict(stream_config, **settings))


def test_environment_overrides_resolve_new_data_paths(tmp_path):
    c = resolve_config(PRESET, {'MASKEDIT_ROOT': str(tmp_path / 'mask'),
                               'INTER_EDIT_TRANSLATIONS_DIR': str(tmp_path / 'translations')})
    assert c['maskedit_root'] == str(tmp_path / 'mask')
    assert c['inter_edit_translations_dir'] == str(tmp_path / 'translations')
    assert Path(c['teacher_transformer_dir']).name == 'checkpoint-8000'


@pytest.mark.parametrize('mode', ['--check-only', '--stop-after-preflight', '--resume'])
def test_preset_launcher_forwards_config_and_mode_from_other_directory(tmp_path, mode):
    shim = tmp_path / 'python'
    shim.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
if sys.argv[1:2] == ['-c']: print(2)
else: print(json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd(), visible=os.environ.get('CUDA_VISIBLE_DEVICES'))))
''')
    shim.chmod(0o755)
    env = dict(os.environ, DIMO_PYTHON=str(shim), CUDA_VISIBLE_DEVICES='3,5', NUM_PROCESSES='1')
    env.pop('GPU_IDS', None)
    env.pop('DIMO_CONFIG', None)
    args = [mode, 'latest'] if mode == '--resume' else [mode]
    result = subprocess.run(['bash', str(ROOT / 'scripts/dimo/train_inter_brush_maskedit.sh'), *args],
                            cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    captured = json.loads(result.stdout.splitlines()[-1])
    argv = captured['argv']
    assert captured['cwd'] == str(ROOT)
    assert captured['visible'] == '3,5'
    assert argv[argv.index('--config') + 1] == str(PRESET)
    assert argv[1] == ('src.dimo.preflight' if mode == '--check-only' else 'src.dimo.train')
    if mode != '--check-only':
        assert argv[-len(args):] == args
