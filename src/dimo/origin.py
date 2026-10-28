"""Strict SFT origin selection; never searches outside the requested run."""
import hashlib
import json
import re
import pickle
import subprocess
import warnings
from pathlib import Path
from safetensors import SafetensorError

PROFILE = dict(sparse_target=True, sparse_target_halo=0, supervise_target_halo=False,
               cache_conditioning=True, time_conditioning='mask_ratio')


def git_revision_or_none():
    try:
        return subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        # A source ZIP has no Git HEAD. Preserve archive and actual source hashes
        # rather than inventing a commit or requiring an unrelated Git checkout.
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def weight_files(folder):
    folder = Path(folder)
    indexes = list(folder.glob('*.index.json'))
    if len(indexes) > 1:
        raise ValueError('Ambiguous weight indexes')
    if indexes:
        names = sorted(set(json.loads(indexes[0].read_text())['weight_map'].values()))
        files = [folder / n for n in names]
        if any(p.parent.resolve() != folder.resolve() for p in files):
            raise ValueError('Weight shard escapes transformer directory')
        files += indexes
    else:
        files = [folder / n for n in ('diffusion_pytorch_model.safetensors', 'diffusion_pytorch_model.bin')
                 if (folder / n).is_file()]
        if len(files) != 1:
            raise ValueError('Expected exactly one complete Transformer weight set')
    if not files or any(not p.is_file() or not p.stat().st_size for p in files):
        raise ValueError('Missing/empty Transformer weights')
    return files


def load_strict(folder):
    from src.transformer import Transformer2DModel
    import torch
    from safetensors.torch import load_file
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    model = Transformer2DModel.from_config(config)
    expected, seen = set(model.state_dict()), set()
    files = weight_files(folder)
    indexes = [p for p in files if p.name.endswith('.index.json')]
    index = json.loads(indexes[0].read_text())['weight_map'] if indexes else None
    for path in files:
        if path in indexes:
            continue
        state = load_file(str(path), device='cpu') if path.suffix == '.safetensors' else torch.load(
            path, map_location='cpu', weights_only=True)
        if seen.intersection(state) or set(state) - expected:
            raise ValueError(f'Duplicate/unexpected Transformer keys in {path}')
        if index is not None and {k for k, v in index.items() if v == path.name} != set(state):
            raise ValueError(f'Shard keys differ from index: {path}')
        # Load one shard at a time; strict completeness is checked over their union.
        model.load_state_dict(state, strict=False)
        seen.update(state)
        del state
    if expected != seen:
        raise ValueError(f'Missing Transformer keys: {sorted(expected - seen)}')
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite Transformer weights')
    return model


def _validate_folder(folder, base, expected, loader):
    folder, base = Path(folder).resolve(), Path(base).resolve()
    config = json.loads((folder / 'config.json').read_text())
    baseline = json.loads((base / 'editmgt/config.json').read_text())
    vq = json.loads((base / 'vqvae/config.json').read_text())
    scheduler = json.loads((base / 'scheduler/scheduler_config.json').read_text())
    if scheduler['mask_token_id'] != baseline['vocab_size'] - 1:
        raise ProfileMismatch('Base Scheduler mask token ID differs from base Transformer')
    mismatch = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if mismatch: raise ProfileMismatch(f'{folder}: profile differs: {mismatch}')
    for key in ('codebook_size', 'vocab_size'):
        if config[key] != baseline[key]: raise ProfileMismatch(f'{key} differs from base')
    if config['codebook_size'] != vq['num_vq_embeddings'] or config['vocab_size'] - 1 < config['codebook_size']:
        raise ProfileMismatch('VQ codebook or mask token ID incompatible')
    # Full merged Transformer only: no inference adapter reconstruction or
    # silent change of lora_part_enable / time / cache profile.
    if (folder / 'adapter_config.json').exists(): raise ProfileMismatch('Provide full merged Transformer weights')
    files = weight_files(folder)
    checksums = {p.name: sha256(p) for p in [folder / 'config.json', *files]}
    model = loader(folder)
    if any(sha256(folder / name) != digest for name,digest in checksums.items()):
        raise ValueError('Teacher changed during strict loading')
    git_sha = git_revision_or_none()
    origin = dict(transformer_dir=str(folder),transformer_config=config,git_sha=git_sha,
                  source_kind='git_worktree' if git_sha else 'source_snapshot',checksums=checksums,
                  mask_token_id=config['vocab_size']-1,profile={k:config[k] for k in expected})
    return model,origin


def discover(run, base, expected=None, loader=load_strict):
    """Largest complete checkpoint within this SFT run only; incomplete skips logged."""
    run=Path(run).resolve()
    expected=PROFILE if expected is None else expected
    if not run.is_dir():raise FileNotFoundError(run)
    candidates=sorted((int(m[1]),p) for p in run.glob('checkpoint-*')
                      if p.is_dir() and (m:=re.fullmatch(r'checkpoint-(\d+)',p.name)))
    rejected=[]
    for step,checkpoint in reversed(candidates):
        folder=checkpoint/'transformer'
        try:
            if checkpoint.resolve().parent!=run or folder.resolve().parent!=checkpoint.resolve():
                raise ProfileMismatch(f'SFT candidate escapes selected run: {checkpoint}')
            model,origin=_validate_folder(folder,base,expected,loader)
            origin.update(SFT_RUN_DIR=str(run),checkpoint=str(checkpoint.resolve()),step=step,
                          teacher_source_type='sft_run',rejected_candidates=rejected)
            if rejected:warnings.warn(f'Skipped incomplete SFT checkpoints: {rejected}',stacklevel=2)
            return model,origin
        except ProfileMismatch:raise
        except (OSError,ValueError,RuntimeError,KeyError,SafetensorError,pickle.UnpicklingError,EOFError) as e:
            rejected.append(dict(checkpoint=str(checkpoint),reason=str(e)))
    raise RuntimeError(f'No complete compatible SFT checkpoint in {run}: {rejected}')


def select_teacher(config, loader=load_strict):
    """Resolve exactly one explicit source, never falling back to another model."""
    sources=[k for k in ('sft_run_dir','teacher_transformer_dir','teacher_hf') if config.get(k)]
    if len(sources)!=1:raise ValueError('Select exactly one explicit Teacher source')
    expected=config['expected_profile']
    if sources==['sft_run_dir']:
        return discover(config['sft_run_dir'],config['base_model_dir'],expected,loader)
    if sources==['teacher_transformer_dir']:
        folder=Path(config['teacher_transformer_dir']).resolve()
        # Accept the explicit checkpoint-N directory as well as transformer/.
        if not (folder/'config.json').is_file() and (folder/'transformer/config.json').is_file():
            folder=folder/'transformer'
        model,origin=_validate_folder(folder,config['base_model_dir'],expected,loader)
        origin.update(teacher_source_type='transformer_dir',checkpoint=str(folder))
        return model,origin
    spec=config['teacher_hf']
    revision=spec.get('revision','')
    if not re.fullmatch(r'[0-9a-f]{40}',revision):raise ValueError('HF revision must be a fixed 40-character SHA')
    from huggingface_hub import snapshot_download
    snapshot=Path(snapshot_download(repo_id=spec['repo_id'],revision=revision,
                  cache_dir=spec.get('cache_dir'),local_files_only=True)).resolve()
    if snapshot.name!=revision:raise ValueError('HF snapshot is not pinned to the requested revision')
    if config.get('output_dir'):
        out=Path(config['output_dir']).resolve()
        if out==snapshot or snapshot in out.parents or out in snapshot.parents:
            raise ValueError('output_dir must be disjoint from the resolved HF snapshot')
    if spec.get('transformer_subdir'):
        folder=(snapshot/spec['transformer_subdir']).resolve()
        folder.relative_to(snapshot)
    else:
        # Inspect the snapshot layout. SparseEdit currently has
        # checkpoint-N/transformer; other releases may store editmgt/ directly.
        candidates=[p.parent for p in snapshot.rglob('config.json')
                    if p.parent.name in ('transformer','editmgt')]
        if len(candidates)!=1:raise ValueError('Ambiguous HF layout; set transformer_subdir explicitly')
        folder=candidates[0]
    model,origin=_validate_folder(folder,config['base_model_dir'],expected,loader)
    origin.update(teacher_source_type='hf_snapshot',hf_repo_id=spec['repo_id'],hf_revision=revision,
                  snapshot_dir=str(snapshot),checkpoint=str(folder))
    return model,origin


class ProfileMismatch(ValueError):
    pass


def verify_origin(origin):
    folder = Path(origin['transformer_dir'])
    for name, digest in origin['checksums'].items():
        if sha256(folder / name) != digest:
            raise ValueError(f'Pinned Teacher changed: {folder / name}')
    return load_strict(folder)
