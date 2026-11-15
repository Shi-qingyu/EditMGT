#!/usr/bin/env python3
"""Record aligned text/source tokens at every vanilla EditMGT block and sampling step.

Capture from the conditional CFG branch after each of the 14 two-stream and 28
single-stream blocks. All 32 fixed Inter-Edit examples from the decoding-order
experiment contribute equally to the mean matrices. The per-sample archives
permit exact recomputation without rerunning the model. No edit mask is used.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_METADATA = ROOT / 'outputs/vanilla_decode_order_quant_32/metadata.json'
OUTPUT = ROOT / 'outputs/vanilla_feature_similarity_32'
FIGURES = ROOT / 'docs/feature_similarity'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('all', 'collect', 'render'), default='all')
    p.add_argument('--source-metadata', type=Path, default=SOURCE_METADATA)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--figure-dir', type=Path, default=FIGURES)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--feature-tokens', type=int, default=32,
                   help='Fixed, evenly spaced token indices in each modality; aligned across all blocks and steps.')
    p.add_argument('--dpi', type=int, default=300)
    return p.parse_args()


def source_metadata(args):
    meta = json.loads(args.source_metadata.read_text())
    keys = meta['selection']['selected_keys']
    if len(keys) != 32 or len(meta['samples']) != 32 or {s['key'] for s in meta['samples']} != set(keys):
        raise ValueError('Expected the completed fixed 32-example Inter-Edit selection')
    if meta['protocol']['model_protocol'] != 'native_editmgt_mask_free_no_cache':
        raise ValueError('Expected released vanilla EditMGT with no condition cache')
    return meta, [next(s for s in meta['samples'] if s['key'] == key) for key in keys]


class FeatureCapture:
    def __init__(self, transformer, tokens, valid_text_positions):
        import torch
        self.torch = torch
        self.blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
        self.double_count = len(transformer.transformer_blocks)
        self.tokens = tokens
        self.valid_text_positions = valid_text_positions
        self.handles = []
        self.step = -1
        self.current = []
        self.text = []
        self.reference = []
        self.text_len = None
        self.ref_len = None
        self.indices = {}
        for block_index, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_hook(self.hook(block_index)))

    def sample(self, tensor, name):
        length = tensor.shape[1]
        prev_length = self.text_len if name == 'text' else self.ref_len
        if prev_length is not None and prev_length != length:
            raise RuntimeError(f'Variable {name} token length {prev_length} -> {length}')
        if name == 'text':
            self.text_len = length
        else:
            self.ref_len = length
        if name not in self.indices:
            if name == 'text':
                valid = self.valid_text_positions.to(tensor.device)
                if valid.numel() < 2 or valid.max().item() >= length:
                    raise ValueError('Expected at least two valid in-range instruction tokens')
                which = self.torch.linspace(0, valid.numel() - 1,
                    steps=min(self.tokens, valid.numel()), device=tensor.device).round().long().unique()
                self.indices[name] = valid.index_select(0, which)
            else:
                self.indices[name] = self.torch.linspace(0, length - 1,
                    steps=min(self.tokens, length), device=tensor.device).round().long().unique()
        # CFG order is [negative, positive]; select only the positive instruction.
        return tensor[1].index_select(0, self.indices[name]).detach().clone().to(self.torch.float16)

    def hook(self, index):
        def capture(_module, _inputs, outputs):
            if index == 0:
                if self.step >= 0:
                    self.flush()
                self.step += 1
            if len(self.current) != index:
                raise RuntimeError(f'Unexpected block order at step {self.step}: {index}')
            if index < self.double_count:
                text, _, reference = outputs
            else:
                if self.text_len is None:
                    raise RuntimeError('Missing text length from first two-stream block')
                target_with_text, reference = outputs
                text = target_with_text[:, :self.text_len]
            if text.shape[0] != 2 or reference.shape[0] != 2:
                raise RuntimeError('Expected CFG batch size 2, negative then positive')
            self.current.append((self.sample(text, 'text'), self.sample(reference, 'source')))
        return capture

    def flush(self):
        if len(self.current) != len(self.blocks):
            raise RuntimeError(f'Step {self.step}: got {len(self.current)} / {len(self.blocks)} blocks')
        self.text.append(self.torch.stack([p[0] for p in self.current]))
        self.reference.append(self.torch.stack([p[1] for p in self.current]))
        self.current = []

    def finish(self, expected_steps):
        self.flush()
        if len(self.text) != expected_steps or len(self.reference) != expected_steps:
            raise RuntimeError(f'Observed {len(self.text)} / {expected_steps} denoising steps')
        return self.torch.stack(self.text), self.torch.stack(self.reference)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def similarity(v):
    """Mean matched-token cosine across steps/layers, for one example and modality."""
    import torch.nn.functional as F
    # [S,L,T,D], per token normalization; never pool unlike token vectors.
    v = F.normalize(v.float(), dim=-1)
    steps, layers, tokens, dim = v.shape
    by_step = v.reshape(steps, layers, tokens * dim)
    layer_sim = (by_step @ by_step.transpose(1, 2)) / tokens  # [S,L,L]
    by_layer = v.permute(1, 0, 2, 3).reshape(layers, steps, tokens * dim)
    step_sim = (by_layer @ by_layer.transpose(1, 2)) / tokens  # [L,S,S]
    return {
        'block_similarity': layer_sim.mean(0).clamp(-1,1).cpu().numpy(),
        'step_similarity': step_sim.mean(0).clamp(-1,1).cpu().numpy(),
        'adjacent_block_by_step': layer_sim.diagonal(offset=1, dim1=1, dim2=2).clamp(-1,1).cpu().numpy(),
        'adjacent_step_by_block': step_sim.diagonal(offset=1, dim1=1, dim2=2).clamp(-1,1).cpu().numpy().T,
    }


def collect(args, meta, rows):
    import torch
    from PIL import Image
    from src.editmgt import init_edit_mgt
    protocol = meta['protocol']
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable for feature capture')
    if args.feature_tokens < 2:
        raise ValueError('Need at least two aligned token positions')
    model_path = protocol['model_path']
    if not Path(model_path).is_dir():
        raise FileNotFoundError(model_path)
    pipe = init_edit_mgt(args.device, enable_bf16=True, base_model_path=model_path,
                         condition_cache_threshold=None)
    for module in (pipe.transformer, pipe.vqvae, pipe.text_encoder, pipe.text_encoder_t5):
        if module is not None:
            module.eval().requires_grad_(False)
    pipe.local_guidance = 0.0
    pipe.set_progress_bar_config(disable=True)
    completed = []
    try:
        with torch.inference_mode():
            for order, row in enumerate(rows, 1):
                file = out / f"{row['key']}.npz"
                if file.is_file():
                    with np.load(file) as existing:
                        if (int(existing['steps']) != protocol['num_inference_steps']
                                or int(existing['feature_tokens']) != args.feature_tokens
                                or str(existing['source_sha256']) != row['artifact_sha256'][row['source']]):
                            raise ValueError(f'Existing incompatible result: {file}')
                    completed.append(row['key'])
                    print(f'[{order}/{len(rows)}] reuse {row["key"]}', flush=True)
                    continue
                source_file = args.source_metadata.parent / row['source']
                sha = hashlib.sha256(source_file.read_bytes()).hexdigest()
                if sha != row['artifact_sha256'][row['source']]:
                    raise ValueError(f'Source image changed: {source_file}')
                source = Image.open(source_file).convert('RGB')
                tokenized = pipe.tokenizer_t5([row['instruction']], truncation=True,
                    padding='max_length', max_length=256, return_tensors='pt')
                valid_text_positions = tokenized.attention_mask[0].nonzero(as_tuple=True)[0]
                capture = FeatureCapture(pipe.transformer, args.feature_tokens, valid_text_positions)
                try:
                    pipe(prompt=[row['instruction']], reference_image=[source], mask_image=None,
                         height=protocol['resolution'], width=protocol['resolution'],
                         num_inference_steps=protocol['num_inference_steps'],
                         guidance_scale=protocol['guidance_scale'],
                         reference_strength=protocol['reference_strength'],
                         negative_prompt=protocol['negative_prompt'],
                         generator=torch.Generator(device=args.device).manual_seed(row['seed']),
                         edit_only_diffusion=False, sparse_target=False, lora_part_enable=False,
                         preserve_unmasked_pixels=False, use_reference_kv_cache=False,
                         use_condition_kv_cache=False, output_type='latent')
                    text, reference = capture.finish(protocol['num_inference_steps'])
                    matrices = {'text_' + k: v for k, v in similarity(text).items()}
                    matrices.update({'source_' + k: v for k, v in similarity(reference).items()})
                    matrices.update(steps=protocol['num_inference_steps'], feature_tokens=args.feature_tokens,
                                    source_sha256=sha, text_token_length=capture.text_len,
                                    source_token_length=capture.ref_len,
                                    text_token_indices=capture.indices['text'].cpu().numpy(),
                                    source_token_indices=capture.indices['source'].cpu().numpy(),
                                    valid_text_tokens=valid_text_positions.numel())
                    np.savez_compressed(file, **matrices)
                finally:
                    capture.close()
                    del capture
                del text, reference
                torch.cuda.empty_cache()
                completed.append(row['key'])
                print(f'[{order}/{len(rows)}] saved {file.name}', flush=True)
    finally:
        del pipe
    return completed


def render(args, meta, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    arrays = {}
    for row in rows:
        f = args.output_dir / f"{row['key']}.npz"
        if not f.is_file():
            raise FileNotFoundError(f'Incomplete capture: {f}')
        with np.load(f) as item:
            if int(item['feature_tokens']) != args.feature_tokens:
                raise ValueError(f'Feature token protocol mismatch: {f}')
            for modality in ('text', 'source'):
                for name in ('block_similarity','step_similarity','adjacent_block_by_step','adjacent_step_by_block'):
                    key = f'{modality}_{name}'
                    arrays.setdefault(key, []).append(item[key].astype(np.float64))
    means = {k: np.stack(v).mean(axis=0) for k,v in arrays.items()}
    sample_stats = {}
    for modality in ('text', 'source'):
        for name in ('adjacent_block_by_step', 'adjacent_step_by_block'):
            key = f'{modality}_{name}'
            per_sample = np.stack(arrays[key]).mean(axis=(1,2))
            sample_stats[key] = dict(mean=float(per_sample.mean()), std=float(per_sample.std(ddof=1)),
                                     min=float(per_sample.min()), max=float(per_sample.max()))
        for name in ('block_similarity', 'step_similarity'):
            key = f'{modality}_{name}'
            per_sample = np.stack(arrays[key]); size = per_sample.shape[-1]
            off_diagonal = np.triu_indices(size, k=1)
            per_sample = per_sample[:,off_diagonal[0],off_diagonal[1]].mean(1)
            sample_stats[key + '_off_diagonal'] = dict(mean=float(per_sample.mean()),
                std=float(per_sample.std(ddof=1)))
    for k,v in means.items():
        if not np.isfinite(v).all() or v.min() < -1.01 or v.max() > 1.01:
            raise ValueError(f'Invalid cosine matrix: {k}')
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    # Each row has its own shared color scale; never rescale cells independently.
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.1), constrained_layout=True)
    for ri,(metric,label) in enumerate((('block_similarity','Transformer block'),
                                        ('step_similarity','Sampling step'))):
        mats = [means[f'{mode}_{metric}'] for mode in ('text','source')]
        smallest = min(np.min(m) for m in mats)
        norm = Normalize(vmin=max(-1., np.floor((smallest - .005)*20)/20), vmax=1.)
        for ci,(mode,title) in enumerate((('text','Text features'),('source','Source image features'))):
            ax = axes[ri,ci]
            matrix = means[f'{mode}_{metric}']
            img = ax.imshow(matrix, vmin=norm.vmin, vmax=norm.vmax,
                            cmap='magma', origin='lower', interpolation='nearest', aspect='equal')
            ax.set(title=title if ri == 0 else None, xlabel=label,
                   ylabel=label if ci == 0 else None)
            if ri == 0:
                ax.axvline(13.5, color='#81d4e6', alpha=.8, linewidth=.85, linestyle='--')
                ax.axhline(13.5, color='#81d4e6', alpha=.8, linewidth=.85, linestyle='--')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            n = matrix.shape[0]
            ax.set_xticks(np.linspace(0,n-1,6).round().astype(int),
                          labels=np.linspace(1,n,6).round().astype(int))
            ax.set_yticks(np.linspace(0,n-1,6).round().astype(int),
                          labels=np.linspace(1,n,6).round().astype(int))
            adj = sample_stats[f'{mode}_adjacent_{"block_by_step" if ri == 0 else "step_by_block"}']['mean']
            ax.text(.03,.97, f'Adjacent cosine: {adj:.3f}', transform=ax.transAxes,
                    va='top', ha='left', fontsize=9, color='white',
                    bbox=dict(facecolor='black',alpha=.6,edgecolor='none',pad=3))
        fig.colorbar(img, ax=axes[ri,:], shrink=.85, pad=.02, label='Mean matched-token cosine similarity')
    paths = {}
    for suffix in ('pdf','svg','png'):
        path = args.figure_dir / f'vanilla_condition_similarity.{suffix}'
        fig.savefig(path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        paths[suffix] = str(path)
    plt.close(fig)
    # Detailed trends can be used as a separate supplementary figure.
    fig, axes = plt.subplots(1,2,figsize=(11.5,3.6),constrained_layout=True)
    colors = dict(text='#2673ad',source='#d26930')
    for mode,label in (('text','Text'),('source','Source image')):
        b = means[f'{mode}_adjacent_block_by_step'].mean(axis=0)
        s = means[f'{mode}_adjacent_step_by_block'].mean(axis=1)
        axes[0].plot(np.arange(1,len(b)+1),b,label=label,color=colors[mode],lw=2)
        axes[1].plot(np.arange(1,len(s)+1),s,label=label,color=colors[mode],lw=2)
    axes[0].axvline(14,color='#6a6a6a',lw=1,linestyle='--',alpha=.7)
    axes[0].set(xlabel='Block pair (i, i + 1)',ylabel='Mean cosine similarity',xlim=(1,41),ylim=(.67,1.01))
    axes[1].set(xlabel='Step pair (t, t + 1)',ylabel='Mean cosine similarity',xlim=(1,35),ylim=(.975,1.001))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    for suffix in ('pdf','svg','png'):
        fig.savefig(args.figure_dir / f'vanilla_condition_adjacent.{suffix}',dpi=args.dpi,
                    bbox_inches='tight',facecolor='white')
    plt.close(fig)
    np.savez_compressed(args.figure_dir/'vanilla_condition_similarity_matrices.npz',**means)
    summary = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
        source_metadata=str(args.source_metadata.resolve()),
        selection=dict(benchmark='Inter-Edit', split='test', language='en', count=len(rows),
                       keys=[s['key'] for s in rows]),
        model='released vanilla EditMGT', config={k:meta['protocol'][k] for k in
            ('model_path','resolution','num_inference_steps','guidance_scale','reference_strength','negative_prompt','dtype')},
        measurement=dict(branch='positive instruction CFG branch (batch index 1)',
            layer_output='post-block residual hidden states; 14 two-stream + 28 single-stream',
            alignment='same token indices across each pair of blocks/steps; L2 normalize each token before dot product',
            sampling='up to 32 evenly spaced non-padding instruction tokens and 32 evenly spaced source image tokens; same indices at all blocks and steps of each image',
            tokens_per_modality=args.feature_tokens, aggregation='per-token mean, then layer/step mean, then image mean',
            cache=False, mask='not supplied'),
        per_image_stats=sample_stats, matrices_file='vanilla_condition_similarity_matrices.npz',
        figures=paths)
    (args.figure_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('Mean similarities:',json.dumps(sample_stats,indent=2),flush=True)
    print('Figure:',paths['pdf'],flush=True)


def main():
    args = parse_args()
    meta, rows = source_metadata(args)
    if args.stage != 'render':
        collect(args, meta, rows)
    if args.stage != 'collect':
        render(args, meta, rows)


if __name__ == '__main__':
    main()
