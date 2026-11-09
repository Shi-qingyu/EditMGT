#!/usr/bin/env python3
r"""Visualize the order of committed tokens during dense MGM image editing.

Examples (from the repository root)::

    python scripts/plot_mgm_motivation.py --data-dir /path/to/Inter-Edit --num-samples 8
    python scripts/plot_mgm_motivation.py --source source.png --mask mask.png \
        --instruction 'Replace the red bells with gold ones.' --num-samples 1
    python scripts/plot_mgm_motivation.py --stage render --visual-sample sample_02832

Decode all 36 steps and show snapshots after steps 6, 12, 18, 24, 30 and 36.
Snapshots decode the actual sampler state; unresolved mask-token cells are
painted solid gray. Tentative predictions for unresolved tokens are not shown.

Outputs under docs/motivation_token_order/: motivation.{pdf,svg,png}, an English caption,
exact per-image curve data, metadata, and source/mask/step images plus token
trajectories for every sample. Use --resume to continue interrupted capture.
Change --output-dir for a different sampling or model protocol. With
--stage render, --visual-sample changes the image strip only; curves still
include all recorded examples in the separately exported numerical data.

The default dataset is Inter-Edit test/en, read directly from its official
metadata and source/mask archives. The annotation mask is NEVER passed to the
generative model or sampler. All target tokens start masked. Capturing scheduler
inputs does not consume RNG or change sampling. Rendering requires no torch/CUDA.
"""

import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INTER_EDIT = Path('/mnt/geminihzceph/public/data/Inter-Edit')
DEFAULT_OUTPUT = ROOT / 'docs/motivation_token_order'
NEGATIVE_PROMPT = ('worst quality, low quality, low res, blurry, distortion, watermark, logo, '
                   'signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('all', 'collect', 'render'), default='all')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_INTER_EDIT,
                        help='Inter-Edit root or its test/ directory; default input is the official test split.')
    parser.add_argument('--language', choices=('en', 'zh', 'all'), default='en')
    parser.add_argument('--input-manifest', type=Path, default=None,
                        help='Benchmark selection.json, or a list/{samples: [...]} of source/mask/instruction records.')
    parser.add_argument('--source', type=Path, help='One explicit source image, instead of a manifest.')
    parser.add_argument('--mask', type=Path, help='Binary annotation mask: white=edit; used ONLY for analysis.')
    instruction = parser.add_mutually_exclusive_group()
    instruction.add_argument('--instruction')
    instruction.add_argument('--instruction-file', type=Path)
    parser.add_argument('--sample-keys', nargs='+', help='Explicit manifest keys; overrides hash selection.')
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--selection-seed', type=int, default=2026)
    parser.add_argument('--visual-sample', help='Key of the selected example displayed in the token-order figure.')
    parser.add_argument('--display-steps', type=int, nargs='+',
                        help='One-based snapshot steps. Default: 6, 12, 18, 24, 30, 36.')
    parser.add_argument('--snapshot-every', type=int, default=6,
                        help='Save actual committed-token states every this many forwards, plus the final step.')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'checkpoint/EditMGT')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--num-inference-steps', type=int, default=36)
    parser.add_argument('--guidance-scale', type=float, default=6.0)
    parser.add_argument('--reference-strength', type=float, default=1.1)
    parser.add_argument('--negative-prompt', default=NEGATIVE_PROMPT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--entropy-chunk-size', type=int, default=256,
                        help='Spatial token chunk size for exact float32 predictive entropy.')
    parser.add_argument('--tar-cache-dir', type=Path, default=ROOT / 'data/inter_edit_tar_index')
    parser.add_argument('--resume', action='store_true', help='Resume only an identical capture protocol and selection.')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args(argv)
    if args.stage != 'render':
        if args.source:
            if not args.mask or not (args.instruction or args.instruction_file):
                parser.error('--source requires --mask and --instruction/--instruction-file')
            if args.input_manifest or args.sample_keys:
                parser.error('--source cannot be combined with --input-manifest or --sample-keys')
        elif args.mask or args.instruction or args.instruction_file:
            parser.error('--mask/--instruction require --source')
        if args.resolution < 32 or args.resolution % 32:
            parser.error('--resolution must be a positive multiple of 32')
        if (args.num_samples < 1 or args.num_inference_steps < 2
                or args.entropy_chunk_size < 1 or args.snapshot_every < 1):
            parser.error('Use positive sample/chunk counts and at least two decoding steps')
        if not 1 <= args.guidance_scale < float('inf') or not 0 < args.reference_strength < float('inf'):
            parser.error('Use finite guidance >=1 and finite reference strength >0')
    if args.dpi < 72:
        parser.error('--dpi must be >=72')
    if args.display_steps is not None:
        if any(step < 1 for step in args.display_steps):
            parser.error('--display-steps are one-based positive integers')
        args.display_steps = sorted(set(args.display_steps))
    return args


def select_rows(rows, count, seed, keys=None):
    """Choose examples independently of model outputs or quality scores."""
    by_key = {}
    for row in rows:
        key = str(row.get('key', row.get('sample_key', row.get('name', ''))))
        if not key or key in by_key:
            raise ValueError(f'Missing or duplicate sample key: {key!r}')
        if not isinstance(row.get('instruction'), str) or not row['instruction'].strip():
            raise ValueError(f'Missing editing instruction: {key}')
        by_key[key] = dict(row, key=key)
    if keys:
        if len(set(keys)) != len(keys):
            raise ValueError('--sample-keys must be unique')
        missing = set(keys) - by_key.keys()
        if missing:
            raise ValueError(f'Unknown sample keys: {sorted(missing)}')
        return [by_key[key] for key in keys]
    rank = lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()
    return [by_key[key] for key in sorted(by_key, key=rank)[:count]]


def input_rows(args):
    if args.source:
        instruction = (args.instruction_file.read_text(encoding='utf-8').strip()
                       if args.instruction_file else args.instruction.strip())
        row = dict(key=args.source.stem, instruction=instruction,
                   source=str(args.source.resolve()), mask=str(args.mask.resolve()))
        return [row], ROOT, dict(strategy='explicit_single_image', candidate_count=1)
    if args.input_manifest is None:
        from src.inter_edit_dataset import InterEditReader, matches_inter_edit_language

        reader = InterEditReader(args.data_dir.resolve(), 'test', args.tar_cache_dir)
        rows = []
        for position, row in enumerate(reader.test_rows):
            if not matches_inter_edit_language(row, args.language):
                continue
            index = int(row.get('index', position))
            source = dict(kind='tar', root=str(reader.root.resolve()),
                          archive='source_images.tar', member=row['source_path'])
            mask = dict(kind='tar', root=str(reader.root.resolve()),
                        archive='masks.tar', member=row['mask_path'])
            rows.append(dict(key=f'sample_{index:05d}', seed_key=f'inter_edit:{index}',
                             instruction=row['instruction'], edit_type=row.get('edit_type'),
                             source=source, mask=mask))
        if not rows:
            raise ValueError(f'No Inter-Edit test examples match --language {args.language}')
        selected = select_rows(rows, args.num_samples, args.selection_seed, args.sample_keys)
        metadata_path = reader.metadata_files[0]
        return selected, reader.root, dict(
            strategy='explicit_keys' if args.sample_keys else 'sha256_rank_independent_of_model_quality',
            benchmark='Inter-Edit', split='test', language=args.language,
            metadata=str(metadata_path.resolve()), metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            candidate_count=len(rows), selection_seed=args.selection_seed)
    path = args.input_manifest.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{path}: supply --input-manifest or --source/--mask/--instruction')
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload.get('samples') if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must contain a nonempty list of source/mask/instruction examples')
    selected = select_rows(rows, args.num_samples, args.selection_seed, args.sample_keys)
    selection = dict(
        strategy='explicit_keys' if args.sample_keys else 'sha256_rank_independent_of_model_quality',
        manifest=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        candidate_count=len(rows), selection_seed=args.selection_seed)
    if isinstance(payload, dict):
        selection.update({name: payload[name] for name in ('benchmark', 'split', 'language')
                          if name in payload})
    return selected, path.parent, selection


def display_steps(total, explicit=None, every=6):
    if type(every) is not int or every < 1:
        raise ValueError('Snapshot interval must be a positive integer')
    values = explicit if explicit is not None else list(range(every, total + 1, every)) + [total]
    if not values or any(type(step) is not int or not 1 <= step <= total for step in values):
        raise ValueError(f'Image steps must be in [1, {total}]')
    return sorted(set(values) | {total})


def asset_spec(value, parent):
    if isinstance(value, str):
        path = Path(value)
        return dict(kind='file', path=str(path if path.is_absolute() else parent / path))
    if not isinstance(value, dict):
        raise ValueError('Each source/mask must be a path or a benchmark image specification')
    return value


def write_json(path, payload):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def file_identity(path):
    stat = path.stat()
    identity = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size <= 8 * 1024 * 1024:
        identity['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def input_identities(rows, parent):
    """Fingerprint actual files used by the benchmark image reader, including archives."""
    paths = set()
    for row in rows:
        for field in ('source', 'mask'):
            spec = asset_spec(row[field], parent)
            if spec['kind'] == 'file':
                paths.add(Path(spec['path']).resolve())
            elif spec['kind'] == 'tar':
                extracted = Path(spec['root']) / spec['member']
                paths.add((extracted if extracted.is_file() else Path(spec['root']) / spec['archive']).resolve())
            elif spec['kind'] == 'parquet':
                paths.add(Path(spec['file']).resolve())
            elif spec['kind'] != 'pie_rle':
                raise ValueError(f'Unsupported input kind: {spec["kind"]}')
    return [file_identity(path) for path in sorted(paths)]


def verify_completed_samples(root, samples):
    for sample in samples:
        for name, expected in sample['artifact_sha256'].items():
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Saved trajectory/frame changed or disappeared: {path}')


class DecodingTrace:
    """Observe one native scheduler call per step, without altering inputs or RNG."""

    def __init__(self, mask_token_id, chunk_size=256):
        self.mask_token_id, self.chunk_size = mask_token_id, chunk_size
        self.entropy, self.argmax, self.unresolved, self.predictions = [], [], [], []
        self.states = []

    def observe_step(self, original_step, *args, **kwargs):
        import numpy as np
        import torch

        arguments = inspect.signature(original_step).bind(*args, **kwargs).arguments
        logits, incoming = arguments['model_output'], arguments['sample']
        if arguments.get('edit_mask') is not None or arguments.get('source_tokens') is not None:
            raise ValueError('Motivation evidence requires dense decoding without source-token locking')
        if arguments.get('local_guidance', 0) != 0:
            raise ValueError('Local guidance changes the dense reconstruction protocol')
        if logits.ndim != 4 or logits.shape[0] != 1 or incoming.ndim != 3:
            raise ValueError('Trace expects one image and [1,V,H,W] model logits')
        if not self.entropy and not incoming.eq(self.mask_token_id).all():
            raise ValueError('Every target token must start masked; the annotation is for analysis only')
        h, w = incoming.shape[-2:]
        # Chunk over spatial tokens, retaining exact float32 entropy without a
        # second full-grid float32 vocabulary probability allocation.
        flat = logits.detach().flatten(2)[0]
        chunks = []
        for start in range(0, h * w, self.chunk_size):
            log_p = flat[:, start:start + self.chunk_size].T.float().log_softmax(-1)
            chunks.append((-(log_p.exp() * log_p).sum(-1)).cpu().numpy())
        entropy = np.concatenate(chunks).reshape(h, w)
        argmax = logits.detach().argmax(1)[0].cpu().numpy().astype(np.int32)
        # Exactly one original call. Capturing logits above consumes no RNG.
        result = original_step(*args, **kwargs)
        prediction = result.pred_original_sample
        state = result.prev_sample
        self.entropy.append(entropy)
        self.argmax.append(argmax)
        self.unresolved.append(state[0].eq(self.mask_token_id).cpu().numpy())
        self.predictions.append(prediction.detach().cpu().clone())
        self.states.append(state.detach().cpu().clone())
        return result

    @contextmanager
    def capture(self, scheduler):
        original_step = scheduler.step
        had_override = 'step' in scheduler.__dict__
        original_override = scheduler.__dict__.get('step')
        scheduler.step = lambda *a, **kw: self.observe_step(original_step, *a, **kw)
        try:
            yield self
        finally:
            if had_override:
                scheduler.step = original_override
            else:
                del scheduler.step

    def arrays(self, token_mask, expected_steps):
        import numpy as np

        if len(self.entropy) != expected_steps:
            raise ValueError(f'Expected {expected_steps} forwards, recorded {len(self.entropy)}')
        unresolved = np.stack(self.unresolved)
        if unresolved[-1].any() or (unresolved[1:] & ~unresolved[:-1]).any():
            raise ValueError('Expected a complete, monotonic native MaskGIT trajectory')
        commitment = np.argmax(~unresolved, axis=0).astype(np.int32) + 1
        return dict(entropy=np.stack(self.entropy), argmax=np.stack(self.argmax),
                    unresolved=unresolved, token_mask=token_mask, commitment_step=commitment,
                    sampler_state=np.stack([state[0].numpy() for state in self.states]),
                    sampled_prediction=np.stack([p[0].numpy() for p in self.predictions]))


def collect(args):
    import numpy as np
    import torch
    from PIL import Image

    rows, parent, selection = input_rows(args)
    if args.visual_sample and args.visual_sample not in {row['key'] for row in rows}:
        raise ValueError('--visual-sample must be among the selected --sample-keys/hash-ranked examples')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Collect real trajectories on a GPU host; '
                           '--stage render can later arrange those saved results on CPU.')
    from src.benchmark_data import BenchmarkImages
    from src.dataset_utils import safe_exif_transpose
    from src.mask_utils import normalize_edit_mask_image
    from src.editmgt import init_edit_mgt
    from scripts.motivation_layout import apply_unresolved_gray

    steps = display_steps(args.num_inference_steps, args.display_steps, args.snapshot_every)
    model_path = args.model_path.resolve()
    config_path = model_path / 'editmgt/config.json'
    model_config = json.loads(config_path.read_text())
    for flag in ('sparse_target', 'token_type_rope', 'cache_conditioning'):
        if model_config.get(flag, False):
            raise ValueError(f'Use the original dense model for this experiment: {flag}=true')
    if model_config.get('time_conditioning', 'legacy') != 'legacy':
        raise ValueError('Use the released dense model with its original time conditioning')
    weights = sorted((model_path / 'editmgt').glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError(f'Missing released dense Transformer weights under {model_path / "editmgt"}')
    component_files = sorted(path for name in ('editmgt', 'text_encoder', 'tokenizer', 'llm_encoder', 'vqvae', 'scheduler')
                             for path in (model_path / name).rglob('*')
                             if path.is_file() and path.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt'))
    protocol = dict(model_label='Dense EditMGT', model_protocol='native_editmgt_mask_free_no_cache',
                    dataset_label=(f'Inter-Edit test/{args.language}'
                                   if selection.get('benchmark') == 'Inter-Edit' else 'User-supplied examples'),
                    model_path=str(model_path), model_files=list(map(file_identity, component_files)),
                    torch_version=str(torch.__version__), python_version=sys.version,
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                    resolution=args.resolution, num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale, reference_strength=args.reference_strength,
                    negative_prompt=args.negative_prompt, dtype='bfloat16' if args.bf16 else 'float32',
                    seed=args.seed, mask_role='analysis_only', source_token_lock=False,
                    sparse_target=False, reference_kv_cache=False, condition_kv_cache=False,
                    output_compositing='none', token_mask_projection='nearest_neighbor',
                    entropy='float32 post-CFG full-grid logits; natural logarithm; all positions',
                    stability='raw per-step argmax disagreement with final-forward raw argmax',
                    frame_type='scheduler prev_sample; uncommitted mask-token cells shown in solid gray',
                    snapshot_interval=args.snapshot_every, unresolved_gray_rgb=[166, 166, 166],
                    masked_token_decode_fill=0,
                    image_preprocessing='source and annotation square-resized together before native preprocessing',
                    entropy_chunk_size=args.entropy_chunk_size, captured_image_steps=steps,
                    collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    implementation_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in sorted((ROOT / 'src').rglob('*.py'))})
    selection['selected_keys'] = [row['key'] for row in rows]
    identity = dict(protocol=protocol, selection=selection, inputs=rows,
                    input_files=input_identities(rows, parent))
    out = args.output_dir.resolve()
    metadata_path = out / 'metadata.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if not args.resume or metadata['identity'] != identity:
            raise ValueError('Output already exists: use --resume with the identical protocol/selection or a new --output-dir')
        verify_completed_samples(out, metadata['samples'])
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'Output directory is nonempty without metadata: {out}')
        out.mkdir(parents=True, exist_ok=True)
        metadata = dict(version=1, identity=identity, protocol=protocol, selection=selection,
                        created_at=datetime.now(timezone.utc).isoformat(), samples=[], skipped=[])
        write_json(metadata_path, metadata)
    completed = {row['key'] for row in metadata['samples']} | {row['key'] for row in metadata['skipped']}
    if all(row['key'] in completed for row in rows):
        return metadata
    pipe = init_edit_mgt(str(device), enable_bf16=args.bf16, base_model_path=str(model_path),
                         condition_cache_threshold=None)
    for module in (pipe.transformer, pipe.vqvae, pipe.text_encoder, pipe.text_encoder_t5):
        if module is not None:
            module.eval().requires_grad_(False)
    pipe.local_guidance = 0.0
    pipe.set_progress_bar_config(disable=True)
    torch.manual_seed(args.seed)
    # Keep one bounded asset reader open for the complete experiment.
    with closing(BenchmarkImages(args.tar_cache_dir)) as images, torch.inference_mode():
        for row in rows:
            if row['key'] in completed:
                continue
            source = safe_exif_transpose(images.image(asset_spec(row['source'], parent))).convert('RGB')
            mask = normalize_edit_mask_image(images.image(asset_spec(row['mask'], parent)))
            source = source.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
            mask = mask.resize(source.size, Image.Resampling.NEAREST)
            side = args.resolution // pipe.vae_scale_factor
            token_mask = np.asarray(mask.resize((side, side), Image.Resampling.NEAREST)) > 0
            if not token_mask.any() or token_mask.all():
                metadata['skipped'].append(dict(key=row['key'], reason='empty_edit_or_context_region_on_token_grid'))
                write_json(metadata_path, metadata)
                continue
            # A hash suffix makes arbitrary external keys safe and collision-resistant.
            name = ''.join(c if c.isalnum() or c in '_.-' else '_' for c in row['key']).strip('.')[:80]
            name = (name or 'sample') + '-' + hashlib.sha256(row['key'].encode()).hexdigest()[:8]
            relative = Path('samples') / name
            folder = out / relative
            folder.mkdir(parents=True, exist_ok=True)
            source.save(folder / 'source.png')
            mask.save(folder / 'mask.png')
            (folder / 'instruction.txt').write_text(row['instruction'] + '\n', encoding='utf-8')
            seed = (args.seed + int.from_bytes(hashlib.sha256(
                row.get('seed_key', row['key']).encode()).digest()[:8], 'little')) % (2**63 - 1)
            trace = DecodingTrace(pipe.scheduler.config.mask_token_id, args.entropy_chunk_size)
            print(f"Capturing {row['key']}: {row['instruction']}", flush=True)
            with trace.capture(pipe.scheduler):
                output = pipe(prompt=[row['instruction']], reference_image=[source], mask_image=None,
                     height=args.resolution, width=args.resolution,
                     num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                     reference_strength=args.reference_strength, negative_prompt=args.negative_prompt,
                     generator=torch.Generator(device=device).manual_seed(seed),
                     edit_only_diffusion=False, sparse_target=False, lora_part_enable=False,
                     preserve_unmasked_pixels=False, use_reference_kv_cache=False,
                     use_condition_kv_cache=False, output_type='latent')
            if not torch.equal(output.images.detach().cpu(), trace.states[-1]):
                raise AssertionError('Recorded final sampler state differs from the actual final pipeline tokens')
            arrays = trace.arrays(token_mask, args.num_inference_steps)
            np.savez_compressed(folder / 'trajectory.npz', **arrays)
            frames = {}
            for step in steps:
                # Decode only the state committed after this step. The VQ helper
                # substitutes code 0 for mask IDs; gray blocks then hide those
                # cells. No tentative future prediction or final-image pixel is
                # used to fill unresolved token positions.
                image = pipe.safe_decode_latents(trace.states[step - 1].to(device), 1,
                                                args.resolution, args.resolution, 'pil')[0]
                image = apply_unresolved_gray(image, arrays['unresolved'][step - 1])
                frame = relative / f'step_{step:03d}.png'
                image.save(out / frame)
                frames[str(step)] = frame.as_posix()
            artifacts = [relative / 'source.png', relative / 'mask.png', relative / 'trajectory.npz',
                         relative / 'instruction.txt', *[Path(value) for value in frames.values()]]
            hashes = {path.as_posix(): hashlib.sha256((out / path).read_bytes()).hexdigest() for path in artifacts}
            metadata['samples'].append(dict(key=row['key'], instruction=row['instruction'], seed=seed,
                directory=relative.as_posix(), source=(relative / 'source.png').as_posix(),
                mask=(relative / 'mask.png').as_posix(), trajectory=(relative / 'trajectory.npz').as_posix(),
                frames=frames, edit_fraction=float(token_mask.mean()), edit_type=row.get('edit_type'), artifact_sha256=hashes,
                inputs=dict(source=row['source'], mask=row['mask'])))
            # Choose a legible visual task using only annotation type/area, not
            # generated quality. Every selected example still enters the curves.
            eligible = [sample for sample in metadata['samples'] if sample.get('edit_type') == 'Local'
                        and 0.03 <= sample['edit_fraction'] <= 0.35]
            metadata['default_visual_sample'] = (eligible or metadata['samples'])[0]['key']
            metadata['visual_selection'] = 'first Local task with mask area in [3%,35%], else first example; independent of model outputs'
            write_json(metadata_path, metadata)
    if not metadata['samples']:
        raise ValueError('No examples with nonempty edit and context regions were captured')
    return metadata


def main(argv=None):
    args = parse_args(argv)
    if args.stage != 'render':
        collect(args)
    if args.stage != 'collect':
        from scripts.motivation_layout import render_figure

        render_figure(args.output_dir.resolve(), args.output_dir.resolve() / 'motivation',
                      sample_key=args.visual_sample, selected_steps=args.display_steps, dpi=args.dpi)
        print(f'Saved {args.output_dir / "motivation.pdf"}', flush=True)


if __name__ == '__main__':
    main()
