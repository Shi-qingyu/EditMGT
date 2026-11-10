"""Export a large, resumable qualitative candidate pool from scored DiMO predictions."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

from src.inter_edit_dataset import is_inter_edit_text_editing
from scripts.dimo.export_qualitative_results import (
    AssetStore, DEFAULT_CACHE, DEFAULT_PREDICTIONS, DEFAULT_RECORDS,
    EDIT_TYPES, HAN, export, load_rows, quality_score, shortlist,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_RECORDS = ROOT / 'outputs/dimo_step4000_sft8000_comparison/sft18/inter_edit/metrics/per_sample.jsonl'
DEFAULT_SFT_PREDICTIONS = ROOT / 'outputs/dimo_step4000_sft8000_comparison/sft18/inter_edit'


def attach_sft(output_dir, samples, records_path, prediction_root):
    by_key = {row['key']: row for row in load_rows(records_path)}
    for number, sample in enumerate(samples, 1):
        key = sample['sample_key']
        row = by_key[key]
        if row['instruction'] != sample['instruction']:
            raise ValueError(f'SFT instruction differs for {key}')
        source = prediction_root / row['prediction']
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / sample['name'] / 'sft18.png'
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size:
                raise ValueError(f'Existing SFT result differs for {key}: {destination}')
        else:
            shutil.copy2(source, destination)
        sample['sft18'] = f"{sample['name']}/sft18.png"
        sample['sft18_model'] = 'EditMGT SFT-8000, 18 steps'
        if number % 50 == 0 or number == len(samples):
            print(f'Attached SFT 18-step results: {number}/{len(samples)}', flush=True)
    (output_dir / 'manifest.json').write_text(
        json.dumps({'samples': samples}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def gallery(output_dir, samples):
    def figure(label, relative):
        src = html.escape(relative, quote=True)
        return (f'<figure><a href="{src}"><img loading="lazy" src="{src}" '
                f'alt="{html.escape(label, quote=True)}"></a><figcaption>{html.escape(label)}</figcaption></figure>')

    cards = []
    for sample in samples:
        category = html.escape(sample['edit_type'], quote=True)
        images = ''.join(figure(label, sample[field]) for label, field in (
            ('Source', 'source'), ('Mask', 'mask'), ('SFT 18 steps', 'sft18'),
            ('DiMO 1 step', 'edited'), ('Target', 'target')))
        overlay = figure('Source + mask', sample['source_mask_overlay'])
        metrics = sample['metrics']
        cards.append(
            f'<section data-category="{category}"><h2>{html.escape(sample["sample_key"])} · {category}</h2>'
            f'<p>{html.escape(sample["instruction"])}</p>'
            f'<p>Mask {sample["mask_ratio"]:.1%} · LPIPS {metrics["S_global_LPIPS"]:.3f} · '
            f'AlphaCLIP-in {metrics["S_in_AlphaCLIP"]:.3f} · '
            f'AlphaCLIP-out {metrics["S_out_AlphaCLIP"]:.3f}</p>'
            f'<div class="images">{images}</div><details><summary>Show source-mask overlay</summary>'
            f'<div class="overlay">{overlay}</div></details></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiMO qualitative candidate pool</title>
<style>body{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;color:#202530}h1{font-size:28px}h2{font-size:19px;margin-bottom:4px}p{margin:6px 0}nav{position:sticky;top:0;background:#fff;padding:12px 0;border-bottom:1px solid #ddd;z-index:2}select{font:inherit;padding:5px}section{padding:20px 0;border-bottom:1px solid #ddd}.images{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:12px}figure{margin:0}img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#eee;border:1px solid #ddd}figcaption{text-align:center;margin-top:4px}.overlay{max-width:380px}summary{cursor:pointer;margin-top:10px}a{color:#1254a5}@media(max-width:800px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<h1>SFT-8000 18 steps vs DiMO EMA-4000 1 step · qualitative candidates</h1>
<p>Inter-Edit test/en. Candidates are ranked by DiMO LPIPS and region AlphaCLIP within each category. They are for visual review, not an unbiased sample of test performance. Source, mask, SFT output, DiMO output, and target are separate files.</p>
<p><a href="manifest.json">Manifest and metrics</a> · <a href="selection.json">Selection record</a></p>
<nav><label>Category <select id="category"><option value="all">All</option><option>Add</option><option>Local</option><option>Remove</option><option>Texture</option></select></label></nav>
''' + '\n'.join(cards) + '''<script>document.getElementById('category').addEventListener('change',e=>{for(const card of document.querySelectorAll('section[data-category]'))card.hidden=e.target.value!=='all'&&card.dataset.category!==e.target.value});</script></html>'''
    (output_dir / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--per-type', type=int, default=100)
    parser.add_argument('--records', type=Path, default=DEFAULT_RECORDS)
    parser.add_argument('--prediction-root', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--sft-records', type=Path, default=DEFAULT_SFT_RECORDS)
    parser.add_argument('--sft-prediction-root', type=Path, default=DEFAULT_SFT_PREDICTIONS)
    parser.add_argument('--attach-only', action='store_true',
                        help='Attach SFT results and rebuild the gallery for an existing complete export')
    parser.add_argument('--exclude-manifest', type=Path,
                        default=ROOT / 'docs/qualitative_results/manifest.json')
    parser.add_argument('--min-mask-ratio', type=float, default=0.01)
    parser.add_argument('--max-mask-ratio', type=float, default=0.35)
    args = parser.parse_args()
    if args.per_type < 1 or not 0 <= args.min_mask_ratio < args.max_mask_ratio < 1:
        parser.error('Invalid sample count or mask-ratio bounds')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.attach_only:
        prior = json.loads(args.exclude_manifest.read_text(encoding='utf-8'))
        excluded = {row['sample_key'] for row in prior['samples']}
        rows = [row for row in load_rows(args.records)
                if row.get('edit_type') in EDIT_TYPES and not HAN.search(row.get('instruction', ''))
                and not is_inter_edit_text_editing(row) and row['key'] not in excluded]
        with AssetStore(args.cache_dir) as assets:
            selected = shortlist(rows, assets, args.per_type, args.min_mask_ratio, args.max_mask_ratio)
            keys = [row['key'] for row in selected]
            selection = {
                'model': 'DiMO EMA-4000, 1 step', 'benchmark': 'Inter-Edit test/en',
                'strategy': 'lowest LPIPS + 2*(1-AlphaCLIP-in) + 2*(1-AlphaCLIP-out) in each edit type',
                'filters': {'text_editing': False, 'han_characters': False,
                            'mask_ratio': [args.min_mask_ratio, args.max_mask_ratio],
                            'excluded_prior_keys': sorted(excluded)},
                'per_type': args.per_type, 'keys': keys,
                'records': str(args.records.resolve()),
                'records_sha256': hashlib.sha256(args.records.read_bytes()).hexdigest(),
                'prediction_root': str(args.prediction_root.resolve()),
            }
            selection_path = output_dir / 'selection.json'
            content = json.dumps(selection, ensure_ascii=False, indent=2) + '\n'
            if selection_path.exists() and selection_path.read_text(encoding='utf-8') != content:
                raise ValueError(f'Selection differs from existing output: {selection_path}')
            selection_path.write_text(content, encoding='utf-8')
            export(selected, keys, assets, args.prediction_root, output_dir, resume=True)
    manifest = json.loads((output_dir / 'manifest.json').read_text(encoding='utf-8'))['samples']
    if len(manifest) != args.per_type * len(EDIT_TYPES):
        raise AssertionError('Incomplete qualitative export')
    attach_sft(output_dir, manifest, args.sft_records, args.sft_prediction_root)
    gallery(output_dir, manifest)
    (output_dir / 'README.md').write_text(
        f'# DiMO qualitative candidate pool\n\n'
        f'{len(manifest)} Inter-Edit test/en examples: {args.per_type} each from Add, Local, Remove, Texture. '
        f'Non-text instructions and {args.min_mask_ratio:.0%}–{args.max_mask_ratio:.0%} edit masks; '
        f'the prior curated keys were excluded. Selection uses per-sample quality metrics and is for '
        f'cherry picking, not average test performance.\n\n'
        f'Browse [all candidates](index.html), [manifest](manifest.json), and [selection record](selection.json). '
        f'Each folder stores separate `source.png`, `mask.png`, `source_mask_overlay.png`, '
        f'`sft18.png`, `edited.png` (DiMO 1 step), `target.png`, and `instruction.txt`.\n', encoding='utf-8')
    print(f'Saved {len(manifest)} qualitative candidates to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
