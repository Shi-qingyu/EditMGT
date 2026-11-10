"""Summarize matched dense/sparse/cache/distillation evaluations and mask-area scaling.

This report reads measured benchmark artifacts; it never fabricates missing
scores or rescales measured latency. All four stages use the same successful
sample keys for every comparison, including comparisons within a mask bin.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics


STAGES = ('dense', 'sparse', 'sparse_cache', 'distill')
LABELS = {'dense': 'Dense', 'sparse': 'Sparse', 'sparse_cache': 'Sparse + cache', 'distill': 'Distilled'}
AREA_FIELDS = ('original_mask_fraction', 'generation_mask_fraction', 'active_target_fraction', 'vq_mask_fraction')
TIMING_SCOPE = ('GPU-synchronized pipeline including image-to-tensor conversion, text encoding, source VQ, '
                'model forwards and VQ decoding; excludes native-size resizing and mask preparation, '
                'model loading, warmup, data reads, PNG writes and metric computation.')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def keyed(rows, source):
    result = {}
    for row in rows:
        key = row.get('key')
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f'Missing or duplicate sample key in {source}: {key!r}')
        result[key] = row
    return result


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return keyed([json.loads(line) for line in handle if line.strip()], path)


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percent / 100
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def number(value, label):
    if isinstance(value, bool):
        raise ValueError(f'Boolean value is not a numeric score: {label}')
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f'Invalid numeric value for {label}: {value!r}') from error
    if math.isnan(result):
        raise ValueError(f'NaN value for {label}')
    return result


def metric_statistics(values, name):
    parsed = [number(value, name) for value in values if value is not None]
    finite = [value for value in parsed if math.isfinite(value)]
    positive = sum(value == math.inf for value in parsed)
    negative = sum(value == -math.inf for value in parsed)
    if positive and negative:
        mean, std = None, None
    elif positive or negative:
        mean, std = ('inf' if positive else '-inf'), None
    elif finite:
        mean, std = statistics.fmean(finite), statistics.pstdev(finite)
    else:
        mean, std = None, None
    return dict(mean=mean, std=std, count=len(parsed), finite_count=len(finite),
                undefined_count=len(values) - len(parsed), positive_infinity_count=positive,
                negative_infinity_count=negative,
                finite_mean=statistics.fmean(finite) if finite else None)


def ablation(record, run):
    settings = run.get('settings', {}).get('generation', {})
    data = dict(record.get('execution', {}).get('ablation', {}), **record.get('ablation', {}))
    data.setdefault('steps', settings.get('num_inference_steps'))
    data.setdefault('cfg', settings.get('guidance_scale'))
    data.setdefault('precision', 'bf16' if settings.get('bf16') else 'float32')
    data.setdefault('weight_source', settings.get('checkpoint'))
    return data


def stage_protocol(stage, records, run, keys):
    first = ablation(records[keys[0]], run)
    protocol_fields = ('steps', 'cfg', 'precision', 'weight_source')
    for key in keys:
        values = ablation(records[key], run)
        if values.get('stage', stage) != stage:
            raise ValueError(f'{stage}/{key}: recorded ablation stage does not match its directory')
        if any(values.get(name) != first.get(name) for name in protocol_fields):
            raise ValueError(f'{stage}: mixed inference settings across samples')
        if values.get('source_hard_lock') is not True:
            raise ValueError(f'{stage}/{key}: expected the same source-token preservation policy in every stage')
        for name in AREA_FIELDS:
            if name not in values:
                raise ValueError(f'{stage}/{key}: missing {name}; use benchmark_efficiency runner outputs')
            value = number(values[name], f'{stage}/{key}/{name}')
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{stage}/{key}: {name} must be a finite fraction in [0,1]')
        if stage == 'dense' and not math.isclose(values['active_target_fraction'], 1., abs_tol=1e-7):
            raise ValueError(f'{stage}/{key}: dense target compute must cover the full target grid')
    if first.get('steps') is None or first.get('cfg') is None or first.get('weight_source') is None:
        raise ValueError(f'{stage}: missing recorded steps, CFG or model identity')
    return {name: first[name] for name in protocol_fields}


def load_benchmark(root, benchmark):
    data = {}
    for stage in STAGES:
        folder = root / stage / benchmark
        selection = keyed(read_json(folder / 'selection.json'), folder / 'selection.json')
        records = keyed([read_json(path) for path in sorted((folder / 'records').glob('*.json'))], folder / 'records')
        scores = read_jsonl(folder / 'metrics/per_sample.jsonl')
        if set(scores) - selection.keys() or set(records) - selection.keys():
            raise ValueError(f'{stage}/{benchmark}: results contain keys outside the selected dataset')
        successful = {key for key, record in records.items() if record.get('status') == 'ok' and key in scores}
        data[stage] = dict(selection=selection, records=records, scores=scores, successful=successful,
                           run=read_json(folder / 'run.json'), summary=read_json(folder / 'metrics/summary.json'))
    reference = data['dense']['selection']
    for stage in STAGES[1:]:
        if data[stage]['selection'] != reference:
            raise ValueError(f'{benchmark}: {stage} selection/source/mask/instruction/seed keys differ from dense')
    matched = sorted(set.intersection(*(entry['successful'] for entry in data.values())))
    if not matched:
        raise ValueError(f'{benchmark}: no successfully scored sample keys are common to all four stages')
    protocols = {stage: stage_protocol(stage, entry['records'], entry['run'], matched) for stage, entry in data.items()}
    if any(protocols[stage] != protocols['dense'] for stage in ('sparse', 'sparse_cache')):
        raise ValueError(f'{benchmark}: dense, sparse and sparse+cache must share weights, steps, CFG and precision')
    if protocols['distill']['precision'] != protocols['dense']['precision']:
        raise ValueError(f'{benchmark}: distillation and Teacher stages must use the same precision')
    metric_names = set(data['dense']['scores'][matched[0]]['metrics'])
    if not metric_names:
        raise ValueError(f'{benchmark}: successful metric rows contain no quality scores')
    for key in matched:
        expected_seed = data['dense']['records'][key].get('seed')
        expected_area = ablation(data['dense']['records'][key], data['dense']['run'])
        for stage, entry in data.items():
            record, score = entry['records'][key], entry['scores'][key]
            if record.get('seed') != expected_seed or score.get('seed') != expected_seed:
                raise ValueError(f'{benchmark}/{key}: generation/metric seeds do not match across stages')
            if set(score['metrics']) != metric_names:
                raise ValueError(f'{stage}/{benchmark}/{key}: requested quality metric sets differ across the comparison')
            area = ablation(record, entry['run'])
            for field in ('original_mask_fraction', 'generation_mask_fraction', 'vq_mask_fraction'):
                if not math.isclose(area[field], expected_area[field], rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError(f'{benchmark}/{key}: {field} differs between stages')
            for field in ('original_mask_sha256', 'vq_mask_sha256', 'generation_mask_sha256'):
                if area.get(field) != expected_area.get(field):
                    raise ValueError(f'{benchmark}/{key}: actual mask contents differ between stages ({field})')
            seconds = number(record.get('inference_seconds'), f'{stage}/{benchmark}/{key}/latency')
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f'{stage}/{benchmark}/{key}: latency must be measured, finite and positive')
    coverage = {}
    for stage, entry in data.items():
        status = Counter(row.get('status', 'unknown') for row in entry['records'].values())
        coverage[stage] = dict(selected=len(reference), record_status_counts=dict(status),
                               missing_records=len(reference) - len(entry['records']),
                               scored=len(entry['scores']), successful_scored=len(entry['successful']),
                               matched=len(matched), unmatched_successful=len(entry['successful'] - set(matched)),
                               excluded_keys=sorted(set(reference) - set(matched)),
                               upstream_complete=entry['summary'].get('complete'),
                               failed_predictions=entry['summary'].get('failed_predictions'),
                               unprocessed_predictions=entry['summary'].get('unprocessed_predictions'))
    return data, matched, sorted(metric_names), protocols, coverage


def aggregate(benchmark, stage, keys, data, protocol, metric_names, label='all', lower=None, upper=None):
    rows = data[stage]
    times = [rows['records'][key]['inference_seconds'] for key in keys]
    dense_times = [data['dense']['records'][key]['inference_seconds'] for key in keys]
    result = dict(benchmark=benchmark, stage=stage, mask_area_bin=label, mask_area_lower=lower,
                  mask_area_upper=upper, count=len(keys), **protocol,
                  mean_seconds=statistics.fmean(times) if times else None,
                  p50_seconds=percentile(times, 50), p95_seconds=percentile(times, 95),
                  speedup_vs_dense=statistics.fmean(dense_times) / statistics.fmean(times) if times else None)
    for name in AREA_FIELDS:
        values = [ablation(rows['records'][key], rows['run'])[name] for key in keys]
        result['mean_' + name] = statistics.fmean(values) if values else None
    for name in ('active_target_tokens', 'full_target_tokens'):
        values = [ablation(rows['records'][key], rows['run']).get(name) for key in keys]
        result['mean_' + name] = statistics.fmean(values) if values and all(v is not None for v in values) else None
    result['metrics'] = {name: metric_statistics([rows['scores'][key]['metrics'][name] for key in keys], name)
                         for name in metric_names}
    return result


def area_edges(value):
    try:
        edges = [float(part.strip()) for part in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Mask bins must be comma-separated fractions, e.g. 0,.05,.1,.2,.4,.6,.8,1') from error
    if (len(edges) < 2 or edges[0] != 0 or edges[-1] != 1
            or any(not math.isfinite(v) for v in edges)
            or any(left >= right for left, right in zip(edges, edges[1:]))):
        raise argparse.ArgumentTypeError('Mask-area bins must increase strictly from 0 to 1')
    return edges


def flat_aggregate(row):
    flat = {name: value for name, value in row.items() if name != 'metrics'}
    for name, stats in row['metrics'].items():
        for field in ('mean', 'count', 'undefined_count', 'positive_infinity_count', 'negative_infinity_count'):
            flat[f'metric/{name}/{field}'] = stats[field]
    return flat


def write_csv(path, rows):
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formatted(value, digits=4):
    if value is None:
        return '—'
    if isinstance(value, str):
        return value
    return f'{value:.{digits}f}'


def table(rows, metric_names, bins=False):
    header = (['Mask area'] if bins else []) + ['Model', 'N', 'Steps', 'CFG', 'Mask %', 'Generation %',
               'Active target %', 'Mean s ↓', 'P50 s ↓', 'P95 s ↓', 'Speedup', *metric_names]
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |']
    for row in rows:
        cells = ([row['mask_area_bin']] if bins else []) + [LABELS[row['stage']], str(row['count']),
                 str(row['steps']), str(row['cfg']),
                 *[formatted(100 * row[name], 2) if row[name] is not None else '—'
                   for name in ('mean_original_mask_fraction', 'mean_generation_mask_fraction', 'mean_active_target_fraction')],
                 *[formatted(row[name]) for name in ('mean_seconds', 'p50_seconds', 'p95_seconds')],
                 formatted(row['speedup_vs_dense'], 2) + '×' if row['speedup_vs_dense'] is not None else '—',
                 *[formatted(row['metrics'][name]['mean']) for name in metric_names]]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--benchmarks', nargs='+', default=['inter_edit', 'pie_bench'],
                        choices=['inter_edit', 'pie_bench'])
    parser.add_argument('--mask-area-bins', type=area_edges, default=area_edges('0,.05,.1,.2,.4,.6,.8,1'))
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    report = dict(version=1, timing_scope=TIMING_SCOPE, mask_area_edges=args.mask_area_bins,
                  area_protocol='Original annotation area at native resolution; fixed natural-area bins, without mask dilation or dataset changes.',
                  comparison_protocol='Dense/sparse/sparse+cache share SFT weights, steps and CFG. Distillation changes weights and step count. The launcher defaults to CFG=1 for all four; if Teacher CFG is overridden, the reported CFG difference is an additional factor.',
                  metric_undefined_policy='No imputation; None and both-sign infinity means are blank, infinities remain strings, and valid/undefined/infinity counts are exported.',
                  benchmarks={})
    overall_csv, bins_csv, sample_csv = [], [], []
    markdown = ['# Dense → sparse → sparse + cache → distillation', '',
                report['comparison_protocol'], '', TIMING_SCOPE, '',
                'Mask-area tables group the original annotations; they do not enlarge masks or change editing targets. '
                'Mask % is annotation coverage, Generation % is the sampled token region after halo/projection, '
                'and Active target % is the target attention-token fraction actually processed. '
                'Every speedup uses the same matched keys in its corresponding area bin.', '',
                'Missing metric means are shown as —. Infinity and undefined counts are preserved in JSON/CSV. '
                'All exclusions and per-stage coverage are reported below; no samples are selected by quality.', '']
    for benchmark in args.benchmarks:
        data, keys, names, protocols, coverage = load_benchmark(root, benchmark)
        overall = [aggregate(benchmark, stage, keys, data, protocols[stage], names) for stage in STAGES]
        area_rows = []
        for index, (lower, upper) in enumerate(zip(args.mask_area_bins, args.mask_area_bins[1:])):
            last = index == len(args.mask_area_bins) - 2
            selected = [key for key in keys if lower <= ablation(data['dense']['records'][key], data['dense']['run'])['original_mask_fraction']
                        and (ablation(data['dense']['records'][key], data['dense']['run'])['original_mask_fraction'] <= upper if last
                             else ablation(data['dense']['records'][key], data['dense']['run'])['original_mask_fraction'] < upper)]
            label = f'[{100*lower:g}, {100*upper:g}' + (']%' if last else ')%')
            area_rows.extend(aggregate(benchmark, stage, selected, data, protocols[stage], names,
                                       label, lower, upper) for stage in STAGES)
        report['benchmarks'][benchmark] = dict(selected=len(data['dense']['selection']), matched=len(keys),
            excluded_keys=sorted(set(data['dense']['selection']) - set(keys)), metric_names=names,
            protocols=protocols, coverage=coverage, overall=overall, by_mask_area=area_rows)
        overall_csv.extend(map(flat_aggregate, overall))
        bins_csv.extend(map(flat_aggregate, area_rows))
        for key in keys:
            for stage in STAGES:
                entry = data[stage]
                record = entry['records'][key]
                row = dict(ablation(record, entry['run']))
                row.update(benchmark=benchmark, stage=stage, key=key, seed=record.get('seed'),
                           inference_seconds=record['inference_seconds'])
                row.update({f'metric/{name}': entry['scores'][key]['metrics'][name] for name in names})
                sample_csv.append(row)
        markdown.extend([f'## {benchmark}', '',
                         f"Selected: {len(data['dense']['selection'])}; matched successful scores across all stages: {len(keys)}; "
                         f"excluded: {len(data['dense']['selection']) - len(keys)}. Exact excluded keys are in summary.json.", '',
                         table(overall, names), '', '### Original mask-area scaling', '', table(area_rows, names, bins=True), '',
                         '### Coverage', '', '| Model | Selected | Scored | Matched | Missing records | Record statuses |',
                         '| --- | ---: | ---: | ---: | ---: | --- |'])
        for stage in STAGES:
            counts = coverage[stage]
            markdown.append(f"| {LABELS[stage]} | {counts['selected']} | {counts['scored']} | {counts['matched']} | "
                            f"{counts['missing_records']} | {json.dumps(counts['record_status_counts'], sort_keys=True)} |")
        markdown.append('')
    write_csv(root / 'overall.csv', overall_csv)
    write_csv(root / 'mask_area.csv', bins_csv)
    write_csv(root / 'per_sample.csv', sample_csv)
    (root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (root / 'summary.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
