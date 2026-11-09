"""Summarize SFT/DiMO quality and vanilla EditMGT speed as Markdown."""
import argparse
import html
import json
from pathlib import Path
import numpy as np


METHODS = ('sft18', 'ema1', 'ema2', 'ema4')
BENCHMARKS = ('magicbrush', 'inter_edit', 'pie_bench')
LABELS = {'vanilla36': 'Vanilla EditMGT · 36 steps',
          'sft18': 'EditMGT SFT-8000 · 18 steps',
          'ema1': 'DiMO EMA-4000 · 1 step',
          'ema2': 'DiMO EMA-4000 · 2 steps + cache',
          'ema4': 'DiMO EMA-4000 · 4 steps + cache'}


def summary(root, method, benchmark):
    path = root / method / benchmark / 'metrics/summary.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if not data.get('complete') or data.get('failed_predictions') or data.get('unprocessed_predictions'):
        raise ValueError(f'Incomplete benchmark: {path}')
    return data


def table(rows):
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)


def numeric_table(data, benchmark, metrics, protocol=None):
    rows = [['Model', *[name for name, _, _ in metrics]],
            ['---', *['---:' for _ in metrics]]]
    for method in METHODS:
        item = data[(method, benchmark)]
        values = item['by_protocol'][protocol]['metrics'] if protocol else item['overall']
        rows.append([LABELS[method], *[format(values[key]['mean'], fmt) for _, key, fmt in metrics]])
    return table(rows)


def read_keys(path):
    with path.open(encoding='utf-8') as handle:
        return {(row['key'], tuple(row.get('groups', [])), row['seed_key'])
                for line in handle if line.strip() for row in [json.loads(line)]}


def mean(data,method,benchmark,metric,protocol=None):
    item=data[(method,benchmark)]
    values=item['by_protocol'][protocol]['metrics'] if protocol else item['overall']
    return values[metric]['mean']


def selection_keys(path):
    rows=json.loads(path.read_text(encoding='utf-8'))
    return {(row['key'],tuple(row.get('groups',[])),row['seed_key']) for row in rows}


def vanilla_timing(root,benchmark):
    """Time only the same successfully scored keys used by SFT and DiMO."""
    folder=root/'vanilla36'/benchmark
    generation=json.loads((folder/'generation_summary.json').read_text(encoding='utf-8'))
    run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    config=run['settings']['generation']
    if (not generation.get('complete') or generation.get('skipped')
            or generation.get('completed')!=generation.get('selected')):
        raise ValueError(f'Incomplete vanilla generation: {folder}')
    if (config.get('model_protocol')!='native_editmgt_mask_free_no_cache'
            or config.get('num_inference_steps')!=36 or config.get('mask_mode')!='none'
            or config.get('guidance_scale')!=6 or config.get('reference_strength')!=1.1
            or config.get('bf16') is not True or config.get('seed')!=42
            or config.get('warmup')!=1 or not config.get('disable_reference_kv_cache')
            or not config.get('disable_condition_kv_cache')):
        raise ValueError(f'Unexpected vanilla protocol: {folder}')
    expected_selection=selection_keys(root/'sft18'/benchmark/'selection.json')
    actual_selection=selection_keys(folder/'selection.json')
    if actual_selection!=expected_selection or len(actual_selection)!=generation['selected']:
        raise ValueError(f'Vanilla selection differs from SFT: {benchmark}')
    scored={key for key,_,_ in read_keys(root/'sft18'/benchmark/'metrics/per_sample.jsonl')}
    records={}
    for path in (folder/'records').glob('*.json'):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('status')!='ok' or row['key'] in records:
            raise ValueError(f'Invalid vanilla record: {path}')
        records[row['key']]=row
    if len(records)!=generation['completed'] or not scored<=records.keys():
        raise ValueError(f'Vanilla predictions do not cover scored SFT keys: {benchmark}')
    seconds=np.asarray([records[key]['inference_seconds'] for key in sorted(scored)],dtype=float)
    if not np.all(np.isfinite(seconds)) or not np.all(seconds>0):
        raise ValueError(f'Invalid vanilla timings: {benchmark}')
    return {'count':len(seconds),'mean_seconds':float(seconds.mean()),
            'p50_seconds':float(np.percentile(seconds,50)),
            'p95_seconds':float(np.percentile(seconds,95)),
            'generation_count':generation['completed']}


def write_gallery(folder, samples):
    """Browse independent image files; never create a combined bitmap."""
    cards=[]
    for row in samples:
        figures=[]
        for name in ('source','mask','target','sft18','ema1','ema2','ema4'):
            path=f'{row["name"]}/{name}.png'
            figures.append(f'<figure><img loading="lazy" src="{html.escape(path)}" alt="{html.escape(name)}">'
                           f'<figcaption>{html.escape(name)}</figcaption></figure>')
        cards.append(f'<section data-type="{html.escape(row["edit_type"])}">'
                     f'<h2>{html.escape(row["sample_id"])} · {html.escape(row["edit_type"])}</h2>'
                     f'<p>{html.escape(row["instruction"])}</p><div class="images">{"".join(figures)}</div></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Inter-Edit train demos</title>
<style>body{font:15px system-ui;margin:24px;color:#222}section{border-top:1px solid #bbb;padding:20px 0}
.images{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:8px}
figure{margin:0}img{width:100%;aspect-ratio:1;object-fit:contain;background:#eee}
figcaption{text-align:center;font-weight:600}select{font:inherit;padding:6px}</style>
<h1>Inter-Edit train demos</h1><p>Each panel loads a separate PNG. These are training examples for demo selection.</p>
<label>Category <select id="category"><option value="all">All</option>
<option>Add</option><option>Local</option><option>Remove</option><option>Texture</option></select></label>
''' + '\n'.join(cards) + '''<script>
document.getElementById('category').onchange=e=>{for(const s of document.querySelectorAll('section'))
s.hidden=e.target.value!=='all'&&s.dataset.type!==e.target.value}</script></html>'''
    (folder/'index.html').write_text(page,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--include-magicbrush-test', action='store_true',
                        help='Require and summarize the official MagicBrush test split too.')
    args = parser.parse_args()
    root = args.output_root.resolve()
    benchmarks = (('magicbrush','magicbrush_test','inter_edit','pie_bench')
                  if args.include_magicbrush_test else BENCHMARKS)
    data = {(method, benchmark): summary(root, method, benchmark)
            for method in METHODS for benchmark in benchmarks}
    for benchmark, split in (('magicbrush','dev'),('magicbrush_test','test')):
        if benchmark not in benchmarks:
            continue
        for method in METHODS:
            dataset = data[(method,benchmark)]['dataset']
            if dataset.get('split') != split or dataset.get('mode') != 'both' or not dataset.get('full_requested_split'):
                raise ValueError(f'Wrong/incomplete MagicBrush split for {method}/{benchmark}: {dataset}')
    coverage = {}
    for benchmark in benchmarks:
        key_sets = [read_keys(root / method / benchmark / 'metrics/per_sample.jsonl') for method in METHODS]
        if any(keys != key_sets[0] for keys in key_sets):
            raise ValueError(f'Methods scored different {benchmark} samples')
        coverage[benchmark] = len(key_sets[0])
    visual = json.loads((root / 'inter_edit_train_visuals/manifest.json').read_text(encoding='utf-8'))
    for row in visual['samples']:
        folder = root / 'inter_edit_train_visuals' / row['name']
        for name in ('source.png','target.png','mask.png','sft18.png','ema1.png','ema2.png','ema4.png'):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
    write_gallery(root / 'inter_edit_train_visuals',visual['samples'])
    vanilla={benchmark:vanilla_timing(root,benchmark) for benchmark in benchmarks}
    base = {benchmark: vanilla[benchmark]['mean_seconds'] for benchmark in benchmarks}
    speed_rows = [['Benchmark', 'Model', 'Measured images', 'Mean s/image ↓', 'P50 s ↓', 'P95 s ↓', 'Speedup vs vanilla ↑'],
                  ['---','---','---:','---:','---:','---:','---:']]
    benchmark_labels = {'magicbrush': 'MagicBrush dev', 'magicbrush_test': 'MagicBrush test',
                        'inter_edit': 'Inter-Edit test/en', 'pie_bench': 'PIE-Bench v1'}
    for benchmark in benchmarks:
        for method in ('vanilla36',*METHODS):
            timing = vanilla[benchmark] if method=='vanilla36' else data[(method, benchmark)]['timing']
            speed_rows.append([benchmark_labels[benchmark], LABELS[method], timing['count'],
                               f"{timing['mean_seconds']:.3f}",
                               f"{timing['p50_seconds']:.3f}",
                               f"{timing['p95_seconds']:.3f}",
                               f"{base[benchmark]/timing['mean_seconds']:.2f}×"])
    test_lines = []
    if args.include_magicbrush_test:
        test_single = data[('sft18','magicbrush_test')]['by_protocol']['single_turn_all']['metrics']['clip-i']['count']
        test_multi = data[('sft18','magicbrush_test')]['by_protocol']['multi_turn_final']['metrics']['clip-i']['count']
        test_lines = [
            '### MagicBrush test：单轮（n=' + str(test_single) + '）',
            '',
            numeric_table(data,'magicbrush_test', [('CLIP-I ↑','clip-i','.4f'),('DINO ↑','dino','.4f'),
                ('L1 ↓','l1','.4f'),('L2/MSE ↓','l2','.4f')], 'single_turn_all'),
            '',
            '### MagicBrush test：多轮最终结果（n=' + str(test_multi) + ' sessions）',
            '',
            numeric_table(data,'magicbrush_test', [('CLIP-I ↑','clip-i','.4f'),('DINO ↑','dino','.4f'),
                ('L1 ↓','l1','.4f'),('L2/MSE ↓','l2','.4f')], 'multi_turn_final'),
            '',
        ]
    coverage_text = (f"MagicBrush dev {coverage['magicbrush']} 张生成图；"
                     + (f"MagicBrush test {coverage['magicbrush_test']} 张生成图；" if args.include_magicbrush_test else '')
                     + f"Inter-Edit {coverage['inter_edit']} 张计分图；PIE-Bench {coverage['pie_bench']} 张计分图。")
    speed_text = '、'.join(
        f"{benchmark_labels[b]} {base[b]/data[('ema1',b)]['timing']['mean_seconds']:.2f}×"
        for b in benchmarks)
    findings = [
        f"- DiMO 1 步相对 vanilla EditMGT 的平均逐图加速为：{speed_text}。这些是包含文本编码与 VQ 的模型管线延迟，并非仅 Transformer forward。",
        f"- MagicBrush dev 单轮 CLIP-I 从 SFT 的 {mean(data,'sft18','magicbrush','clip-i','single_turn_all'):.4f} 降至 DiMO 1 步的 {mean(data,'ema1','magicbrush','clip-i','single_turn_all'):.4f}；多轮最终 CLIP-I 从 {mean(data,'sft18','magicbrush','clip-i','multi_turn_final'):.4f} 降至 {mean(data,'ema1','magicbrush','clip-i','multi_turn_final'):.4f}。Inter-Edit 全图 LPIPS 从 {mean(data,'sft18','inter_edit','S_global_LPIPS'):.4f} 升至 {mean(data,'ema1','inter_edit','S_global_LPIPS'):.4f}，显示这两个有 GT 的基准存在画质代价。",
        f"- PIE-Bench 的 DiMO 1 步目标文本 CLIP 为 {mean(data,'ema1','pie_bench','clip_similarity_target_image'):.4f}，SFT 为 {mean(data,'sft18','pie_bench','clip_similarity_target_image'):.4f}；未编辑区 PSNR 分别为 {mean(data,'ema1','pie_bench','psnr_unedit_part'):.2f} 与 {mean(data,'sft18','pie_bench','psnr_unedit_part'):.2f} dB。文本对齐与背景保留不能用单一指标概括。",
        f"- 当前 2/4 步近似缓存设置未在 MagicBrush dev 单轮 CLIP-I 上改善 1 步结果（1/2/4 步分别为 {mean(data,'ema1','magicbrush','clip-i','single_turn_all'):.4f}/{mean(data,'ema2','magicbrush','clip-i','single_turn_all'):.4f}/{mean(data,'ema4','magicbrush','clip-i','single_turn_all'):.4f}）；它们增加延迟，其他指标的变化见表格。",
    ]
    if args.include_magicbrush_test:
        findings.insert(2,
            f"- 官方 MagicBrush test 单轮 CLIP-I：SFT {mean(data,'sft18','magicbrush_test','clip-i','single_turn_all'):.4f}，DiMO 1/2/4 步 {mean(data,'ema1','magicbrush_test','clip-i','single_turn_all'):.4f}/{mean(data,'ema2','magicbrush_test','clip-i','single_turn_all'):.4f}/{mean(data,'ema4','magicbrush_test','clip-i','single_turn_all'):.4f}；多轮最终 CLIP-I：SFT {mean(data,'sft18','magicbrush_test','clip-i','multi_turn_final'):.4f}，DiMO 1 步 {mean(data,'ema1','magicbrush_test','clip-i','multi_turn_final'):.4f}。")
    lines = [
        '# SFT-8000 与 DiMO EMA-4000 质量、vanilla EditMGT 速度基线及训练集可视化',
        '',
        '## 实验设置',
        '',
        '- 基模型：`runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-8000`，18 步、CFG=6；采用 checkpoint 训练的低噪声条件缓存，阈值 0.8。',
        '- 蒸馏模型：`runs/region_dimo_v3_inter_brush_maskedit_bs256/checkpoint-4000` 的 EMA 导出；1 步使用独立 categorical 采样且无缓存，2/4 步使用 cosine MaskGIT 式迭代及 context K/V cache，CFG=1。',
        '- **速度基线**：发布版 `checkpoint/EditMGT/editmgt`（vanilla EditMGT），36 步、CFG=6、reference strength=1.1、发布版默认 negative prompt、mask-free、无 KV cache；仅用于速度比较，未将它混入 SFT/DiMO 的质量表。',
        '- 2/4 步在首个 forward 捕获 context，此时 mask ratio=1，高于训练的 0.8 缓存阶段；后续分别重用 1/3 次。这是**近似的提前缓存**，不能称为严格等价的低噪声缓存。每张图的实际缓存元数据见 `records/*.json`。',
        '- SFT 与 DiMO 质量评测采用相同的原始标注 mask、1024 分辨率、BF16、seed 42、8 张 H20、直接 VQ 解码且不做源图像素合成；vanilla 速度评测使用同样的硬件、分辨率、精度和 seed，但遵循其原生 mask-free 推理协议。',
        '- MagicBrush 使用 dev' + (' 和 test' if args.include_magicbrush_test else '') + ' 集的 single/multi both 模式；Inter-Edit 使用完整英文 test 子集；PIE-Bench 使用 v1 全集。基准计分样本在四种方法间逐条对齐。',
        *(['- MagicBrush test 分片来自 [osunlp/MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush/tree/6c254560879df7851233ea0fd2ae2a62764d0ef2/data)，revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`。'] if args.include_magicbrush_test else []),
        '',
        '## 质量结果',
        '',
        f"覆盖率：{coverage_text}所有结果 `complete=true`，失败生成数为零。箭头表示较优方向。",
        '',
        '### MagicBrush dev：单轮（n=528）',
        '',
        numeric_table(data,'magicbrush', [('CLIP-I ↑','clip-i','.4f'),('DINO ↑','dino','.4f'),
            ('L1 ↓','l1','.4f'),('L2/MSE ↓','l2','.4f')], 'single_turn_all'),
        '',
        '### MagicBrush dev：多轮最终结果（n=266 sessions）',
        '',
        numeric_table(data,'magicbrush', [('CLIP-I ↑','clip-i','.4f'),('DINO ↑','dino','.4f'),
            ('L1 ↓','l1','.4f'),('L2/MSE ↓','l2','.4f')], 'multi_turn_final'),
        '',
        *test_lines,
        '### Inter-Edit test/en',
        '',
        '两条空 mask 注释在各方法中一致跳过；计分数见覆盖率。',
        '',
        numeric_table(data,'inter_edit', [('LPIPS ↓','S_global_LPIPS','.4f'),
            ('AlphaCLIP-in ↑','S_in_AlphaCLIP','.4f'),('AlphaCLIP-out ↑','S_out_AlphaCLIP','.4f'),
            ('BSS ↓','BSS_Laplacian','.4f')]),
        '',
        '### PIE-Bench v1：编辑对齐及结构（n=700）',
        '',
        'PIE-Bench 没有编辑后 GT；Target CLIP 使用目标文本描述，Structure Distance 以源图为参考。',
        '',
        numeric_table(data,'pie_bench', [('Target CLIP ↑','clip_similarity_target_image','.4f'),
            ('Edit-part Target CLIP ↑','clip_similarity_target_image_edit_part','.4f'),
            ('Structure Distance ↓','structure_distance','.4f')]),
        '',
        '### PIE-Bench v1：未编辑区域保留',
        '',
        '下表每个指标的有效样本数请以原始 summary 的 `count/undefined_count` 为准；无未编辑像素的样本不会进入相应均值。',
        '',
        numeric_table(data,'pie_bench', [('PSNR ↑','psnr_unedit_part','.4f'),
            ('LPIPS ↓','lpips_unedit_part','.4f'),('MSE ↓','mse_unedit_part','.6f'),
            ('SSIM ↑','ssim_unedit_part','.4f')]),
        '',
        '## 速度对比',
        '',
        '每张图在 GPU 同步前后计时，包含图像转张量、文本编码、source VQ、模型 forward 和 VQ 解码；不含原生尺寸 resize 与 mask 准备、首次模型加载、预热、数据读取、PNG 写盘和指标计算。DiMO 记录的时间扣除了与 vanilla/SFT 计时器同样位于推理管线外的准备及后处理时间。各方法每个 GPU 预热 1 张图。下表按相同样本键统计：Inter-Edit 的 vanilla 生成了 3413 张，但其中两张在 mask-conditioned 质量评测中因空 mask 跳过，速度表只取共同的 3411 张。MagicBrush 每个 split 均包含单轮与连续多轮两种模式。速度倍数按同一 benchmark 的平均延迟相除，**以 vanilla EditMGT 为 1.00×**。',
        '',
        table(speed_rows),
        '',
        '## 供论文写作的观察',
        '',
        *findings,
        '',
        '## Inter-Edit 训练集可视化素材',
        '',
        f"按 `sample_id` 的 SHA256 稳定选取 Add/Local/Remove/Texture 各 {visual['per_type']} 条，共 {len(visual['samples'])} 条英文训练样本。每条的 `source.png`、`mask.png`、`target.png`、`sft18.png`、`ema1.png`、`ema2.png`、`ema4.png` **分别保存为独立文件**；可在 [浏览页](inter_edit_train_visuals/index.html)筛选和查看。对应 instruction、样本 ID、类别和 seed 见 [manifest](inter_edit_train_visuals/manifest.json)。这些是训练集样本，仅用于 demo/cherry pick，不作为泛化证据。",
        '',
        '## 解释与限制',
        '',
        '- DiMO 的 2/4 步是从一步蒸馏模型构造的推理扩展，首步 context cache 为高噪声近似快照；质量变化既包含步数，也包含该缓存策略。',
        '- vanilla、SFT 与 DiMO 的推理步数、mask 使用、缓存和提示词设置不同；速度表体现各自实际配置，不是隔离单一组件的因果消融。SFT 与 DiMO 的质量表也不能据此归因于单独一个训练或推理组件。',
        '- MagicBrush 只评估 L1/L2/CLIP-I/DINO，没有 CLIP-T；Inter-Edit 只报告英文子集。PIE 的未编辑区指标对空区域未定义。',
        '- MagicBrush/Inter-Edit 按原生尺寸计分；PIE 在 512×512 计分，文本 CLIP 指标为 `100 × max(cosine, 0)`，不能直接与 MagicBrush 的 0–1 CLIP-I 数值比较。',
        '- 速度是当前 8×H20、BF16、batch=1 的 per-image 延迟；跨硬件或 batch size 不应直接外推。',
        '',
        '## 原始结果',
        '',
        '| Model | ' + ' | '.join(benchmark_labels[benchmark] for benchmark in benchmarks) + ' |',
        '|' + '|'.join('---' for _ in range(len(benchmarks)+1)) + '|',
    ]
    for method in METHODS:
        links = [f'[summary]({method}/{benchmark}/metrics/summary.json)' for benchmark in benchmarks]
        lines.append('| ' + ' | '.join([LABELS[method], *links]) + ' |')
    lines += ['', 'vanilla 速度原始记录：' + '、'.join(
        f'[{benchmark_labels[benchmark]}](vanilla36/{benchmark}/generation_summary.json)'
        for benchmark in benchmarks) + '。',
        '', '训练集可视化索引：[manifest](inter_edit_train_visuals/manifest.json)。', '']
    destination = root / 'results.md'
    content = '\n'.join(lines)
    destination.write_text(content,encoding='utf-8')
    docs = root.parent.parent / 'docs' / 'dimo_step4000_sft8000_results.md'
    docs.parent.mkdir(parents=True,exist_ok=True)
    docs_content = content
    prefix = f'../outputs/{root.name}/'
    for name in ('vanilla36',*METHODS,'inter_edit_train_visuals'):
        docs_content = docs_content.replace(f']({name}/',f']({prefix}{name}/')
    docs.write_text(docs_content,encoding='utf-8')
    print(f'Saved {destination}',flush=True)
    print(f'Saved {docs}',flush=True)


if __name__=='__main__':main()
