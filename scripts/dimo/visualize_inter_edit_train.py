"""Reproducible, separate-file Inter-Edit train examples for paper inspection."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import time

from PIL import Image
import torch

from src.benchmark_utils import InferenceRunner, sample_seed, safe_name
from src.dimo.benchmark import parse_args as benchmark_args
from src.dimo.infer import OneStepEditor, load_components, load_role
from src.inter_edit_dataset import (InterEditReader, TarAssetReader,
                                    is_inter_edit_text_editing,
                                    matches_inter_edit_language)
from src.mask_utils import normalize_edit_mask_image


TYPES = ('Add', 'Local', 'Remove', 'Texture')
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'outputs/dimo_step4000_sft8000_comparison/inter_edit_train_visuals'


def manifest_path(output):
    return output / 'manifest.json'


def rows_from_manifest(output):
    return json.loads(manifest_path(output).read_text(encoding='utf-8'))['samples']


def prepare(args):
    if manifest_path(args.output_dir).exists():
        raise FileExistsError(f'Visualization selection already exists: {manifest_path(args.output_dir)}')
    reader = InterEditReader(args.data_dir, 'train', args.cache_dir,
                             translations_dir=args.translations_dir)
    heaps = {name: [] for name in TYPES}
    candidate_count = args.per_type * 3
    for shard in reader.metadata_files:
        for row in reader.read_metadata(shard):
            category = row.get('edit_type')
            if (category not in heaps or not matches_inter_edit_language(row, 'en')
                    or is_inter_edit_text_editing(row)):
                continue
            identifier = str(row['sample_id'])
            score = int.from_bytes(hashlib.sha256(f'{args.seed}:{identifier}'.encode()).digest()[:8], 'big')
            candidate = (-score, identifier, str(shard), row)
            heap = heaps[category]
            if len(heap) < candidate_count:
                heapq.heappush(heap, candidate)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, candidate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    with TarAssetReader(reader.root, reader.cache_dir) as assets:
        for category in TYPES:
            accepted = 0
            for _, identifier, shard, row in sorted(heaps[category], key=lambda item: -item[0]):
                raw = reader.read_sample(row, assets)
                mask = normalize_edit_mask_image(raw['edit_mask'])
                if mask.getbbox() is None:
                    continue
                name = f'{category.lower()}_{safe_name(identifier)}'
                folder = args.output_dir / name
                if folder.exists():
                    raise FileExistsError(folder)
                folder.mkdir()
                raw['input'].convert('RGB').save(folder / 'source.png')
                raw['output'].convert('RGB').save(folder / 'target.png')
                mask.save(folder / 'mask.png')
                samples.append(dict(name=name, sample_id=identifier, edit_type=category,
                                    instruction=row['instruction'], metadata_shard=Path(shard).name,
                                    source=f'{name}/source.png', target=f'{name}/target.png',
                                    mask=f'{name}/mask.png',
                                    seed=sample_seed(args.seed, f'inter_edit_train:{identifier}')))
                accepted += 1
                if accepted == args.per_type:
                    break
            if accepted != args.per_type:
                raise RuntimeError(f'Only {accepted}/{args.per_type} valid {category} samples')
    payload=dict(split='train', selection='lowest SHA256(seed:sample_id) in each edit_type',
                 seed=args.seed, per_type=args.per_type, language='en',
                 exclude_text_editing=True, translated_prompts=True,
                 source_root=str(reader.root), samples=samples)
    manifest_path(args.output_dir).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Saved {len(samples)} independent source/mask/target examples to {args.output_dir}',flush=True)


def run_sft(args):
    benchmark,_=benchmark_args(['--benchmark','inter_edit','--backend','teacher',
        '--checkpoint',str(args.sft_checkpoint),'--model-path',str(args.base_model),
        '--guidance-scale','6','--num-inference-steps','18','--bf16',
        '--output-dir',str(args.output_dir/'_benchmark_unused'),'--stage','generate'])
    runner=InferenceRunner(benchmark)
    try:
        for index,row in enumerate(rows_from_manifest(args.output_dir),1):
            output=args.output_dir/row['name']/'sft18.png'
            if output.is_file():continue
            with Image.open(args.output_dir/row['source']) as source, Image.open(args.output_dir/row['mask']) as mask:
                image,info=runner.generate(source,mask,row['instruction'],row['seed'])
            image.save(output)
            output.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
            print(f'SFT {index}: {row["name"]}',flush=True)
    finally:
        runner.close()


def run_dimo(args):
    model,origin=load_role(args.dimo_checkpoint,'ema')
    components=load_components(args.base_model,torch.device(args.device),'bf16')
    editor=OneStepEditor(model,*components,device=args.device,precision='bf16',
                         resolution=1024,halo=None,r_init=1,
                         lora_part_enable=origin['config'].get('lora_part_enable',True))
    for index,row in enumerate(rows_from_manifest(args.output_dir),1):
        with Image.open(args.output_dir/row['source']) as source, Image.open(args.output_dir/row['mask']) as mask:
            for steps in (1,2,4):
                output=args.output_dir/row['name']/f'ema{steps}.png'
                if output.is_file():continue
                editor.num_inference_steps=steps
                editor.context_cache=steps>1
                if torch.cuda.is_available():torch.cuda.synchronize(editor.device)
                start=time.perf_counter()
                image,info=editor.edit(source,mask,row['instruction'],seed=row['seed'])
                if torch.cuda.is_available():torch.cuda.synchronize(editor.device)
                info['inference_seconds']=time.perf_counter()-start
                image.save(output)
                output.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
        print(f'DiMO {index}: {row["name"]}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','sft','dimo'])
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--data-dir',type=Path,default=Path('/mnt/geminihzceph/public/data/Inter-Edit'))
    parser.add_argument('--translations-dir',type=Path,default=ROOT/'data/inter_edit_zh_to_en_qwen35_9b')
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'data/inter_edit_tar_index')
    parser.add_argument('--sft-checkpoint',type=Path,default=ROOT/'runs/editmgt_sparse_inter_edit_en_cache_magicbrush_no_dilation_maskfix/checkpoint-8000')
    parser.add_argument('--dimo-checkpoint',type=Path,default=ROOT/'runs/region_dimo_v3_inter_brush_maskedit_bs256/export-ema-step4000')
    parser.add_argument('--base-model',type=Path,default=ROOT/'checkpoint/EditMGT')
    parser.add_argument('--per-type',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.per_type<1:parser.error('--per-type must be positive')
    {'prepare':prepare,'sft':run_sft,'dimo':run_dimo}[args.stage](args)


if __name__=='__main__':main()
