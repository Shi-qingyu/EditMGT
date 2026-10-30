"""Region-DiMO backend for unchanged EditMGT MagicBrush/Inter-Edit benchmarks.

The adapter binds only the runner/config factory in this command's process.
Dataset selection, multi-turn dependencies, masks, ordering, GPU partition,
metric suites, error handling and summaries remain the upstream implementation.
"""
import argparse
from contextlib import contextmanager
import gc,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch
import torch.multiprocessing as mp

from src.dimo.infer import OneStepEditor,load_role,load_components,resolve_halo_profile
from src.dimo.origin import sha256,weight_files


def protocol_config(args,benchmark=None):
    config=_ORIGINAL_GENERATION_CONFIG(args,benchmark)
    if args.dimo_backend in ('student','ema'):
        checkpoint=Path(args.checkpoint)
        identity_file=checkpoint/('COMPLETE' if (checkpoint/'COMPLETE').is_file() else 'export.json')
        config.update(model_protocol='region_dimo_independent_categorical_one_step',
            dimo_model=args.dimo_backend,r_init=args.dimo_r_init,noise_emb_perturb=0.0,
            guidance_scale=1.0,num_inference_steps=1,disable_reference_kv_cache=True,
            disable_condition_kv_cache=True,cache_protocol='disabled',
            checkpoint_manifest_sha256=sha256(identity_file))
        identity=json.loads(identity_file.read_text())
        profile_path=checkpoint/'config.json' if (checkpoint/'export.json').is_file() else checkpoint.parent/'teacher/config.json'
        profile=json.loads(profile_path.read_text())
        config.update(resolve_halo_profile(profile,args.sparse_target_halo))
        config.update(cache_conditioning=profile.get('cache_conditioning'),
                      condition_cache_threshold=profile.get('condition_cache_threshold'),
                      reference_kv_cache_threshold=None)
        config['training_recipe']=identity.get('config')
        config['source_metadata']=identity.get('metadata',identity.get('source_metadata'))
    else:
        config.update(model_protocol='teacher_iterative_benchmark',dimo_model='teacher')
        folder=Path(args.checkpoint) if args.checkpoint else Path(args.model_path)/'editmgt'
        if (folder/'transformer/config.json').is_file():folder=folder/'transformer'
        if (folder/'config.json').is_file():
            config['teacher_checksums']={p.name:sha256(p) for p in [folder/'config.json',*weight_files(folder)]}
            config['teacher_source_profile']=json.loads((folder/'config.json').read_text())
    return config


class StudentRunner:
    def __init__(self,args):
        self.args,self.warmed_up=args,False
        model,origin=load_role(args.checkpoint,args.dimo_backend)
        precision='bf16' if args.bf16 else 'no'
        device=torch.device(args.device)
        components=load_components(args.model_path,device,precision)
        self.editor=OneStepEditor(model,*components,device=device,precision=precision,
            resolution=args.resolution,halo=args.sparse_target_halo,r_init=args.dimo_r_init,
            lora_part_enable=origin['config'].get('lora_part_enable',True))
        self.metadata=dict(backend=args.dimo_backend,model_origin=origin,torch_version=str(torch.__version__),
            device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            precision=precision,warmup_generations_per_process=args.warmup,
            model_protocol='region_dimo_independent_categorical_one_step',cfg=1,cache='disabled',
            cache_conditioning=model.config.cache_conditioning,condition_cache_threshold=model.config.condition_cache_threshold,
            **self.editor.halo_profile,
            benchmark_time_scope='synchronized encoders + source VQ + one categorical Student forward + VQ decode; image resize included')

    def generate(self,source,mask,instruction,seed):
        device=self.editor.device
        def forward():return self.editor.edit(source,mask,instruction,seed=seed,
                         preserve_unmasked_pixels=self.args.preserve_unmasked_pixels)
        if not self.warmed_up:
            for _ in range(self.args.warmup):forward()
            self.warmed_up=True
        if device.type=='cuda':
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        image,info=forward()
        if device.type=='cuda':torch.cuda.synchronize(device)
        info.update(inference_seconds=time.perf_counter()-start,execution=self.metadata,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        return image,info

    def close(self):
        del self.editor
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


# Capture upstream factory once, outside the temporary command-local bindings.
from src.benchmark_utils import generation_config as _ORIGINAL_GENERATION_CONFIG


@contextmanager
def backend_context(args):
    import src.benchmark_utils as bu
    import src.benchmark_parallel as bp
    with patch.object(bu,'generation_config',protocol_config),patch.object(bp,'_spawn_workers',spawn_workers):
        if args.dimo_backend=='teacher':yield
        else:
            with patch.object(bu,'InferenceRunner',StudentRunner):yield


def benchmark_worker(rank,args,*worker_args):
    from src.benchmark_parallel import _benchmark_worker
    with backend_context(args):_benchmark_worker(rank,args,*worker_args)


def spawn_workers(args,run,shards,names,worker_dirs,devices,metric_devices):
    # Same spawn lifecycle as upstream; the small trampoline installs the
    # backend in each fresh interpreter, including GPU metrics-only workers.
    context=mp.spawn(benchmark_worker,args=(args,run.root,run.info,shards,names,worker_dirs,devices,metric_devices),
                     nprocs=len(shards),join=False)
    try:
        while not context.join():pass
    except BaseException:
        for process in context.processes:
            if process.is_alive():process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():process.kill();process.join()
        raise


def parse_args(argv=None):
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--benchmark',choices=['magicbrush','inter_edit'],required=True)
    p.add_argument('--backend',choices=['student','ema','teacher'],default='student')
    p.add_argument('--r-init',type=float,default=1.0)
    p.add_argument('--sparse-target-halo',type=int,default=None)
    own,remaining=p.parse_known_args(argv)
    if own.sparse_target_halo is not None:
        if own.sparse_target_halo<0:p.error('--sparse-target-halo must be nonnegative')
        remaining.extend(['--sparse-target-halo',str(own.sparse_target_halo)])
    if own.benchmark=='inter_edit':from eval import benchmark_inter_edit as entry
    else:from eval import benchmark_magicbrush as entry
    before=sys.argv
    try:
        sys.argv=[before[0],*remaining]
        args=entry.parse_args()
    finally:sys.argv=before
    args.dimo_backend,args.dimo_r_init=own.backend,own.r_init
    if own.backend!='teacher':
        # Upstream defaults to zero; DiMO must distinguish omission from CLI 0.
        args.sparse_target_halo=own.sparse_target_halo
        if (args.guidance_scale!=1 or args.num_inference_steps!=1 or args.negative_prompt!=''
                or args.reference_strength!=1 or args.mask_mode!='provided' or args.vanilla or args.cache_first_step):
            p.error('Student/EMA require --guidance-scale 1 --num-inference-steps 1, empty negative prompt, provided mask, reference strength 1, full DiMO checkpoint; cache-first-step unsupported')
    return args,entry


def main():
    args,entry=parse_args(sys.argv[1:])
    with backend_context(args):return entry.main(args)


if __name__=='__main__':raise SystemExit(main())
