"""Real tiny EditMGT training through the same update/checkpoint code as CUDA."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed

from src.dimo.checkpoint import (training_state, state_digest, save_checkpoint,
                                 load_checkpoint, check_roundtrip, rng_state, microbatches_per_update)
from src.dimo.core import clone_models, generation_mask
from src.dimo.optimization import FREEZE_DEFAULTS,configure_trainability,build_optimizers,build_schedulers
from src.dimo.data import ReplayStream
from src.dimo.train import run_update
from src.timestep_sampling import MaskRatioSampler
from src.transformer import Transformer2DModel


def tiny_model(**kwargs):
    return Transformer2DModel(num_layers=1, num_single_layers=1, attention_head_dim=12,
        num_attention_heads=2, joint_attention_dim=24, pooled_projection_dim=8,
        axes_dims_rope=(4, 4, 4), vocab_size=32, codebook_size=31,
        downsample=True, upsample=True, sparse_target=True, sparse_target_halo=0,
        time_conditioning='mask_ratio', cache_conditioning=True, **kwargs)


class Samples(torch.utils.data.IterableDataset):
    def __iter__(self):
        index = 0
        while True:
            yield dict(source=torch.randint(0, 31, (6, 8)),
                       augmentation=random.random() + np.random.rand(), index=index)
            index += 1


def smoke(out, steps=1, rank=0, workers=1, r_init=1.0, fake_rounds=1, device="cpu", precision="no"):
    torch.set_num_threads(1)
    handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    if workers > 1:
        handlers.append(InitProcessGroupKwargs(backend='nccl' if device=='cuda' else 'gloo', timeout=timedelta(seconds=180)))
    acc = Accelerator(cpu=device=='cpu',mixed_precision=precision,kwargs_handlers=handlers)
    set_seed(91)
    teacher, student, auxiliary, ema = clone_models(tiny_model())
    for model in (teacher,student,auxiliary,ema):model.enable_sparse_target(halo=1,supervise_halo=True)
    config=dict(gradient_accumulation=2,fake_rounds=fake_rounds,auxiliary_label_smoothing=0.1,
        student_initial_mask_ratio=r_init,ema_decay=0.9999,ema_warmup=True,mixed_precision=precision,smoke=True,
        teacher_cfg=6.0,noise_emb_perturb=0.3,student_lr=1e-4,auxiliary_lr=1e-4,
        adam_betas=[0.9,0.999],adam_epsilon=1e-8,weight_decay=0.01,
        lr_scheduler='constant_with_warmup',lr_warmup_steps=100,**{k:True for k in FREEZE_DEFAULTS})
    parameter_report=configure_trainability(dict(teacher=teacher,student=student,auxiliary=auxiliary,ema=ema),config)
    teacher.to(acc.device);ema.to(acc.device)
    student.enable_gradient_checkpointing()
    auxiliary.enable_gradient_checkpointing()
    opts=build_optimizers(dict(student=student,auxiliary=auxiliary),config)
    schedulers=build_schedulers(opts,config)
    opt_s,opt_a=opts['student'],opts['auxiliary']
    student, auxiliary, opt_s, opt_a = acc.prepare(student, auxiliary, opt_s, opt_a)
    models = dict(student=student, auxiliary=auxiliary, ema=ema)
    optimizers = dict(student=opt_s, auxiliary=opt_a)
    set_seed(92 + rank)
    original = torch.zeros(2, 6, 8, dtype=torch.bool,device=acc.device)
    original[0, 1, 3] = True
    original[1, 2:5, 2:6] = True
    rows, cols = torch.meshgrid(torch.arange(3,device=acc.device), torch.arange(4,device=acc.device), indexing='ij')
    ids = torch.stack((torch.zeros_like(rows), rows, cols), -1).reshape(-1, 3)
    hidden, pooled, micro = torch.randn(2,3,24,device=acc.device), torch.randn(2,8,device=acc.device), torch.randn(2,5,device=acc.device)
    def new_stream(consumed=0):
        return ReplayStream(torch.utils.data.DataLoader(Samples(), batch_size=2), 66 + rank, consumed)
    stream = new_stream()
    positions = []
    def batch_fn():
        source = next(stream)['source'].to(acc.device)
        positions.append(stream.consumed - 1)
        cond = dict(encoder_hidden_states=hidden, pooled_projections=pooled, micro_conds=micro,
                    img_ids=ids, reference_image_ids=ids, txt_ids=torch.zeros(3, 3,device=acc.device),
                    target_edit_mask=original, reference_image_hidden_states=source, lora_part_enable=True)
        empty = dict(cond, encoder_hidden_states=torch.zeros_like(hidden), pooled_projections=torch.zeros_like(pooled))
        return source, original, generation_mask(original, teacher), cond, empty
    frozen = state_digest(teacher.state_dict())
    initial = state_digest(student.state_dict())
    sampler = MaskRatioSampler()
    for step in range(steps):
        losses = run_update(acc,teacher,models,optimizers,schedulers,batch_fn,sampler,config,True,global_step=step)
    per_update = microbatches_per_update(config)
    assert stream.consumed == steps * per_update
    assert positions == list(range(steps * per_update))
    checkpoint = save_checkpoint(acc, out, steps, stream.consumed, config, models, optimizers, schedulers)
    progress = check_roundtrip(acc, checkpoint, config, models, optimizers, schedulers)
    run_update(acc, teacher, models, optimizers, schedulers, batch_fn, sampler, config, True,global_step=steps)
    expected_positions = positions[-per_update:]
    uninterrupted = state_digest(training_state(acc, models, optimizers, schedulers))
    uninterrupted_rng = state_digest(rng_state())
    load_checkpoint(acc, checkpoint, config, models, optimizers, schedulers)
    stream = new_stream(progress['consumed_batches'])
    run_update(acc, teacher, models, optimizers, schedulers, batch_fn, sampler, config, True,global_step=steps)
    assert positions[-per_update:] == expected_positions, 'Resumed data positions differ'
    assert uninterrupted == state_digest(training_state(acc, models, optimizers, schedulers)), 'Resumed step differs'
    assert uninterrupted_rng == state_digest(rng_state()), 'Resumed RNG differs'
    assert frozen == state_digest(teacher.state_dict()), 'Teacher changed'
    assert initial != state_digest(student.state_dict()), 'Student never updated'
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in ema.parameters())
    common = state_digest(training_state(acc, models, optimizers, schedulers))
    if workers > 1:
        hashes = [None] * workers
        torch.distributed.all_gather_object(hashes, common)
        assert len(set(hashes)) == 1, f'DDP states differ across ranks: {hashes}'
    losses['student']=[float(x) for x in losses['student']]
    losses['auxiliary']=[[float(x) for x in r] for r in losses['auxiliary']]
    result = dict(steps=steps,device=str(acc.device),precision=precision,parameter_report=parameter_report, rank=rank, workers=workers, gradient_isolation=True,
                  exact_resume=True, rng_resume=True, data_progress=progress['consumed_batches'],
                  microbatches_per_update=per_update, resumed_positions=positions[-per_update:],
                  r_init=r_init, fake_rounds=fake_rounds, losses=losses, state_digest=common)
    (Path(out) / f'result-rank{rank}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    acc.wait_for_everyone()
    if workers > 1:
        torch.distributed.destroy_process_group()
    return result


def worker(rank,workers,rendezvous,out,steps,r_init,fake_rounds,device,precision):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(workers),
                      LOCAL_WORLD_SIZE=str(workers), MASTER_ADDR='127.0.0.1', MASTER_PORT='29591', USE_LIBUV='0')
    if device=='cuda':torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl' if device=='cuda' else 'gloo',rank=rank,world_size=workers,
        init_method=Path(rendezvous).as_uri(), timeout=timedelta(seconds=180))
    smoke(out,steps,rank,workers,r_init,fake_rounds,device,precision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[1, 2], default=1)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--mixed-precision',choices=['no','bf16'],default='no')
    parser.add_argument('--output', required=True)
    parser.add_argument('--r-init', type=float, default=1.0)
    parser.add_argument('--fake-rounds', type=int, default=1)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.device=='cuda' and torch.cuda.device_count()<args.workers:parser.error('Requested smoke workers exceed visible CUDA devices')
    if args.steps<1:parser.error('steps must be positive')
    if args.workers == 1:
        print(json.dumps(smoke(out,args.steps,r_init=args.r_init,fake_rounds=args.fake_rounds,device=args.device,precision=args.mixed_precision), indent=2))
    else:
        with tempfile.TemporaryDirectory() as temporary:
            torch.multiprocessing.spawn(worker, args=(2, str(Path(temporary) / 'rendezvous'), str(out),
                                                      args.steps,args.r_init,args.fake_rounds,args.device,args.mixed_precision),nprocs=2)
        print(f'Two-process {args.device} training and exact checkpoint continuation passed')


if __name__ == '__main__':
    main()
