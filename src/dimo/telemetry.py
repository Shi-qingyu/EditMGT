"""Read-only diagnostics from existing training tensors; no random draws/forwards."""
from datetime import datetime
import math
from pathlib import Path
import json
import torch
import torch.distributed as dist


def validate_monitoring(config):
    m=config.get('monitoring')
    if m is None:return
    for k in ['loss_every','grad_every','diagnostic_every']:
        if type(m.get(k)) is not int or m[k]<1:raise ValueError(f'monitoring.{k} must be positive integer')
    if 'token_every' in m and (type(m['token_every']) is not int or m['token_every']<1):
        raise ValueError('monitoring.token_every must be positive integer')
    if any(type(x) is not int or x<1 for x in m.get('probe_steps',[])):
        raise ValueError('Monitoring probe steps must be positive integers')


def grad_norm(model):
    """Full accumulated L2 norm, without clipping/mutating gradients."""
    grads=[p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:raise RuntimeError('Missing gradients while measuring gradient norm')
    norms=torch._foreach_norm(grads,2)
    return torch.linalg.vector_norm(torch.stack([n.float() for n in norms]))


@torch.no_grad()
def probability_statistics(student,teacher,auxiliary,q,chunk_size=128):
    """Exact per-sample means over Q, then sample mean, matching loss weighting.

    Student logits are one-step initial logits. Teacher configured CFG and Aux CFG1 logits
    are evaluated on the same pseudo-corruption and mask ratio. Units: nats.
    Chunking bounds diagnostic memory without subsampling any edited token.
    """
    names=['entropy/student_q_nats','entropy/teacher_q_nats','entropy/auxiliary_q_nats',
           'gap/teacher_aux_kl_nats','gap/teacher_aux_tv','gap/teacher_aux_top1_agreement']
    sums=torch.zeros(len(names),device=student.device,dtype=torch.float64)
    vocab=student.shape[-1]
    for s,t,a,mask in zip(student.detach(),teacher.detach(),auxiliary.detach(),q):
        indices=mask.flatten().nonzero().flatten()
        if not len(indices):raise RuntimeError('Empty diagnostic Q')
        s,t,a=[x.reshape(-1,vocab) for x in [s,t,a]]
        sample_sum=torch.zeros_like(sums)
        for ids in indices.split(chunk_size):
            ls,lt,la=[x.index_select(0,ids).float().log_softmax(-1) for x in [s,t,a]]
            ps,pt,pa=[x.exp() for x in [ls,lt,la]]
            values=[-(ps*ls).sum(-1),-(pt*lt).sum(-1),-(pa*la).sum(-1),
                (pt*(lt-la)).sum(-1),.5*(pt-pa).abs().sum(-1),
                (lt.argmax(-1)==la.argmax(-1)).float()]
            sample_sum+=torch.stack([v.double().sum() for v in values])
        sums+=sample_sum/len(indices)
    if not torch.isfinite(sums).all():raise RuntimeError('Nonfinite entropy / Teacher-Aux diagnostic')
    return {k:v for k,v in zip(names,sums)},len(q)


def histogram_statistics(counts):
    """Global counts of generated codes in G, excluding source-locked exterior."""
    counts=counts.double();valid=counts[:-1];total=counts.sum();n=valid.sum()
    p=valid/n.clamp_min(1);positive=p>0
    entropy=-(p[positive]*p[positive].log()).sum()
    return dict(token_count=int(total.item()),valid_token_count=int(n.item()),
        invalid_token_count=int(counts[-1].item()),unique_codes=int((valid>0).sum().item()),
        vocab_coverage=float((valid>0).double().mean().item()),
        empirical_entropy_nats=float(entropy.item()),perplexity=float(entropy.exp().item()),
        top1_mass=float(p.max().item()),top10_mass=float(p.topk(min(10,len(p))).values.sum().item()))


class Telemetry:
    def __init__(self,config,global_step,device,codebook_size):
        self.enabled='monitoring' in config
        self.step=(global_step+1) if global_step is not None else None
        self.device=device;self.codebook_size=codebook_size;self.sums={};self.counts={};self.histograms={}
        if not self.enabled:
            self.loss_due=self.grad_due=self.diagnostic_due=self.token_due=False;return
        if self.step is None:raise ValueError('Monitoring requires global_step')
        m=config['monitoring'];probe=self.step in m.get('probe_steps',[])
        self.loss_due=probe or self.step%m['loss_every']==0
        self.grad_due=probe or self.step%m['grad_every']==0
        self.diagnostic_due=probe or self.step%m['diagnostic_every']==0
        self.token_due=probe or self.step%m.get('token_every',200)==0
        self.config=config

    def add_sum(self,name,value,count=1):
        if not torch.is_tensor(value):value=torch.tensor(value,device=self.device,dtype=torch.float64)
        value=value.detach().double()
        self.sums[name]=self.sums.get(name,torch.zeros((),device=self.device,dtype=torch.float64))+value
        self.counts[name]=self.counts.get(name,0)+count

    @torch.no_grad()
    def observe(self,student,teacher,auxiliary,q,sampled,region):
        if self.diagnostic_due:
            values,n=probability_statistics(student,teacher,auxiliary,q)
            for k,v in values.items():self.add_sum(k,v,n)
        self.add_sum('region/q_tokens',q.flatten(1).sum(1).double().sum(),len(q))
        self.add_sum('region/g_tokens',region.flatten(1).sum(1).double().sum(),len(q))
        if self.token_due:self.tokens('student_phase',sampled,region)

    @torch.no_grad()
    def tokens(self,name,tokens,region):
        if not self.token_due:return
        values=tokens[region].long()
        values=torch.where((values>=0)&(values<self.codebook_size),values,self.codebook_size)
        hist=torch.bincount(values,minlength=self.codebook_size+1)
        self.histograms[name]=self.histograms.get(name,torch.zeros_like(hist))+hist

    def publish(self,acc,out,losses,optimizers,seconds):
        if not (self.loss_due or self.grad_due or self.diagnostic_due or self.token_due):return None
        if self.loss_due:
            self.add_sum('loss/student',sum(losses['student'])/len(losses['student']))
            aux=[x for r in losses['auxiliary'] for x in r]
            self.add_sum('loss/auxiliary',sum(aux)/len(aux))
            for name,opt in optimizers.items():self.add_sum('lr/'+name,opt.param_groups[0]['lr'])
            self.add_sum('nan/nonfinite_losses',0)
            self.add_sum('nan/nonfinite_gradients',0)
            self.add_sum('ema/decay',losses['ema_decay'])
        keys=sorted(self.sums)
        metrics={}
        if keys:
            packed=torch.stack([torch.stack([self.sums[k],torch.tensor(self.counts[k],device=self.device,dtype=torch.float64)]) for k in keys])
            if acc.num_processes>1:dist.all_reduce(packed,op=dist.ReduceOp.SUM)
            metrics={k:float((v[0]/v[1]).item()) for k,v in zip(keys,packed)}
        elapsed=torch.tensor(seconds,device=self.device,dtype=torch.float64)
        if acc.num_processes>1:dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        metrics['timing/update_seconds_max_rank']=float(elapsed.item())
        histograms={}
        for name,counts in sorted(self.histograms.items()):
            if acc.num_processes>1:dist.all_reduce(counts,op=dist.ReduceOp.SUM)
            stats=histogram_statistics(counts)
            metrics.update({f'tokens/{name}/{k}':v for k,v in stats.items()})
            if acc.is_main_process:histograms[name]=counts.cpu().numpy()
        if not all(math.isfinite(v) for v in metrics.values()):raise RuntimeError('Nonfinite reduced telemetry')
        record=dict(step=self.step,timestamp=datetime.now().astimezone().isoformat(),metrics=metrics,
            world_size=acc.num_processes,loss_due=self.loss_due,grad_due=self.grad_due,
            diagnostic_due=self.diagnostic_due,token_due=self.token_due,
            teacher_cfg=self.config.get('teacher_cfg',4.0),noise_emb_perturb=self.config.get('noise_emb_perturb',0.0),
            source_metadata=self.config.get('source_metadata',{}))
        if acc.is_main_process:
            if histograms:
                import numpy as np
                folder=Path(out)/'logs/token_histograms';folder.mkdir(exist_ok=True)
                path=folder/f'step-{self.step:06d}.npz'
                np.savez_compressed(path,**histograms);record['histogram_path']=str(path)
            with (Path(out)/'logs/metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        return record
