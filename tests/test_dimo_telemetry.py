import math
from types import SimpleNamespace
import json
import torch
from src.dimo.telemetry import probability_statistics,histogram_statistics,Telemetry,grad_norm


def test_exact_entropy_kl_and_tv_use_q_and_per_sample_weighting():
    torch.manual_seed(5)
    s,t,a=[torch.randn(2,3,4,7,requires_grad=True) for _ in range(3)]
    q=torch.zeros(2,3,4,dtype=torch.bool);q[0,0,0]=True;q[1,:2,:3]=True
    stats,n=probability_statistics(s,t,a,q,chunk_size=2)
    ls,lt,la=[x.log_softmax(-1) for x in [s,t,a]]
    pt=lt.exp();pa=la.exp()
    values={'entropy/student_q_nats':-(ls.exp()*ls).sum(-1),
        'entropy/teacher_q_nats':-(pt*lt).sum(-1),
        'entropy/auxiliary_q_nats':-(pa*la).sum(-1),
        'gap/teacher_aux_kl_nats':(pt*(lt-la)).sum(-1),
        'gap/teacher_aux_tv':.5*(pt-pa).abs().sum(-1),
        'gap/teacher_aux_top1_agreement':(lt.argmax(-1)==la.argmax(-1)).float()}
    assert n==2
    for k,v in values.items():
        expected=sum(v[i][q[i]].double().mean() for i in range(2))
        torch.testing.assert_close(stats[k],expected)
        assert not stats[k].requires_grad
    equal,_=probability_statistics(s,t,t,q)
    assert equal['gap/teacher_aux_tv']==0 and equal['gap/teacher_aux_kl_nats']==0


def test_token_histograms_ignore_locked_exterior_and_track_invalid_codes(tmp_path):
    c=dict(monitoring=dict(loss_every=10,grad_every=50,diagnostic_every=100,token_every=100,probe_steps=[]))
    tel=Telemetry(c,99,torch.device('cpu'),4)
    tokens=torch.tensor([[[0,0,1],[3,9,0]]]);g=torch.tensor([[[0,1,1],[1,1,0]]],dtype=torch.bool)
    tel.tokens('student_phase',tokens,g)
    assert tel.histograms['student_phase'].tolist()==[1,1,0,1,1]
    h=histogram_statistics(tel.histograms['student_phase'])
    assert h['token_count']==4 and h['invalid_token_count']==1 and h['unique_codes']==3
    assert math.isclose(h['perplexity'],3.)


def test_cadence_and_unclipped_norm():
    c=dict(monitoring=dict(loss_every=10,grad_every=50,diagnostic_every=100,token_every=100,probe_steps=[1,20]))
    for step,expected in [(1,(1,1,1)),(10,(1,0,0)),(20,(1,1,1)),(50,(1,1,0)),(100,(1,1,1)),(101,(0,0,0))]:
        tel=Telemetry(c,step-1,torch.device('cpu'),4)
        assert (tel.loss_due,tel.grad_due,tel.diagnostic_due)==expected
    m=torch.nn.Linear(2,1,bias=False);m.weight.grad=torch.tensor([[3.,4.]])
    assert grad_norm(m)==5
    assert torch.equal(m.weight.grad,torch.tensor([[3.,4.]]))


def test_single_process_publish_and_global_histogram_artifact(tmp_path):
    (tmp_path/'logs').mkdir()
    tel=Telemetry(dict(monitoring=dict(loss_every=10,grad_every=50,diagnostic_every=100,token_every=100)),99,torch.device('cpu'),4)
    tel.add_sum('grad/student_pre_clip_l2',3.)
    tel.tokens('student_phase',torch.tensor([[[0,1]]]),torch.ones(1,1,2,dtype=torch.bool))
    opt=torch.optim.AdamW(torch.nn.Linear(2,1).parameters(),lr=1e-6)
    acc=SimpleNamespace(num_processes=1,is_main_process=True)
    rec=tel.publish(acc,tmp_path,dict(student=[1.,3.],auxiliary=[[2.,4.]],ema_decay=.8),
        dict(student=opt,auxiliary=opt),2.)
    assert rec['metrics']['loss/student']==2
    assert rec['metrics']['tokens/student_phase/unique_codes']==2
    assert json.loads((tmp_path/'logs/metrics.jsonl').read_text())['step']==100
    assert (tmp_path/'logs/token_histograms/step-000100.npz').exists()


def _distributed_worker(rank,root):
    import torch.distributed as dist
    from pathlib import Path
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(root)/'rendezvous'),rank=rank,world_size=2)
    tel=Telemetry(dict(monitoring=dict(loss_every=10,grad_every=50,diagnostic_every=100,token_every=100)),99,torch.device('cpu'),4)
    tel.add_sum('grad/student_pre_clip_l2',1+2*rank)
    tel.tokens('student_phase',torch.tensor([[[rank]]]),torch.ones(1,1,1,dtype=torch.bool))
    opt=torch.optim.AdamW(torch.nn.Linear(2,1).parameters(),lr=1e-6)
    rec=tel.publish(SimpleNamespace(num_processes=2,is_main_process=rank==0),Path(root),
        dict(student=[2+2*rank],auxiliary=[[3+2*rank]],ema_decay=.8),dict(student=opt,auxiliary=opt),1+rank)
    assert rec['metrics']['loss/student']==3
    assert rec['metrics']['grad/student_pre_clip_l2']==2
    assert rec['metrics']['tokens/student_phase/unique_codes']==2
    assert rec['metrics']['tokens/student_phase/token_count']==2
    assert rec['metrics']['timing/update_seconds_max_rank']==2
    dist.destroy_process_group()


def test_distributed_reduction_keeps_global_unique_histogram(tmp_path):
    (tmp_path/'logs').mkdir()
    torch.multiprocessing.start_processes(_distributed_worker,args=(str(tmp_path),),nprocs=2,start_method='fork',join=True)
