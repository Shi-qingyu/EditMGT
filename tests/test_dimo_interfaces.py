"""Strict configuration/origin/checkpoint/inference and upstream adapter contracts."""
import copy,json,os,subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest,torch
from PIL import Image
from diffusers import VQModel

from test_dimo_update import case,deterministic_cpu
from src.dimo.smoke import tiny_model
from src.dimo.config import resolve_config,validate_config
from src.dimo.origin import select_teacher,discover,ProfileMismatch,sha256,load_strict
from src.dimo.infer import OneStepEditor,load_role
from src.dimo.checkpoint import save_checkpoint


@pytest.fixture
def origin_dirs(tmp_path):
    base=tmp_path/'base';run=tmp_path/'sft'
    for name in ['editmgt','vqvae','scheduler']:(base/name).mkdir(parents=True)
    (base/'editmgt/config.json').write_text(json.dumps(dict(codebook_size=31,vocab_size=32)))
    (base/'vqvae/config.json').write_text(json.dumps(dict(num_vq_embeddings=31)))
    (base/'scheduler/scheduler_config.json').write_text(json.dumps(dict(mask_token_id=31)))
    for step in [9,20,100]:tiny_model().save_pretrained(run/f'checkpoint-{step}'/'transformer')
    return run,base


def test_explicit_transformer_and_checkpoint_dir_strict_origin(origin_dirs):
    run,base=origin_dirs
    from src.dimo.origin import PROFILE
    config=dict(base_model_dir=str(base),expected_profile=PROFILE)
    _,a=select_teacher(dict(config,teacher_transformer_dir=str(run/'checkpoint-20')))
    _,b=select_teacher(dict(config,teacher_transformer_dir=str(run/'checkpoint-20/transformer')))
    assert a['checksums']==b['checksums'] and a['transformer_config']['sparse_target_halo']==0
    with pytest.raises(ValueError,match='exactly one'):
        select_teacher(dict(config,sft_run_dir=str(run),teacher_transformer_dir=a['transformer_dir']))
    with pytest.raises(FileNotFoundError):select_teacher(dict(config,teacher_transformer_dir=str(run/'missing')))


def test_latest_complete_numeric_only_and_profile_mismatch_blocks(origin_dirs):
    run,base=origin_dirs
    _,origin=discover(run,base)
    assert origin['step']==100
    (run/'checkpoint-101/transformer').mkdir(parents=True)
    with pytest.warns(UserWarning,match='incomplete'):
        _,origin=discover(run,base)
    assert origin['step']==100 and origin['rejected_candidates']
    path=run/'checkpoint-101/transformer/config.json'
    config=json.loads((run/'checkpoint-100/transformer/config.json').read_text())
    config['time_conditioning']='legacy';path.write_text(json.dumps(config))
    with pytest.raises(ProfileMismatch):discover(run,base)


def test_hf_pinned_local_snapshot_layout_inspected(origin_dirs,tmp_path,monkeypatch):
    import shutil,huggingface_hub
    from src.dimo.origin import PROFILE
    run,base=origin_dirs
    revision='a'*40;snapshot=tmp_path/'snapshots'/revision
    shutil.copytree(run/'checkpoint-20',snapshot/'checkpoint-20')
    calls=[]
    def download(**kwargs):calls.append(kwargs);return str(snapshot)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    config=dict(base_model_dir=str(base),expected_profile=PROFILE,
                teacher_hf=dict(repo_id='QingyuShi/SparseEdit',revision=revision))
    _,origin=select_teacher(config)
    assert origin['hf_revision']==revision and origin['transformer_dir'].endswith('checkpoint-20/transformer')
    assert calls[0]['local_files_only'] is True
    config['teacher_hf']['revision']='main'
    with pytest.raises(ValueError,match='fixed'):select_teacher(config)


def test_historical_defaults_and_v3_preset_are_distinct(tmp_path):
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'configs/dimo/region_dimo_v3.example.json').read_text())
    keys=['freeze_student_embedding','freeze_student_mlm_head','freeze_auxiliary_embedding','freeze_auxiliary_mlm_head']
    assert all(c[k] is True for k in keys)
    assert (c['teacher_cfg'],c['noise_emb_perturb'],c['lr_warmup_steps'])==(6.0,.3,100)
    for k in [*keys,'teacher_cfg','noise_emb_perturb','lr_warmup_steps','lr_scheduler']:c.pop(k)
    path=tmp_path/'old.json';path.write_text(json.dumps(c))
    old=resolve_config(path,{})
    assert all(old[k] is False for k in keys)
    assert old['teacher_cfg']==4 and old['noise_emb_perturb']==0 and old['lr_scheduler']=='constant'


def test_nonexistent_explicit_manifest_never_falls_back(origin_dirs,tmp_path):
    root=Path(__file__).resolve().parents[1]
    run,base=origin_dirs
    c=json.loads((root/'configs/dimo/region_dimo_v3.example.json').read_text())
    c.update(sft_run_dir=str(run),base_model_dir=str(base),output_dir=str(tmp_path/'new-run'),
             data_manifest=str(tmp_path/'missing.jsonl'),data_mode='local_stream')
    with pytest.raises(FileNotFoundError):validate_config(c)
    c['teacher_cfg']=float('nan')
    with pytest.raises(ValueError,match='teacher_cfg'):validate_config(c)
    c['teacher_cfg']=6;c['teacher_transformer_dir']=str(run/'checkpoint-20')
    with pytest.raises(ValueError,match='exactly one'):validate_config(c)


def test_infer_direct_vq_decode_seed_reproducibility_and_optional_composite(monkeypatch):
    import numpy as np
    import src.dataset_utils as data
    model=tiny_model().eval()
    vq=VQModel(block_out_channels=(16,),latent_channels=4,vq_embed_dim=4,
               num_vq_embeddings=31,norm_num_groups=4,lookup_from_codebook=True).eval()
    monkeypatch.setattr(data,'tokenize_prompt',lambda *a,**kw:None)
    hidden,pooled=torch.randn(1,3,24),torch.randn(1,8)
    monkeypatch.setattr(data,'encode_prompt',lambda *a,**kw:(hidden,pooled))
    editor=OneStepEditor(model,[],None,vq,SimpleNamespace(config=SimpleNamespace(mask_token_id=31)),
                         device='cpu',resolution=16,halo=1)
    source=Image.new('RGB',(16,16),'gray');mask=Image.new('L',(16,16));mask.paste(255,(6,6,9,9))
    before=torch.get_rng_state()
    a,protocol=editor.edit(source,mask,'paint red',seed=7)
    assert torch.equal(torch.get_rng_state(),before)
    b,_=editor.edit(source,mask,'paint red',seed=7)
    assert np.array_equal(np.array(a),np.array(b))
    assert protocol['cfg']==1 and protocol['steps']==1 and protocol['noise_emb_perturb']==0
    assert protocol['output_compositing']=='none' and protocol['halo']==1
    assert protocol['halo_source']==protocol['supervise_target_halo_source']=='cli'
    composite,p=editor.edit(source,mask,'paint red',seed=7,preserve_unmasked_pixels=True)
    outside=np.array(mask)==0
    assert np.array_equal(np.array(composite)[outside],np.array(source)[outside])
    assert p['preserve_unmasked_pixels'] and not p['direct_vq_decode']
    with pytest.raises(ValueError,match='Empty'):editor.edit(source,Image.new('L',(16,16)),'paint red')


def test_load_selected_role_and_export_preserves_config(case,tmp_path,monkeypatch):
    c=case
    config=dict(c.config,mixed_precision='no',resolution=16)
    (tmp_path/'teacher').mkdir()
    c.teacher.save_pretrained(tmp_path/'teacher')
    origin=dict(pinned_teacher_checksums={'config.json':sha256(tmp_path/'teacher/config.json')})
    (tmp_path/'teacher_origin.json').write_text(json.dumps(origin))
    checkpoint=save_checkpoint(c.acc,tmp_path,0,0,config,c.models,c.optimizers,c.schedulers)
    model,metadata=load_role(checkpoint,'ema')
    assert metadata['model']=='ema' and metadata['config']==config
    for key,value in model.state_dict().items():assert torch.equal(value,c.models['ema'].state_dict()[key])
    from src.dimo import export
    destination=tmp_path/'exported'
    monkeypatch.setattr('sys.argv',['export','--checkpoint',str(checkpoint),'--model','ema','--output',str(destination)])
    export.main()
    exported,info=load_role(destination,'ema')
    assert info['config']==config and info['inference_defaults']['cfg']==1
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in exported.state_dict().items())
    with pytest.raises(ValueError,match='role'):load_role(destination,'student')


@pytest.mark.parametrize('backend',['student','ema'])
@pytest.mark.parametrize('halo',[None,0,1])
@pytest.mark.parametrize('saved_supervise',[False,True])
def test_benchmark_adapter_keeps_upstream_order_masks_metrics_and_restores_bindings(
        tmp_path,monkeypatch,backend,halo,saved_supervise):
    from src.dimo.benchmark import backend_context,parse_args,StudentRunner,protocol_config
    import src.benchmark_utils as bu
    import src.benchmark_parallel as bp
    checkpoint=tmp_path/'checkpoint-1';checkpoint.mkdir()
    (checkpoint/'COMPLETE').write_text('{}')
    (tmp_path/'teacher').mkdir()
    profile=dict(sparse_target_halo=1,supervise_target_halo=saved_supervise)
    (tmp_path/'teacher/config.json').write_text(json.dumps(profile))
    halo_args=[] if halo is None else ['--sparse-target-halo',str(halo)]
    args,_=parse_args(['--benchmark','inter_edit','--backend',backend,'--checkpoint',str(checkpoint),
            '--guidance-scale','1','--num-inference-steps','1','--stage','generate','--device','cpu',*halo_args])
    assert args.sparse_target_halo==halo
    expected_halo=1 if halo is None else halo
    expected_supervise=saved_supervise if halo is None else bool(halo)
    expected_source='checkpoint' if halo is None else 'cli'
    runner,config,worker=bu.InferenceRunner,bu.generation_config,bp._spawn_workers
    select,score,partition=bu.generate_benchmark,bu.score_benchmark,bp.shard_inter_edit_samples
    with backend_context(args):
        assert bu.InferenceRunner is StudentRunner
        settings=bu.generation_config(args,'inter_edit')
        assert settings['guidance_scale']==1 and settings['dimo_model']==backend and settings['cache_protocol']=='disabled'
        assert settings['sparse_target_halo']==expected_halo
        assert settings['supervise_target_halo'] is expected_supervise
        assert settings['halo_source']==settings['supervise_target_halo_source']==expected_source
        assert (bu.generate_benchmark,bu.score_benchmark,bp.shard_inter_edit_samples)==(select,score,partition)
    assert (bu.InferenceRunner,bu.generation_config,bp._spawn_workers)==(runner,config,worker)
    import src.dimo.benchmark as benchmark
    model=tiny_model();model.register_to_config(**profile)
    vq=torch.nn.Identity();vq.config=SimpleNamespace(num_vq_embeddings=31)
    scheduler=SimpleNamespace(config=SimpleNamespace(mask_token_id=31))
    monkeypatch.setattr(benchmark,'load_role',lambda path,role:(model,dict(config={},model=role)))
    monkeypatch.setattr(benchmark,'load_components',lambda *a:([],None,vq,scheduler))
    student_runner=StudentRunner(args)
    assert model.config.sparse_target_halo==expected_halo
    assert model.config.supervise_target_halo is expected_supervise
    assert student_runner.metadata['sparse_target_halo']==expected_halo
    assert student_runner.metadata['supervise_target_halo'] is expected_supervise
    assert student_runner.metadata['halo_source']==expected_source
    student_runner.close()
    with pytest.raises(SystemExit):parse_args(['--benchmark','inter_edit','--backend','student','--checkpoint',str(tmp_path)])


def test_launcher_respects_visible_gpus_and_explicit_override(tmp_path):
    import sys
    root=Path(__file__).resolve().parents[1]
    shim=tmp_path/'python'
    shim.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
if sys.argv[1:2]==['-c']:print(3)
else:print(json.dumps(dict(argv=sys.argv[1:],visible=os.environ.get('CUDA_VISIBLE_DEVICES'),workers=os.environ['DIMO_NUM_PROCESSES'])))
''')
    shim.chmod(0o755)
    env=dict(os.environ,DIMO_PYTHON=str(shim),DIMO_CONFIG='private.json',CUDA_VISIBLE_DEVICES='3,5,7')
    env.pop('NUM_PROCESSES',None);env.pop('GPU_IDS',None)
    def launch(env):return subprocess.run(['bash','scripts/dimo/run_formal.sh','--stop-after-preflight'],cwd=root,env=env,text=True,capture_output=True)
    result=launch(env)
    assert result.returncode==0
    record=json.loads(result.stdout.splitlines()[-1])
    assert record['visible']=='3,5,7' and record['workers']=='3'
    assert '--nproc_per_node=3' in record['argv']
    result=launch(dict(env,GPU_IDS='6',NUM_PROCESSES='1'))
    record=json.loads(result.stdout.splitlines()[-1])
    assert record['visible']=='6' and record['workers']=='1' and 'torch.distributed.run' not in record['argv']
    result=launch(dict(env,NUM_PROCESSES='4'))
    assert result.returncode!=0 and 'Invalid NUM_PROCESSES' in result.stderr


@pytest.mark.parametrize('halo',[None,0,1])
def test_dimo_teacher_preserves_original_benchmark_halo_default(tmp_path,halo):
    from src.dimo.benchmark import parse_args
    halo_args=[] if halo is None else ['--sparse-target-halo',str(halo)]
    args,_=parse_args(['--benchmark','inter_edit','--backend','teacher',
                      '--checkpoint',str(tmp_path),'--stage','generate',*halo_args])
    assert args.sparse_target_halo==(0 if halo is None else halo)


@pytest.mark.parametrize('profile',[
    {},dict(sparse_target_halo=1),dict(sparse_target_halo=True,supervise_target_halo=True),
    dict(sparse_target_halo=1,supervise_target_halo=None),
])
def test_dimo_halo_rejects_missing_or_invalid_saved_settings(profile):
    from src.dimo.infer import resolve_halo_profile
    with pytest.raises(ValueError,match='[Hh]alo'):
        resolve_halo_profile(profile)
