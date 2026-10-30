"""Official sequential generator/fake_model phases, using real tiny EditMGT."""
import copy
from functools import wraps
import json
from types import SimpleNamespace

from accelerate import Accelerator
import pytest
import torch
import torch.nn.functional as F

import src.dimo.core as core
import src.dimo.train as training
from src.dimo.checkpoint import (save_checkpoint, load_checkpoint, state_digest, training_state,
                                 rng_state, microbatches_per_update, CHECKPOINT_KIND)
from src.dimo.origin import sha256
from src.dimo.smoke import tiny_model
from src.timestep_sampling import MaskRatioSampler


def test_bf16_sparse_distillation_forward_backward(case):
    c = case
    source, original, region, cond, empty = c.batch_fn()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, _ = core.distillation_losses(c.teacher, c.models['student'], c.models['auxiliary'],
            source, original, region, 31, cond, empty, torch.full((len(source),), 0.7))
    assert torch.isfinite(loss)
    loss.backward()
    training.assert_gradients(c.models['student'], 'Student', True)
    training.assert_gradients(c.teacher, 'Teacher', False)
    training.assert_gradients(c.models['auxiliary'], 'Auxiliary', False)


@pytest.mark.parametrize('warmup', [False, True])
def test_budget_extension_restores_full_state_and_exact_next_update(case, tmp_path, warmup):
    c = case
    c.config['max_steps'] = 3
    c.config['ema_warmup'] = warmup
    def update():
        return training.run_update(c.acc, c.teacher, c.models, c.optimizers,
                                   c.schedulers, c.batch_fn, c.sampler, c.config, True,
                                   global_step=len(c.positions) // 4)
    update()
    checkpoint = save_checkpoint(c.acc, tmp_path, 1, 4, c.config,
                                 c.models, c.optimizers, c.schedulers)
    expected_losses = update()
    expected = state_digest(training_state(c.acc, c.models, c.optimizers, c.schedulers))
    expected_rng = state_digest(rng_state())
    expected_positions = list(c.positions)
    c.config['max_steps'] = 6
    progress = load_checkpoint(c.acc, checkpoint, c.config, c.models, c.optimizers, c.schedulers)
    c.positions[:] = c.positions[:progress['consumed_batches']]
    resumed_losses = update()
    assert resumed_losses == expected_losses
    assert resumed_losses['ema_decay'] == (min(0.8, 2 / 11) if warmup else 0.8)
    assert c.positions == expected_positions
    assert state_digest(training_state(c.acc, c.models, c.optimizers, c.schedulers)) == expected
    assert state_digest(rng_state()) == expected_rng


def test_diagnostics_preserve_update_rng_and_resume(case,tmp_path):
    from src.dimo.telemetry import Telemetry
    c=case
    c.config.update(ema_warmup=True,monitoring=dict(loss_every=10,grad_every=50,diagnostic_every=100,probe_steps=[1]))
    checkpoint=save_checkpoint(c.acc,tmp_path,0,0,c.config,c.models,c.optimizers,c.schedulers)
    expected_loss=training.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,c.batch_fn,c.sampler,c.config,False,global_step=0)
    expected_state=state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers))
    expected_rng=state_digest(rng_state())
    load_checkpoint(c.acc,checkpoint,c.config,c.models,c.optimizers,c.schedulers);c.positions.clear()
    telemetry=Telemetry(c.config,0,torch.device('cpu'),c.teacher.config.codebook_size)
    actual_loss=training.run_update(c.acc,c.teacher,c.models,c.optimizers,c.schedulers,c.batch_fn,c.sampler,c.config,False,global_step=0,telemetry=telemetry)
    assert expected_loss==actual_loss
    assert state_digest(training_state(c.acc,c.models,c.optimizers,c.schedulers))==expected_state
    assert state_digest(rng_state())==expected_rng
    assert 'entropy/student_q_nats' in telemetry.sums
    assert 'grad/auxiliary_unclipped_l2' in telemetry.sums
    assert set(telemetry.histograms)=={'student_phase','aux_targets'}




def test_warmup_rejects_missing_step_before_updating(case):
    c = case
    c.config['ema_warmup'] = True
    before = state_digest(training_state(c.acc, c.models, c.optimizers, c.schedulers))
    with pytest.raises(ValueError, match='global_step'):
        training.run_update(c.acc, c.teacher, c.models, c.optimizers, c.schedulers,
                            c.batch_fn, c.sampler, c.config, True)
    assert c.positions == []
    assert state_digest(training_state(c.acc, c.models, c.optimizers, c.schedulers)) == before


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(108)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def case():
    acc = Accelerator(cpu=True)
    teacher, student, auxiliary, ema = core.clone_models(tiny_model())
    models = dict(student=student, auxiliary=auxiliary, ema=ema)
    optimizers = {name: torch.optim.AdamW(models[name].parameters(), lr=1e-3)
                  for name in ('student', 'auxiliary')}
    schedulers = {name: torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
                  for name, opt in optimizers.items()}
    original = torch.zeros(2, 6, 8, dtype=torch.bool)
    original[0, 1, 3] = True
    original[1, 2:6, 4:8] = True
    rows, cols = torch.meshgrid(torch.arange(3), torch.arange(4), indexing='ij')
    ids = torch.stack((torch.zeros_like(rows), rows, cols), -1).reshape(-1, 3)
    hidden, pooled = torch.randn(2, 3, 24), torch.randn(2, 8)
    positions = []
    def batch_fn():
        position = len(positions)
        positions.append(position)
        source = torch.full((2, 6, 8), position, dtype=torch.long)
        cond = dict(encoder_hidden_states=hidden, pooled_projections=pooled,
                    micro_conds=torch.full((2, 5), float(position)), img_ids=ids,
                    reference_image_ids=ids, txt_ids=torch.zeros(3, 3), target_edit_mask=original,
                    reference_image_hidden_states=source, lora_part_enable=True)
        empty = dict(cond, encoder_hidden_states=torch.zeros_like(hidden), pooled_projections=torch.zeros_like(pooled))
        return source, original, core.generation_mask(original, teacher), cond, empty
    config = dict(gradient_accumulation=2, fake_rounds=1, auxiliary_label_smoothing=0.1,
                  student_initial_mask_ratio=1.0, ema_decay=0.8, mixed_precision='no')
    return SimpleNamespace(acc=acc, teacher=teacher, models=models, optimizers=optimizers,
                           schedulers=schedulers, batch_fn=batch_fn, sampler=MaskRatioSampler(),
                           config=config, positions=positions, original=original)


@pytest.mark.parametrize('fake_rounds', [1, 2])
@pytest.mark.parametrize('halo', [0, 1])
def test_official_order_fresh_batches_updated_student_gradients_and_ema(case, monkeypatch, fake_rounds, halo):
    c = case
    c.config['fake_rounds'] = fake_rounds
    for model in (c.teacher, *c.models.values()):
        model.register_to_config(sparse_target_halo=halo, supervise_target_halo=bool(halo))
    student, auxiliary, ema = (c.models[k] for k in ('student', 'auxiliary', 'ema'))
    teacher_before = state_digest(c.teacher.state_dict())
    student_before = state_digest(student.state_dict())
    aux_before = state_digest(auxiliary.state_dict())
    ema_before = copy.deepcopy(ema.state_dict())
    events, masks, student_steps, auxiliary_steps, ce_calls = [], [], [], [], []
    generated_targets = []
    handles = []
    def forward_hook(label):
        def hook(model, args, kwargs):
            position = int(kwargs['micro_conds'][0, 0])
            assert torch.equal(kwargs['target_edit_mask'], c.original)
            events.append((label, position, torch.is_grad_enabled(), state_digest(student.state_dict())))
            if label == 'student' and not torch.is_grad_enabled():
                assert student_steps and state_digest(student.state_dict()) == student_steps[0]
                assert state_digest(ema.state_dict()) == state_digest(ema_before)
        return hook
    for name, model in (('student', student), ('auxiliary', auxiliary), ('teacher', c.teacher)):
        handles.append(model.register_forward_pre_hook(forward_hook(name), with_kwargs=True))
    def mask_spy(region, ratio):
        q = actual_mask(region, ratio)
        masks.append(q)
        assert not (q & ~region).any()
        events.append(('pseudo_mask', len(c.positions) - 1))
        return q
    actual_mask = core.pseudo_mask
    monkeypatch.setattr(core, 'pseudo_mask', mask_spy)
    monkeypatch.setattr(training, 'pseudo_mask', mask_spy)
    actual_ce = core.auxiliary_ce
    actual_generate = core.generate_one_step
    def generate_spy(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs['r_init'] == c.config['student_initial_mask_ratio']
        tokens = actual_generate(*args, **kwargs)
        generated_targets.append((len(c.positions) - 1, tokens.clone()))
        events.append(('aux_targets', len(c.positions) - 1))
        return tokens
    monkeypatch.setattr(training, 'generate_one_step', generate_spy)
    def ce_spy(logits, targets, q, **kwargs):
        assert not targets.requires_grad
        assert kwargs['label_smoothing'] == 0.1
        assert generated_targets[-1][0] == len(c.positions) - 1
        assert torch.equal(targets, generated_targets[-1][1])
        ce_calls.append((len(c.positions) - 1, targets.clone(), q))
        return actual_ce(logits, targets, q, **kwargs)
    monkeypatch.setattr(training, 'auxiliary_ce', ce_spy)
    actual_student_step = c.optimizers['student'].step
    @wraps(actual_student_step)
    def student_step(*args, **kwargs):
        assert c.positions == [0, 1]
        assert state_digest(auxiliary.state_dict()) == aux_before
        assert all(p.grad is None for p in auxiliary.parameters())
        assert any(p.grad is not None and p.grad.count_nonzero() for p in student.parameters())
        result = actual_student_step(*args, **kwargs)
        student_steps.append(state_digest(student.state_dict()))
        assert student_steps[-1] != student_before
        events.append(('student_step',))
        return result
    monkeypatch.setattr(c.optimizers['student'], 'step', student_step)
    actual_aux_step = c.optimizers['auxiliary'].step
    @wraps(actual_aux_step)
    def aux_step(*args, **kwargs):
        assert len(student_steps) == 1
        assert state_digest(student.state_dict()) == student_steps[0]
        assert all(p.grad is None for p in student.parameters())
        assert any(p.grad is not None and p.grad.count_nonzero() for p in auxiliary.parameters())
        assert state_digest(ema.state_dict()) == state_digest(ema_before)
        result = actual_aux_step(*args, **kwargs)
        auxiliary_steps.append(len(c.positions))
        events.append(('auxiliary_step',))
        return result
    monkeypatch.setattr(c.optimizers['auxiliary'], 'step', aux_step)
    actual_ema = core.update_ema
    def ema_spy(*args, **kwargs):
        assert len(auxiliary_steps) == fake_rounds
        events.append(('ema',))
        return actual_ema(*args, **kwargs)
    monkeypatch.setattr(training, 'update_ema', ema_spy)
    losses = training.run_update(c.acc, c.teacher, c.models, c.optimizers, c.schedulers,
                                c.batch_fn, c.sampler, c.config, True)
    for handle in handles:
        handle.remove()
    per_update = 2 * (1 + fake_rounds)
    assert c.positions == list(range(per_update))
    assert losses['microbatches'] == per_update
    assert len(losses['student']) == 2 and len(losses['auxiliary']) == fake_rounds
    assert len(student_steps) == 1 and auxiliary_steps == list(range(4, per_update + 1, 2))
    assert c.schedulers['student'].last_epoch == 1
    assert c.schedulers['auxiliary'].last_epoch == fake_rounds
    assert [e[1] for e in events if e[0] == 'student' and e[2]] == [0, 1]
    assert [e[1] for e in events if e[0] == 'student' and not e[2]] == list(range(2, per_update))
    assert [e[1] for e in events if e[0] == 'auxiliary' and e[2]] == list(range(2, per_update))
    step_index = events.index(('student_step',))
    assert all(i < step_index for i, e in enumerate(events) if e[0] == 'student' and e[2])
    assert all(i > step_index for i, e in enumerate(events) if e[0] == 'auxiliary' and e[2])
    assert all(e[3] == student_steps[0] for e in events if e[0] == 'student' and not e[2])
    assert [position for position, _, _ in ce_calls] == list(range(2, per_update))
    assert [position for position, _ in generated_targets] == list(range(2, per_update))
    for position in range(2, per_update):
        assert events.index(('aux_targets', position)) < events.index(('pseudo_mask', position))
    assert len(masks) == per_update and len({q.data_ptr() for q in masks}) == per_update
    assert state_digest(c.teacher.state_dict()) == teacher_before
    assert state_digest(auxiliary.state_dict()) != aux_before
    assert events[-1] == ('ema',)
    for key, value in ema.state_dict().items():
        expected = torch.lerp(ema_before[key], student.state_dict()[key], 1 - c.config['ema_decay'])
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert all(p.grad is None for model in (c.teacher, student, auxiliary, ema) for p in model.parameters())


def test_auxiliary_hard_ce_default_smoothing_and_per_sample_mask_normalization():
    logits = torch.randn(2, 2, 3, 7, requires_grad=True)
    labels = torch.randint(7, (2, 2, 3))
    q = torch.tensor([[[1, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 0, 0]]], dtype=torch.bool)
    values = F.cross_entropy(logits.movedim(-1, 1), labels, label_smoothing=0.1, reduction='none')
    expected = (values[0][q[0]].mean() + values[1][q[1]].mean()) / 2
    loss = core.auxiliary_ce(logits, labels, q)
    torch.testing.assert_close(loss, expected)
    assert not torch.isclose(loss, core.auxiliary_ce(logits, labels, q, label_smoothing=0))
    loss.backward()
    assert logits.grad[~q].eq(0).all()


@pytest.mark.parametrize('r_init', [1.0, 0.5])
def test_one_step_initialization_uses_random_vq_codes_inside_g_and_source_outside(case, monkeypatch, r_init):
    c = case
    source, original, region, cond, _ = c.batch_fn()
    # Make random-code initialization observable without relying on a chance draw.
    monkeypatch.setattr(core.torch, 'randint', lambda high, size, device: torch.full(size, 5, device=device))
    calls = []
    handle = c.models['student'].register_forward_pre_hook(lambda m, a, kw: calls.append(kw), with_kwargs=True)
    logits, sampled = core.generate_one_step(c.models['student'], source, original, 31, cond,
                                             r_init=r_init, return_logits=True)
    handle.remove()
    assert len(calls) == 1 and logits.requires_grad and not sampled.requires_grad
    call = calls[0]
    assert torch.equal(call['target_edit_mask'], original)
    assert call['mask_ratio'].eq(r_init).all() and call['timestep'].eq(r_init).all()
    assert torch.equal(call['hidden_states'][~region], source[~region])
    assert torch.equal(sampled[~region], source[~region])
    masked = call['hidden_states'].eq(31)
    assert not (masked & ~region).any()
    assert torch.equal(masked.flatten(1).sum(1), (region.flatten(1).sum(1) * r_init).round().clamp_min(1).long())
    if r_init == 0.5:
        assert call['hidden_states'][region & ~masked].eq(5).all()


@pytest.mark.parametrize('r_init', [0, -1, 1.1, float('nan'), True])
def test_invalid_initial_mask_ratios_are_rejected(case, r_init):
    source, original, _, cond, _ = case.batch_fn()
    with pytest.raises(ValueError, match='r_init'):
        core.generate_one_step(case.models['student'], source, original, 31, cond, r_init=r_init)


@pytest.fixture
def saved_case(case, tmp_path):
    c = case
    training.run_update(c.acc, c.teacher, c.models, c.optimizers, c.schedulers,
                        c.batch_fn, c.sampler, c.config, True)
    path = save_checkpoint(c.acc, tmp_path, 1, len(c.positions), c.config, c.models, c.optimizers, c.schedulers)
    return c, path


@pytest.mark.parametrize('damage', ['manifest_v1', 'progress_v1', 'common_v1', 'old_data_position'])
def test_incompatible_legacy_state_rejected_before_models_optimizers_or_rng_change(saved_case, damage):
    c, path = saved_case
    manifest_path = path / 'COMPLETE'
    manifest = json.loads(manifest_path.read_text())
    if damage == 'manifest_v1':
        manifest.pop('kind')
        manifest.pop('update_rule')
    else:
        filename = 'training.pt' if damage == 'common_v1' else 'rank-0.pt'
        state = torch.load(path / filename, weights_only=False)
        if damage == 'old_data_position':
            state['consumed_batches'] = c.config['gradient_accumulation']
        else:
            state['kind'] = 'region-dimo-v1'
        torch.save(state, path / filename)
        manifest['checksums'][filename] = sha256(path / filename)
    manifest_path.write_text(json.dumps(manifest))
    before = state_digest(training_state(c.acc, c.models, c.optimizers, c.schedulers))
    rng_before = state_digest(rng_state())
    with pytest.raises(RuntimeError, match='Incompatible'):
        load_checkpoint(c.acc, path, c.config, c.models, c.optimizers, c.schedulers)
    assert before == state_digest(training_state(c.acc, c.models, c.optimizers, c.schedulers))
    assert rng_before == state_digest(rng_state())


def test_checkpoint_counts_both_phases_and_rejects_partial_update(case, tmp_path):
    c = case
    assert microbatches_per_update(c.config) == 4
    with pytest.raises(RuntimeError, match='both phases'):
        save_checkpoint(c.acc, tmp_path, 1, 2, c.config, c.models, c.optimizers, c.schedulers)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fake_rounds', [0, -1, 1.5, True])
def test_invalid_fake_rounds_are_rejected(fake_rounds):
    with pytest.raises(ValueError, match='positive integers'):
        microbatches_per_update(dict(gradient_accumulation=2, fake_rounds=fake_rounds))


def test_read_and_prepare_uses_one_guard_per_microbatch(monkeypatch):
    from src.dimo.data import ReplayStream
    events=[]
    def batches():
        for index in range(2):
            events.append(('read',index))
            yield index
    def prepare(batch,*args):
        events.append(('prepare',batch))
        return ('prepared',batch)
    original=training.collective_call
    def guard(acc,action):
        events.append(('guard',))
        return original(acc,action)
    monkeypatch.setattr(training,'prepare_batch',prepare)
    monkeypatch.setattr(training,'collective_call',guard)
    acc=SimpleNamespace(num_processes=1,is_main_process=True)
    stream=ReplayStream(iter(batches()),seed=17)
    results=[training.prepare_next_batch(acc,stream,None,None,None,None) for _ in range(2)]
    assert results==[(0,('prepared',0)),(1,('prepared',1))]
    assert events==[('guard',),('read',0),('prepare',0),('guard',),('read',1),('prepare',1)]
    assert stream.consumed==2


def _batch_preparation_error_worker(rank,folder):
    from datetime import timedelta
    from pathlib import Path
    from unittest.mock import patch
    import torch.distributed as dist
    folder=Path(folder)
    dist.init_process_group('gloo',rank=rank,world_size=2,
                            init_method=(folder/'rendezvous').as_uri(),timeout=timedelta(seconds=30))
    records={}
    acc=SimpleNamespace(num_processes=2,is_main_process=rank==0,process_index=rank)
    try:
        for failure in ('none','read','prepare'):
            class Stream:
                def __next__(self):
                    if rank==1 and failure=='read':raise OSError('rank 1 read failed')
                    return rank
            def prepare(batch,*args):
                if rank==1 and failure=='prepare':raise ValueError('rank 1 prepare failed')
                return ('prepared',batch)
            with patch.object(training,'prepare_batch',prepare), \
                    patch.object(dist,'all_gather_object',wraps=dist.all_gather_object) as exchange:
                try:
                    result=training.prepare_next_batch(acc,Stream(),None,None,None,None)
                    assert failure=='none'
                    assert result==(rank,('prepared',rank))
                    records[failure]=dict(ok=True)
                except RuntimeError as error:
                    assert failure!='none'
                    assert f'rank 1 {failure} failed' in str(error)
                    records[failure]=dict(ok=False,error=str(error))
                assert exchange.call_count==1
        (folder/f'result-rank{rank}.json').write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


def test_read_and_prepare_propagates_rank_errors_with_one_exchange(tmp_path):
    torch.multiprocessing.spawn(_batch_preparation_error_worker,args=(str(tmp_path),),nprocs=2)
    records=[json.loads((tmp_path/f'result-rank{rank}.json').read_text()) for rank in range(2)]
    assert records[0]==records[1]
    assert records[0]['none']['ok']
    assert not records[0]['read']['ok'] and not records[0]['prepare']['ok']
