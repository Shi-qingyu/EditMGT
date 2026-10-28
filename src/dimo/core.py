"""Mask invariants and the upstream DiMO forward-KL logit gradient."""
import copy
import math
import torch
import torch.nn.functional as F
from src.sparse_utils import expand_edit_mask_for_halo


def generation_mask(original, model):
    c = model.config
    if c.supervise_target_halo and c.sparse_target_halo > 0:
        h, w = original.shape[-2:]
        grid = (h // 2, w // 2) if model.down_block.downsample is not None else (h, w)
        return expand_edit_mask_for_halo(original, (h, w), grid, c.sparse_target_halo)
    return original.bool()


def pseudo_mask(region, ratio):
    flat = region.flatten(1)
    if (~flat.any(1)).any():
        raise ValueError('Empty generation region')
    if ratio.shape != (len(region),) or not torch.isfinite(ratio).all() or ((ratio < 0) | (ratio > 1)).any():
        raise ValueError('Expected one finite mask ratio in [0,1] per sample')
    # Uniform exact-count sampling, matching SFT/upstream DiMO's mask counts.
    count = (flat.sum(1) * ratio).round().long().clamp_min(1)
    order = torch.rand_like(flat.float()).masked_fill(~flat, 2).argsort(1).argsort(1)
    q = (order < count[:, None]) & flat
    return q.reshape_as(region)


def masked_mean(values, mask):
    return ((values * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()


def clone_models(model):
    teacher = model.eval().requires_grad_(False)
    student = copy.deepcopy(model).train().requires_grad_(True)
    auxiliary = copy.deepcopy(model).train().requires_grad_(True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    return teacher, student, auxiliary, ema


def ema_decay_at_step(decay, global_step=None):
    """Official DiMO rampup; step is completed updates BEFORE this update."""
    if global_step is None:
        return decay
    if type(global_step) is not int or global_step < 0:
        raise ValueError('EMA global_step must be a nonnegative integer')
    return min(decay, (1 + global_step) / (10 + global_step))


@torch.no_grad()
def update_ema(ema, student, decay, global_step=None):
    rate = ema_decay_at_step(decay, global_step)
    for e, s in zip(ema.parameters(), student.parameters()):
        if global_step is None:
            e.lerp_(s, 1 - rate)
        else:
            # Match official train_DIMO_Meissonic.py, including operation order.
            e.detach().mul_(rate).add_(s.to(e.device), alpha=1 - rate)
    for e, s in zip(ema.buffers(), student.buffers()):
        e.copy_(s)
    return rate


def composite_pixels(decoded, source, original_pixel_mask):
    return torch.where(original_pixel_mask.bool(), decoded, source)


def fkl_surrogate(student_logits, teacher_logits, auxiliary_logits, q):
    """DiMO FKL logit gradient p_aux - p_teacher; normalized per edited sample.

    Follows upstream train_DIMO_Meissonic.py FKL/MSE gradient injection,
    with explicit per-sample Q normalization for regional training.
    All logits use [B,H,W,V].
    """
    direction = (auxiliary_logits.float().softmax(-1) - teacher_logits.float().softmax(-1)).detach()
    z = student_logits.float()
    loss = 0.5 * (z - (z - direction).detach()).square().sum(-1)
    return masked_mean(loss, q)


def auxiliary_ce(logits, sampled_tokens, q, label_smoothing=0.1):
    loss = F.cross_entropy(logits.movedim(-1, 1).float(), sampled_tokens.detach(),
                           label_smoothing=label_smoothing, reduction='none')
    return masked_mean(loss, q)


def predict(model, tokens, ratio, conditions, noise_emb_perturb=0.0):
    mode = getattr(getattr(model, 'module', model), 'config').time_conditioning
    logits = model(hidden_states=tokens, timestep=ratio if mode == 'mask_ratio' else ratio * 1000,
                   mask_ratio=ratio, sparse_target=True, noise_emb_perturb=noise_emb_perturb, **conditions)
    return logits.movedim(1, -1)


def distillation_losses(teacher, student, auxiliary, source, original, region, mask_id,
                        conditions, empty_conditions, ratio, r_init=1.0, diagnostics=None,
                        teacher_cfg=4.0, noise_emb_perturb=0.0):
    if isinstance(teacher_cfg, bool) or not math.isfinite(teacher_cfg) or teacher_cfg < 1:
        raise ValueError('teacher_cfg must be finite and >=1')
    if (not torch.equal(conditions['target_edit_mask'], original)
            or not torch.equal(empty_conditions['target_edit_mask'], original)):
        raise ValueError('Transformer must receive the original edit mask')
    logits, sampled = generate_one_step(student, source, original, mask_id, conditions,
                                        r_init=r_init, return_logits=True, noise_emb_perturb=noise_emb_perturb)
    with torch.no_grad():
        q = pseudo_mask(region, ratio)
        corrupted = torch.where(q, mask_id, sampled)
        conditional = predict(teacher, corrupted, ratio, conditions)
        unconditional = predict(teacher, corrupted, ratio, empty_conditions)
        teacher_logits = unconditional + teacher_cfg * (conditional - unconditional)
        # Only the gradient-enabled Auxiliary forward owns its DDP reducer.
        auxiliary_logits = predict(getattr(auxiliary, 'module', auxiliary), corrupted, ratio, conditions)
    student_loss = fkl_surrogate(logits, teacher_logits, auxiliary_logits, q)
    if diagnostics is not None:
        diagnostics(logits, teacher_logits, auxiliary_logits, q, sampled, region)
    return student_loss, (corrupted, sampled, q, ratio)


def generate_one_step(student, source, original, mask_id, conditions, r_init=1.0, return_logits=False,
                      noise_emb_perturb=0.0):
    """Official random-code initialization within G, with source hard lock outside.

    r_init=1 masks all of G for editing. The r_init=0.5 ablation masks half
    of initially uniform VQ codes in G, rather than leaking source/GT into G.
    Gradients on the returned logits are enabled only in the Student phase;
    Auxiliary target generation and previews call this under no_grad.
    """
    if isinstance(r_init, bool) or not math.isfinite(r_init) or not 0 < r_init <= 1:
        raise ValueError('r_init must be finite and in (0,1]')
    if not torch.equal(conditions['target_edit_mask'], original):
        raise ValueError('Transformer must receive the original edit mask')
    bare_student = getattr(student, 'module', student)
    region = generation_mask(original, bare_student)
    ratio = torch.full((len(source),), r_init, device=source.device, dtype=torch.float32)
    with torch.no_grad():
        if r_init == 1:
            initial = torch.where(region, mask_id, source)
        else:
            random_codes = torch.randint(bare_student.config.codebook_size, source.shape, device=source.device)
            initial = torch.where(region, random_codes, source)
            initial = torch.where(pseudo_mask(region, ratio), mask_id, initial)
    logits = predict(student, initial, ratio, conditions, noise_emb_perturb=noise_emb_perturb)
    with torch.no_grad():
        sampled = torch.distributions.Categorical(logits=logits.float()).sample()
        sampled = torch.where(region, sampled, source)
    return (logits, sampled) if return_logits else sampled
