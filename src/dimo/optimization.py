"""Trainability and optimizer-update-counted schedules; no checkpoint key changes."""
import torch


FREEZE_DEFAULTS = {f'freeze_{role}_{part}': False
                   for role in ('student', 'auxiliary') for part in ('embedding', 'mlm_head')}


def configure_trainability(models, config):
    report = {}
    for role, model in models.items():
        frozen = []
        if role in ('teacher', 'ema'):
            model.requires_grad_(False)
        else:
            for part, name in [('embedding', 'embed'), ('mlm_head', 'mlm_layer')]:
                flag = config.get(f'freeze_{role}_{part}', False)
                if type(flag) is not bool:
                    raise ValueError(f'freeze_{role}_{part} must be boolean')
                if flag:
                    getattr(model, name).requires_grad_(False)
                    frozen.append(name)
        report[role] = dict(total_parameters=sum(p.numel() for p in model.parameters()),
                            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            frozen_modules=frozen)
    return report


def build_optimizers(models, config):
    kwargs = dict(betas=tuple(config['adam_betas']), eps=config['adam_epsilon'],
                  weight_decay=config['weight_decay'], foreach=False)
    return {role: torch.optim.AdamW([p for p in models[role].parameters() if p.requires_grad],
                                    lr=config[f'{role}_lr'], **kwargs)
            for role in ('student', 'auxiliary')}


def build_schedulers(optimizers, config):
    kind = config.get('lr_scheduler', 'constant')
    warmup = config.get('lr_warmup_steps', 0)
    if kind not in ('constant', 'constant_with_warmup'):
        raise ValueError('Expected constant or constant_with_warmup')
    if type(warmup) is not int or warmup < 0 or (kind == 'constant_with_warmup' and warmup < 1):
        raise ValueError('constant_with_warmup requires positive lr_warmup_steps')
    # LambdaLR initializes n=0. First optimizer step uses 0 LR, as in official
    # diffusers constant_with_warmup; after 100 scheduler steps LR is nominal.
    # Called exactly once AFTER each corresponding optimizer.step(), never
    # wrapped by Accelerator (which could multiply steps by world_size).
    def multiplier(n):
        return min(1.0, float(n) / warmup) if kind == 'constant_with_warmup' else 1.0
    return {role: torch.optim.lr_scheduler.LambdaLR(opt, multiplier) for role, opt in optimizers.items()}
