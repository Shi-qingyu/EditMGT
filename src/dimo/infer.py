"""Public Region-DiMO categorical one-step inference (Student/EMA CFG=1)."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.dimo.checkpoint import verify_checkpoint, CHECKPOINT_KIND, UPDATE_RULE
from src.dimo.core import generate_one_step
from src.dimo.origin import load_strict, sha256
from src.mask_utils import normalize_edit_mask_image, resize_binary_edit_mask


def load_role(checkpoint, role):
    """Load full local v3 checkpoint or an explicit inference export, strictly."""
    if role not in ('student','ema'):raise ValueError('Choose student or ema')
    checkpoint=Path(checkpoint).resolve()
    if (checkpoint/'export.json').is_file():
        metadata=json.loads((checkpoint/'export.json').read_text())
        if metadata['model']!=role:raise ValueError('Export role differs from requested model')
        for name,digest in metadata['checksums'].items():
            if Path(name).name!=name or sha256(checkpoint/name)!=digest:raise ValueError('Corrupt inference export')
        return load_strict(checkpoint),metadata
    header=verify_checkpoint(checkpoint)
    # Teacher was pinned on initialization, including runtime Halo; no SFT
    # weights or SFT optimizer are loaded during inference.
    origin_path=checkpoint.parent/'teacher_origin.json'
    origin=json.loads(origin_path.read_text())
    teacher=checkpoint.parent/'teacher'
    if sha256(teacher/'config.json')!=origin['pinned_teacher_checksums']['config.json']:
        raise ValueError('Pinned Teacher configuration changed')
    from src.transformer import Transformer2DModel
    model=Transformer2DModel.from_config(json.loads((teacher/'config.json').read_text()))
    state=torch.load(checkpoint/'training.pt',map_location='cpu',weights_only=False,mmap=True)
    if state.get('kind')!=CHECKPOINT_KIND or state.get('update_rule')!=UPDATE_RULE:
        raise ValueError('Incompatible training state')
    model.load_state_dict(state['models'][role],strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite inference weights')
    metadata=dict(model=role,checkpoint=str(checkpoint),global_step=header['global_step'],
                  config=header['config'],teacher_origin=origin,
                  training_checksums=header['checksums'],source_metadata=header.get('metadata',{}))
    return model,metadata


def validate_base_components(base):
    base=Path(base).resolve()
    for name in ('editmgt','text_encoder','tokenizer','llm_encoder','vqvae','scheduler'):
        if not (base/name).is_dir():raise FileNotFoundError(f'Missing explicit local base component: {base/name}')
    # Prevent upstream CLIP_Gemma2's remote Gemma fallback by checking the local
    # component before invoking its loader. No implicit model substitutions.
    for name in ('llm_encoder','text_encoder','vqvae'):
        if not (base/name/'config.json').is_file():raise FileNotFoundError(base/name/'config.json')
    return base


def load_components(base,device,precision):
    from src.v2_model import init_base_model,init_text_encoder
    base=validate_base_components(base)
    encoders,tokenizers=init_text_encoder('CLIP_Gemma2',str(base))
    vq,scheduler=init_base_model(str(base))
    for component in encoders:
        component.eval().requires_grad_(False).to(device,dtype=torch.bfloat16 if precision=='bf16' else torch.float32)
    vq.eval().requires_grad_(False).to(device,dtype=torch.float32)
    return encoders,tokenizers,vq,scheduler


def autocast(device,precision):
    if precision not in ('no','bf16'):raise ValueError('precision must be no or bf16')
    return torch.autocast(device.type,dtype=torch.bfloat16) if precision=='bf16' else nullcontext()


def resolve_halo_profile(profile, halo=None):
    """Keep the saved radius/supervision pair unless CLI explicitly overrides it."""
    try:
        saved_halo = profile['sparse_target_halo']
        saved_supervise = profile['supervise_target_halo']
    except KeyError as error:
        raise ValueError('Checkpoint must explicitly record both Halo settings') from error
    if type(saved_halo) is not int or saved_halo < 0 or type(saved_supervise) is not bool:
        raise ValueError('Invalid checkpoint Halo radius or supervision flag')
    if halo is not None and (type(halo) is not int or halo < 0):
        raise ValueError('halo must be a nonnegative integer')
    source = 'checkpoint' if halo is None else 'cli'
    return dict(sparse_target_halo=saved_halo if halo is None else halo,
                supervise_target_halo=saved_supervise if halo is None else bool(halo),
                halo_source=source, supervise_target_halo_source=source)


class OneStepEditor:
    def __init__(self,model,encoders,tokenizers,vq,scheduler,*,device,precision='no',
                 resolution=1024,halo=None,r_init=1.0,lora_part_enable=True,noise_emb_perturb=0.0):
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
        self.model=model.eval().requires_grad_(False).to(self.device)
        self.halo_profile=resolve_halo_profile(self.model.config,halo)
        if halo is not None:
            self.model.enable_sparse_target(halo=self.halo_profile['sparse_target_halo'],
                                          supervise_halo=self.halo_profile['supervise_target_halo'])
        self.model._dimo_lora_part_enable=lora_part_enable
        self.encoders,self.tokenizers,self.vq=encoders,tokenizers,vq
        for component in [*encoders,vq]:component.eval().requires_grad_(False)
        if vq.config.num_vq_embeddings!=model.config.codebook_size or scheduler.config.mask_token_id!=model.config.vocab_size-1:
            raise ValueError('VQ codebook / mask token differs from checkpoint')
        if type(resolution) is not int or resolution<16 or resolution%16:raise ValueError('Invalid explicit resolution')
        if isinstance(noise_emb_perturb,bool) or not 0<=noise_emb_perturb<1:raise ValueError('Invalid noise_emb_perturb')
        self.precision,self.resolution,self.r_init=precision,resolution,r_init
        self.noise_emb_perturb=noise_emb_perturb

    @torch.inference_mode()
    def edit(self,source,mask,prompt,*,seed=42,preserve_unmasked_pixels=False):
        from src.dataset_utils import tokenize_prompt,encode_prompt
        from src.v2_utils import prepare_cond_token
        from src.mask_utils import resize_edit_mask_to_tokens
        from src.pipeline import _prepare_latent_image_ids
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Nonempty prompt required')
        source=source.convert('RGB')
        native_size=source.size
        native_mask=resize_binary_edit_mask(normalize_edit_mask_image(mask),(source.height,source.width))
        size=(self.resolution,self.resolution)
        resized=source.resize(size,Image.Resampling.BILINEAR)
        resized_mask=resize_binary_edit_mask(native_mask,(self.resolution,self.resolution))
        if resized_mask.getbbox() is None:raise ValueError('Empty input mask')
        pixels=torch.from_numpy(np.array(resized,dtype=np.float32)/255).permute(2,0,1)[None].to(self.device)
        pixel_mask=torch.from_numpy(np.array(resized_mask)>0)[None,None].to(self.device)
        # Sampling uses Torch RNG, but is local to this request and restores the
        # caller's CPU/device RNG. No other visible GPUs are seeded/touched.
        devices=[self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:torch.cuda.default_generators[devices[0]].manual_seed(seed)
            tokens=prepare_cond_token(None,pixels,self.vq)
            side=int(tokens.shape[-1]**0.5)
            tokens=tokens.reshape(1,side,side)
            original=resize_edit_mask_to_tokens(pixel_mask,(side,side)).reshape_as(tokens)
            if not original.any():raise ValueError('Empty VQ mask')
            prompt_ids=tokenize_prompt(self.tokenizers,prompt,'CLIP_Gemma2',device=self.device)
            hidden,pooled=encode_prompt(self.encoders,prompt_ids,'CLIP_Gemma2')
            ids=_prepare_latent_image_ids(side,side,self.device,tokens.dtype,up_sample=self.model.down_block.downsample is None)
            conditions=dict(encoder_hidden_states=hidden,pooled_projections=pooled,
                micro_conds=torch.tensor([[self.resolution,self.resolution,0,0,6.0]],device=self.device),
                img_ids=ids,reference_image_ids=ids,txt_ids=torch.zeros(hidden.shape[1],3,device=self.device,dtype=tokens.dtype),
                reference_image_hidden_states=tokens,target_edit_mask=original,
                lora_part_enable=self.model._dimo_lora_part_enable)
            with autocast(self.device,self.precision):
                generated=generate_one_step(self.model,tokens,original,self.model.config.vocab_size-1,conditions,
                            r_init=self.r_init,noise_emb_perturb=self.noise_emb_perturb)
            b,h,w=generated.shape
            decoded=self.vq.decode(generated.flatten(),force_not_quantize=True,
                    shape=(b,h,w,self.vq.config.latent_channels)).sample.float().clamp(0,1)
            image=Image.fromarray((decoded[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
        image=image.resize(native_size,Image.Resampling.BICUBIC)
        # Explicit opt-in only, consistently recorded for comparisons.
        if preserve_unmasked_pixels:image=Image.composite(image,source,native_mask)
        return image,dict(seed=seed,steps=1,cfg=1.0,halo=self.model.config.sparse_target_halo,
            supervise_target_halo=self.model.config.supervise_target_halo,r_init=self.r_init,
            halo_source=self.halo_profile['halo_source'],
            supervise_target_halo_source=self.halo_profile['supervise_target_halo_source'],
            noise_emb_perturb=self.noise_emb_perturb,diagnostic_perturbation=self.noise_emb_perturb!=0,
            precision=self.precision,resolution=self.resolution,sampling='independent_categorical',
            output_compositing='original_pixel_mask_at_generation_and_native_resolution' if preserve_unmasked_pixels else 'none',
            preserve_unmasked_pixels=preserve_unmasked_pixels,direct_vq_decode=not preserve_unmasked_pixels,
            cache='disabled',inference_kv_cache=False,cache_conditioning=self.model.config.cache_conditioning,
            condition_cache_threshold=self.model.config.condition_cache_threshold,
            lora_part_enable=self.model._dimo_lora_part_enable)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','source','mask','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model',choices=['student','ema'],required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--base-model-dir',default=os.environ.get('BASE_MODEL_DIR'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--mixed-precision',choices=['no','bf16'])
    parser.add_argument('--resolution',type=int)
    parser.add_argument('--halo',type=int)
    parser.add_argument('--r-init',type=float,default=1.0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--noise-emb-perturb',type=float,default=0.0,help='Explicit diagnostic mode; normal inference is 0')
    parser.add_argument('--preserve-unmasked-pixels',action=argparse.BooleanOptionalAction,default=False)
    args=parser.parse_args()
    model,metadata=load_role(args.checkpoint,args.model)
    config=metadata['config']
    base=args.base_model_dir or config.get('base_model_dir')
    if not base:parser.error('Explicit --base-model-dir / BASE_MODEL_DIR required')
    device=torch.device(args.device)
    precision=args.mixed_precision or config['mixed_precision']
    components=load_components(base,device,precision)
    editor=OneStepEditor(model,*components,device=device,precision=precision,
        resolution=args.resolution or config['resolution'],halo=args.halo,r_init=args.r_init,
        lora_part_enable=config.get('lora_part_enable',True),noise_emb_perturb=args.noise_emb_perturb)
    with Image.open(args.source) as source,Image.open(args.mask) as mask:
        image,protocol=editor.edit(source,mask,args.prompt,seed=args.seed,preserve_unmasked_pixels=args.preserve_unmasked_pixels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    image.save(args.output)
    args.output.with_suffix(args.output.suffix+'.json').write_text(json.dumps(dict(protocol=protocol,model_origin=metadata,
        source=str(args.source.resolve()),mask=str(args.mask.resolve()),prompt=args.prompt),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
