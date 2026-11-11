"""Matched dense -> sparse -> cached sparse -> distilled benchmark adapter.

Only command-local runner/config factories are replaced. Official benchmark
sample selection, quality metrics, GPU sharding and resumable files are reused.
The dense control uses the SAME sparse-trained SFT weights, ROI conditioning,
generation region and source hard lock, but computes all target tokens.
The sparse+cache stage captures text/reference KV on the first forward and
keeps that cache frozen for every remaining forward.
"""
import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.multiprocessing as mp

import src.benchmark_utils as bu
from src.dimo.benchmark import StudentRunner
from src.dimo.core import generate_one_step, generation_mask
from src.mask_utils import normalize_edit_mask_image, resize_binary_edit_mask, resize_edit_mask_to_tokens
from src.sparse_utils import prepare_sparse_target_layout


STAGES = ('dense', 'sparse', 'sparse_cache', 'distill')
ORIGINAL_RUNNER = bu.InferenceRunner
ORIGINAL_CONFIG = bu.generation_config
TIME_SCOPE = ('GPU-synchronized pipeline: image-to-tensor/device copy, text encoding, source VQ, '
              'model forwards and VQ decoding; excludes native resize, annotation-mask preparation, '
              'mask-area bookkeeping, loading, warmup, image I/O and quality metrics')


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def protocol_config(args, benchmark=None):
    config = ORIGINAL_CONFIG(args, benchmark)
    config.update(model_protocol='matched_efficiency_ablation', ablation_stage=args.ablation_variant,
                  same_sft_weights_for_first_three=True, source_hard_lock=True,
                  target_execution='dense' if args.ablation_variant == 'dense' else 'sparse',
                  roi_shape_conditioning='original_ROI_bbox_in_all_variants',
                  mask_projection='shared_nearest_neighbor_with_centroid_fallback',
                  mask_area_experiment='natural_original_annotation_area; masks never resized to change task area',
                  benchmark_time_scope=TIME_SCOPE, vq_dtype='float32',
                  transformer_dtype='bfloat16' if args.bf16 else 'float32',
                  runtime_halo=args.sparse_target_halo, runtime_supervise_halo=bool(args.sparse_target_halo),
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.ablation_variant == 'distill':
        checkpoint = Path(args.checkpoint)
        manifest = checkpoint / ('export.json' if (checkpoint / 'export.json').is_file() else 'COMPLETE')
        config.update(distill_role=args.distill_role, distill_sampling='independent_categorical_one_step',
                      checkpoint_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    return config


class MatchedRunner:
    def __init__(self, args):
        self.args = args
        self.variant = args.ablation_variant
        self.is_distill = self.variant == 'distill'
        self.warmed_up = False
        self.shape_features = None
        self.token_mask = None
        self.shape_hook = None
        self.original_forward = None
        if self.is_distill:
            self.delegate = StudentRunner(args)
            self.model = self.delegate.editor.model
            vq = self.delegate.editor.vq
        else:
            self.delegate = ORIGINAL_RUNNER(args)
            self.model = self.delegate.pipe.transformer
            vq = self.delegate.pipe.vqvae
        self.model.enable_sparse_target(halo=args.sparse_target_halo,
                                        supervise_halo=bool(args.sparse_target_halo))
        # Both backends use exactly the same component precision. Native DiMO
        # otherwise keeps FP32 weights under autocast while Pipeline loads BF16.
        self.model.to(dtype=torch.bfloat16 if args.bf16 else torch.float32)
        vq.to(dtype=torch.float32)
        self.vq_scale = 2 ** (len(vq.config.block_out_channels) - 1)
        if not self.model.config.cache_conditioning:
            raise ValueError('Matched cache ablation requires a cache-conditioned SFT/DiMO checkpoint')
        if abs(self.model.config.condition_cache_threshold - args.condition_cache_threshold) > 1e-8:
            raise ValueError('Distilled checkpoint prefix phase differs from --condition-cache-threshold; '
                             'select the trained common threshold for both checkpoints')
        if not self.is_distill:
            self.original_mask_preprocess = self.delegate.pipe.mask_processor.preprocess
            self.delegate.pipe.mask_processor.preprocess = self.shared_mask_preprocess
        if self.variant == 'dense':
            self.original_forward = self.model.forward
            self.model.forward = self.dense_forward
            if self.model.sparse_shape_embed is not None:
                self.shape_hook = self.model.sparse_shape_embed.register_forward_pre_hook(self.roi_shape_input)
        self.metadata = dict(self.delegate.metadata)
        self.metadata.update(ablation_stage=self.variant, benchmark_time_scope=TIME_SCOPE,
                             vq_dtype='float32', transformer_dtype='bfloat16' if args.bf16 else 'float32',
                             runtime_halo=args.sparse_target_halo, runtime_supervise_halo=bool(args.sparse_target_halo))
        self.delegate.metadata = self.metadata

    def dense_forward(self, *args, **kwargs):
        # Keep Pipeline's sparse mask/halo path; only the Transformer executes
        # densely. ROI conditioning is retained by roi_shape_input below.
        kwargs['sparse_target'] = False
        return self.original_forward(*args, **kwargs)

    def roi_shape_input(self, module, inputs):
        if self.shape_features is None:
            raise RuntimeError('Dense ROI shape was not prepared for this image')
        features = self.shape_features.expand(inputs[0].shape[0], -1).to(inputs[0])
        return (features, *inputs[1:])

    def shared_mask_preprocess(self, image, height=None, width=None, **kwargs):
        if self.token_mask is None or (height, width) != tuple(self.token_mask.shape[-2:]):
            raise ValueError('Unexpected mask projection request in matched pipeline')
        return self.token_mask[:, None].float()

    def prepare_region(self, source, mask):
        """Shared analysis/preparation outside the timed pipeline, without GT."""
        native = resize_binary_edit_mask(normalize_edit_mask_image(mask), (source.height, source.width))
        resized = resize_binary_edit_mask(native, (self.args.resolution, self.args.resolution))
        pixels = torch.from_numpy((np.asarray(resized) > 0).copy())[None, None]
        side = self.args.resolution // self.vq_scale
        original = resize_edit_mask_to_tokens(pixels, (side, side)).reshape(1, side, side)
        grid = (side // 2, side // 2) if self.model.down_block.downsample is not None else (side, side)
        layout = prepare_sparse_target_layout(original, (side, side), grid, 1,
                                               torch.device('cpu'), self.args.sparse_target_halo)
        generated = generation_mask(original, self.model)
        self.token_mask = original
        if self.variant == 'dense' and self.model.sparse_shape_embed is not None:
            from diffusers.models.embeddings import get_timestep_embedding

            features = get_timestep_embedding(layout.shape.flatten(), 256, flip_sin_to_cos=True,
                                              downscale_freq_shift=0).reshape(1, 512)
            parameter = next(self.model.parameters())
            self.shape_features = features.to(device=parameter.device, dtype=parameter.dtype)
        full = grid[0] * grid[1]
        active = full if self.variant == 'dense' else int(layout.valid.sum().item())
        area = dict(stage=self.variant, steps=self.args.num_inference_steps, cfg=self.args.guidance_scale,
                    precision='bf16' if self.args.bf16 else 'float32', weight_source=str(self.args.checkpoint.resolve()),
                    original_mask_fraction=float((np.asarray(native) > 0).mean()),
                    generation_mask_fraction=float(generated.float().mean().item()),
                    active_target_fraction=active / full, active_target_tokens=active, full_target_tokens=full,
                    vq_mask_fraction=float(original.float().mean().item()), source_hard_lock=True,
                    original_mask_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                    vq_mask_sha256=tensor_hash(original), generation_mask_sha256=tensor_hash(generated),
                    halo=self.args.sparse_target_halo, vq_grid=[side, side], transformer_grid=list(grid),
                    mask_projection='nearest_neighbor_with_centroid_fallback',
                    cache_enabled=self.variant == 'sparse_cache', time_scope=TIME_SCOPE)
        return native, area

    def teacher_forward(self, source, mask, instruction, seed):
        pipe, args = self.delegate.pipe, self.args
        return pipe(prompt=[instruction], reference_image=[source], mask_image=mask,
                    height=args.resolution, width=args.resolution,
                    num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                    reference_strength=1., negative_prompt='',
                    generator=torch.Generator(device=pipe._execution_device).manual_seed(seed),
                    edit_only_diffusion=True, sparse_target=True, sparse_target_halo=args.sparse_target_halo,
                    preserve_unmasked_pixels=False, lora_part_enable=True,
                    use_reference_kv_cache=self.variant == 'sparse_cache',
                    use_condition_kv_cache=self.variant == 'sparse_cache',
                    reference_kv_cache_threshold=args.reference_kv_cache_threshold,
                    condition_kv_cache_reuse_steps=args.num_inference_steps - 1 if args.cache_first_step else None,
                    allow_early_condition_cache=args.cache_first_step).images[0].convert('RGB')

    def distill_forward(self, source, instruction, seed):
        from PIL import Image
        from src.dataset_utils import encode_prompt, tokenize_prompt
        from src.dimo.infer import autocast
        from src.pipeline import _prepare_latent_image_ids
        from src.v2_utils import prepare_cond_token

        editor = self.delegate.editor
        device = editor.device
        # Mask resizing and projection are already done by prepare_region, as
        # for Pipeline. Only the device transfer is inside the pipeline timer.
        original = self.token_mask.to(device=device)
        pixels = torch.from_numpy(np.array(source, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            source_tokens = prepare_cond_token(None, pixels, editor.vq).reshape_as(original)
            prompt_ids = tokenize_prompt(editor.tokenizers, instruction, 'CLIP_Gemma2', device=device)
            hidden, pooled = encode_prompt(editor.encoders, prompt_ids, 'CLIP_Gemma2')
            side = original.shape[-1]
            ids = _prepare_latent_image_ids(side, side, device, source_tokens.dtype,
                                            up_sample=self.model.down_block.downsample is None)
            conditions = dict(encoder_hidden_states=hidden, pooled_projections=pooled,
                micro_conds=torch.tensor([[self.args.resolution, self.args.resolution, 0, 0, 6.]], device=device),
                img_ids=ids, reference_image_ids=ids, txt_ids=torch.zeros(hidden.shape[1], 3, device=device,
                dtype=source_tokens.dtype), reference_image_hidden_states=source_tokens,
                target_edit_mask=original, lora_part_enable=self.model._dimo_lora_part_enable)
            with autocast(device, editor.precision):
                generated = generate_one_step(self.model, source_tokens, original,
                    self.model.config.vocab_size - 1, conditions, r_init=1., noise_emb_perturb=0.)
            decoded = editor.vq.decode(generated.flatten(), force_not_quantize=True,
                shape=(1, side, side, editor.vq.config.latent_channels)).sample.float().clamp(0, 1)
            return Image.fromarray((decoded[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))

    @torch.inference_mode()
    def generate(self, source, mask, instruction, seed):
        from PIL import Image

        native_mask, area = self.prepare_region(source, mask)
        native_size = source.size
        resized_source = source.convert('RGB').resize((self.args.resolution, self.args.resolution), Image.Resampling.BILINEAR)
        resized_mask = resize_binary_edit_mask(native_mask, (self.args.resolution, self.args.resolution))
        device = torch.device(self.args.device)
        def forward():
            if self.is_distill:
                return self.distill_forward(resized_source, instruction, seed)
            return self.teacher_forward(resized_source, resized_mask, instruction, seed)
        if not self.warmed_up:
            for _ in range(self.args.warmup):
                forward()
            self.warmed_up = True
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        image = forward()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        info = dict(seed=seed, inference_seconds=elapsed, ablation=area, execution=self.metadata,
                    sparse_target=self.variant != 'dense', benchmark_time_scope=TIME_SCOPE,
                    output_compositing='none', preserve_unmasked_pixels=False,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
        if not self.is_distill:
            info['condition_cache'] = self.delegate.pipe.last_condition_cache_info
            info['inference_schedule'] = self.delegate.pipe.last_inference_schedule_info
        if not self.is_distill:
            expected = area['generation_mask_fraction']
            actual = self.delegate.pipe.last_generation_mask
            if actual is None or abs(float(actual.float().mean()) - expected) > 1e-7:
                raise RuntimeError('Actual Pipeline generation mask differs from the shared mask protocol')
        return image.resize(native_size, Image.Resampling.BICUBIC), info

    def close(self):
        if self.shape_hook is not None:
            self.shape_hook.remove()
            self.shape_hook = None
        if self.original_forward is not None:
            self.model.forward = self.original_forward
            self.original_forward = None
        if not self.is_distill:
            self.delegate.pipe.mask_processor.preprocess = self.original_mask_preprocess
        self.model = None
        self.shape_features = None
        self.delegate.close()


@contextmanager
def backend_context(args):
    import src.benchmark_parallel as bp

    with patch.object(bu, 'InferenceRunner', MatchedRunner), patch.object(bu, 'generation_config', protocol_config), \
            patch.object(bp, '_spawn_workers', spawn_workers):
        yield


def benchmark_worker(rank, args, *worker_args):
    from src.benchmark_parallel import _benchmark_worker

    with backend_context(args):
        _benchmark_worker(rank, args, *worker_args)


def spawn_workers(args, run, shards, names, worker_dirs, devices, metric_devices):
    context = mp.spawn(benchmark_worker,
                       args=(args, run.root, run.info, shards, names, worker_dirs, devices, metric_devices),
                       nprocs=len(shards), join=False)
    try:
        while not context.join():
            pass
    except BaseException:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--benchmark', choices=('inter_edit', 'pie_bench'), required=True)
    parser.add_argument('--variant', choices=STAGES, required=True)
    parser.add_argument('--distill-role', choices=('student', 'ema'), default='ema')
    own, remaining = parser.parse_known_args(argv)
    if own.benchmark == 'inter_edit':
        from eval import benchmark_inter_edit as entry
    else:
        from eval import benchmark_pie_bench as entry
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        args = entry.parse_args()
    finally:
        sys.argv = previous
    args.ablation_variant, args.distill_role = own.variant, own.distill_role
    args.dimo_backend, args.dimo_r_init = own.distill_role, 1.0
    if args.vanilla or args.checkpoint is None or args.mask_mode != 'provided' or args.preserve_unmasked_pixels:
        parser.error('All variants require an explicit checkpoint, the same provided mask and direct VQ decode')
    if args.reference_strength != 1 or args.negative_prompt != '':
        parser.error('Use reference strength=1 and empty negative prompt')
    if own.variant == 'distill' and (args.num_inference_steps != 1 or args.guidance_scale != 1):
        parser.error('The distilled endpoint requires one step and CFG=1')
    if own.variant == 'sparse_cache':
        if args.num_inference_steps < 2 or args.disable_reference_kv_cache or args.disable_condition_kv_cache:
            parser.error('sparse_cache requires >=2 forwards with condition/reference caching enabled')
        # Capture once at step 0, then reuse without refreshing at phase changes.
        args.cache_first_step = True
    elif args.cache_first_step or not args.disable_reference_kv_cache or not args.disable_condition_kv_cache:
        parser.error('Disable both runtime caches for dense, sparse and one-step distill')
    return args, entry


def main():
    args, entry = parse_args(sys.argv[1:])
    with backend_context(args):
        return entry.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
