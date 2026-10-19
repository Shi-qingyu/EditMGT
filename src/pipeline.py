# Copyright 2025 EditMGT Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from termcolor import cprint
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import VQModel
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from transformers import AutoModel, AutoTokenizer
from src.scheduler import Scheduler
from src.local_utils import smooth_local_scores, tokens_to_attn, rescale_scores
from src.transformer import Transformer2DModel, get_text_encoder_length
from src.dataset_utils import tokenize_prompt, encode_prompt, process_image
from src.v2_utils import prepare_cond_token
from src.attention_utils import extract_image_to_text_attention, num_attention_layers
from src.reference_cache import ReferenceKVCache, ConditionKVCache
from src.sparse_utils import expand_edit_mask_for_halo

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

def _prepare_latent_image_ids(height, width, device, dtype, up_sample=False):
    if up_sample:
        height = 2 * height
        width = 2 * width
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pilimage_to_token(pipe, image):
    latents = pipe.vqvae.encode(image.to(dtype=pipe.vqvae.dtype, device=pipe._execution_device)).latents
    latents_bsz, channels, latents_height, latents_width = latents.shape
    latents = pipe.vqvae.quantize(latents)[2][2].reshape(latents_bsz, latents_height, latents_width)
    return latents


def _expand_image_batch(values, batch_size, num_images_per_prompt, name):
    """Broadcast a shared image/mask or repeat one per prompt."""
    if values.shape[0] == batch_size:
        return values
    if values.shape[0] == 1:
        return values.expand(batch_size, *values.shape[1:])
    if values.shape[0] * num_images_per_prompt == batch_size:
        return values.repeat_interleave(num_images_per_prompt, dim=0)
    raise ValueError(f"{name} batch {values.shape[0]} does not match output batch {batch_size}")


class Pipeline(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: Transformer2DModel
    scheduler: Scheduler
    tokenizer_t5: AutoTokenizer
    text_encoder_t5: AutoModel

    model_cpu_offload_seq = "text_encoder->text_encoder_t5->transformer->vqvae"

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: Transformer2DModel,
        scheduler: Scheduler,
        tokenizer_t5: AutoTokenizer,
        text_encoder_t5: AutoModel,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            tokenizer_t5=tokenizer_t5,
            text_encoder_t5=text_encoder_t5,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor,
                                                  do_normalize=False, do_convert_rgb=True)

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
            do_resize=True,
        )

        # [1,2] is to capture the weight of the transformer block with idx 1 and 2
        self.attention_enable_blocks = None 
        # Local attention threshold [0, 1]
        self.local_guidance=0
        
        self.current_attention_list = {}
        self.selected_mask_token = {}
        self.local_scores_list = {}
        self.local_query_text = None
 
    def safe_decode_latents(
        self, latents, batch_size, height, width, output_type,
        preserve_image=None, pixel_edit_mask=None,
    ):
        decode_latents = latents.clone()
        mask_token_id = self.scheduler.config.mask_token_id
        # Replace the mask token with 0 (pure black)
        decode_latents[decode_latents == mask_token_id] = 0
        
        needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast
        if needs_upcasting:
            self.vqvae.float()
        
        decoded = self.vqvae.decode(
            decode_latents,
            force_not_quantize=True,
            shape=(
                batch_size,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
                self.vqvae.config.latent_channels,
            ),
        ).sample.clip(0, 1)
        if preserve_image is not None:
            if pixel_edit_mask is None:
                raise ValueError("Pixel preservation requires the original pixel edit mask")
            # Keep source RGB in float32 even when the VQ model uses BF16.
            # Unchanged VQ tokens alone do not preserve source pixels exactly.
            decoded = torch.where(
                pixel_edit_mask.to(device=decoded.device, dtype=torch.bool),
                decoded.float(), preserve_image.to(device=decoded.device, dtype=torch.float32),
            )
        decoded = self.image_processor.postprocess(decoded, output_type)
        
        if needs_upcasting:
            self.vqvae.half()
        
        return decoded
        
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 12,
        guidance_scale: float = 10.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_encoder_hidden_states: Optional[torch.Tensor] = None,
        output_type="pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        micro_conditioning_aesthetic_score: int = 6,
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
        mask_image: PipelineImageInput = None,  # inpaint
        return_target_step_images: List[int] = None,   # See attention and intermdiate image: Which steps to decode
        reference_image: PipelineImageInput = None,
        reference_strength: float = 1,
        lora_part_enable: Optional[bool] = False,       
        lora_scale: Optional[float] = None,           
        edit_only_diffusion: bool = False,
        return_attention: bool = False,
        attention_layers: Optional[List[int]] = None,
        attention_keywords: Optional[List[str]] = None,
        sparse_target: Optional[bool] = None,
        sparse_target_halo: Optional[int] = 0,
        use_reference_kv_cache: bool = True,
        reference_kv_cache_threshold: float = 0.3,
        use_condition_kv_cache: Optional[bool] = None,
        inference_shift: Optional[float] = None,
        preserve_unmasked_pixels: bool = True,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            sparse_target (`bool`, *optional*):
                Gather edit-mask target tokens plus halo after the stem. This enables
                edit-only diffusion and preserves source tokens outside the generation mask.
                For supervise_target_halo checkpoints the generation mask includes halo.
                Defaults to the checkpoint setting when an edit mask is provided.
            sparse_target_halo (`int`, *optional*):
                Halo radius on the Transformer grid. Defaults to 0 (no dilation),
                including for older checkpoints. Explicit None uses the saved radius.
            preserve_unmasked_pixels (`bool`, *optional*, defaults to `True`):
                For edit-only diffusion, composite decoded RGB inside the ORIGINAL
                pixel mask onto the source image. Halo tokens still provide context
                and denoise normally, but cannot change the output outside this mask.
                This applies to decoded outputs, including intermediate previews;
                output_type="latent" returns the generation latents without compositing.
            inference_shift (`float`, *optional*):
                Compatibility argument; only 1.0 is accepted. All generations,
                previews and benchmarks use the unshifted cosine mask schedule.
            use_reference_kv_cache (`bool`, *optional*, defaults to `True`):
                Capture full per-layer reference K/V once at the low-noise
                transition, then pass only text and target through the model.
                The cache is local to this generation and never stores target K/V.
            reference_kv_cache_threshold (`float`, *optional*, defaults to `0.3`):
                Start capture when the scheduled remaining editable mask fraction
                is at most this value (0..1). This uses the cosine
                schedule before each forward, without reading GPU token counts.
                Higher thresholds reuse more steps. Zero disables reuse.
                Cache-conditioned checkpoints cannot capture before their saved
                low-noise threshold; smaller values may delay cache capture.
            use_condition_kv_cache (`bool`, *optional*):
                Cache both text and complete reference for cache_conditioning
                checkpoints. Defaults to use_reference_kv_cache for those models.
                False disables reuse while retaining trained masks/modulation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 16):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.IntTensor`, *optional*):
                Pre-generated tokens representing latent vectors in `self.vqvae`, to be used as inputs for image
                gneration. If not provided, the starting latents will be completely masked.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                pooled and projected final hidden states.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated penultimate hidden states from the text encoder providing additional text conditioning.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_encoder_hidden_states (`torch.Tensor`, *optional*):
                Analogous to `encoder_hidden_states` for the positive prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            micro_conditioning_aesthetic_score (`int`, *optional*, defaults to 6):
                The targeted aesthetic score according to the laion aesthetic classifier. See
                https://laion.ai/blog/laion-aesthetics/ and the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            micro_conditioning_crop_coord (`Tuple[int]`, *optional*, defaults to (0, 0)):
                The targeted height, width crop coordinates. See the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            temperature (`Union[int, Tuple[int, int], List[int]]`, *optional*, defaults to (2, 0)):
                Configures the temperature scheduler on `self.scheduler` see `Scheduler#set_timesteps`.
        Examples:

        Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images.
        """

        previous_attention_blocks = self.attention_enable_blocks
        if return_attention:
            if prompt is None:
                raise ValueError("return_attention=True requires the original prompt text")
            self.attention_enable_blocks = (
                list(attention_layers) if attention_layers is not None
                else list(range(num_attention_layers(self.transformer)))
            )
        if self.attention_enable_blocks is not None:
            self.transformer.register_attention_hooks(self.attention_enable_blocks)
        self.current_attention_list = {}
        self.selected_mask_token = {}
        self.local_scores_list = {}
        self.last_attention_analysis = None
        self.last_generation_mask = None
        self.last_edit_region_info = None

        if sparse_target is None:
            sparse_target = self.transformer.config.sparse_target and mask_image is not None
        if sparse_target:
            edit_only_diffusion = True
        if sparse_target_halo is not None and (not isinstance(sparse_target_halo, int) or sparse_target_halo < 0):
            raise ValueError("sparse_target_halo must be a non-negative integer")
        if not 0.0 <= reference_kv_cache_threshold <= 1.0:
            raise ValueError("reference_kv_cache_threshold must be between 0 and 1")
        cache_conditioning = self.transformer.config.cache_conditioning
        if use_condition_kv_cache and not cache_conditioning:
            raise ValueError("Joint text/reference caching requires a cache_conditioning checkpoint")
        use_condition_kv_cache = (cache_conditioning and use_reference_kv_cache
                                  if use_condition_kv_cache is None else use_condition_kv_cache)

        if (prompt_embeds is not None and encoder_hidden_states is None) or (
            prompt_embeds is None and encoder_hidden_states is not None
        ):
            raise ValueError("pass either both `prompt_embeds` and `encoder_hidden_states` or neither")

        if edit_only_diffusion and (reference_image is None or mask_image is None):
            raise ValueError(
                "edit_only_diffusion=True requires both reference_image (source) and mask_image (1=edit)"
            )
        if sparse_target and not self.transformer.config.sparse_target:
            self.transformer.enable_sparse_target()

        if (negative_prompt_embeds is not None and negative_encoder_hidden_states is None) or (
            negative_prompt_embeds is None and negative_encoder_hidden_states is not None
        ):
            raise ValueError(
                "pass either both `negatve_prompt_embeds` and `negative_encoder_hidden_states` or neither"
            )

        if (prompt is None and prompt_embeds is None) or (prompt is not None and prompt_embeds is not None):
            raise ValueError("pass only one of `prompt` or `prompt_embeds`")

        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        if height is None:
            height = self.transformer.config.sample_size * self.vae_scale_factor

        if width is None:
            width = self.transformer.config.sample_size * self.vae_scale_factor

        if prompt_embeds is None:
            encoder_hidden_states, prompt_embeds = encode_prompt(
                    [self.text_encoder, self.text_encoder_t5] if self.text_encoder_t5 is not None else self.text_encoder,
                    tokenize_prompt([self.tokenizer, self.tokenizer_t5] if self.tokenizer_t5 is not None else self.tokenizer, prompt, 
                                    self.transformer.text_encoder_architecture, device=self._execution_device), 
                    self.transformer.text_encoder_architecture
                )

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)

                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]

                negative_encoder_hidden_states, negative_prompt_embeds = encode_prompt(
                    [self.text_encoder, self.text_encoder_t5] if self.text_encoder_t5 is not None else self.text_encoder,
                    tokenize_prompt([self.tokenizer, self.tokenizer_t5] if self.tokenizer_t5 is not None else self.tokenizer, negative_prompt, 
                                    self.transformer.text_encoder_architecture, device=self._execution_device), 
                    self.transformer.text_encoder_architecture
                )        

            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1)
            negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)
            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])

        # Note that the micro conditionings _do_ flip the order of width, height for the original size
        # and the crop coordinates. This is how it was done in the original code base
        micro_conds = torch.tensor(
            [
                width,
                height,
                micro_conditioning_crop_coord[0],
                micro_conditioning_crop_coord[1],
                micro_conditioning_aesthetic_score,
            ],
            device=self._execution_device,
            dtype=encoder_hidden_states.dtype,
        )
        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(2 * batch_size if guidance_scale > 1.0 else batch_size, -1)

        self.scheduler.set_timesteps(
            num_inference_steps, temperature, self._execution_device, inference_shift=inference_shift,
        )
        mask_ratios = self.scheduler.mask_ratios
        self.last_inference_schedule_info = {
            "inference_shift": self.scheduler.inference_shift,
            "masking_schedule": self.scheduler.config.masking_schedule,
            "mask_ratios": list(mask_ratios),  # N forwards plus the terminal clean state
            "time_conditioning": self.transformer.config.time_conditioning,
            "legacy_conditioning_timesteps": (
                list(self.scheduler.conditioning_timesteps)
                if self.transformer.config.time_conditioning == "legacy" else None
            ),
        }

        starting_mask_ratio = 1.0

        token_edit_mask = None
        target_region_mask = None
        pixel_edit_mask = None
        source_pixel_values = None
        composite_source = bool(edit_only_diffusion and preserve_unmasked_pixels)
        if mask_image is not None:
            token_edit_mask = self.mask_processor.preprocess(
                mask_image, height // self.vae_scale_factor, width // self.vae_scale_factor
            )
            if composite_source:
                # Retain the annotation at image resolution, independently of
                # VQ/Transformer-grid projection and halo dilation.
                pixel_edit_mask = _expand_image_batch(
                    self.mask_processor.preprocess(mask_image, height, width).bool(),
                    batch_size, num_images_per_prompt, "Pixel edit mask",
                )

        shape = (batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = torch.full(
            shape, self.scheduler.config.mask_token_id, dtype=torch.long, device=self._execution_device
        )  # 1024: torch.Size([bs, 64, 64])  512: torch.Size([bs, 32, 32])
        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order

        if reference_image is not None:
            if edit_only_diffusion:
                # Full-frame preprocessing keeps a supplied pixel mask aligned.
                # Reuse these pixels for VQ encoding and the final composition;
                # source RGB must not pass through a lossy encode/decode cycle.
                source_pixel_values = self.image_processor.preprocess(
                    reference_image, height=height, width=width,
                ).to(device=self._execution_device, dtype=torch.float32)
                reference_image = source_pixel_values.to(dtype=self.vqvae.dtype)
            elif isinstance(reference_image, list):
                reference_image = [process_image(img, size=height)['image'].to(dtype=self.vqvae.dtype, device=self._execution_device).unsqueeze(0) for img in reference_image]
                reference_image = torch.cat(reference_image, dim=0)
            else:
                reference_image = process_image(reference_image, size=height)['image'].to(dtype=self.vqvae.dtype, device=self._execution_device) 
                reference_image = reference_image.unsqueeze(0)

            reference_image_hidden_states = prepare_cond_token(split_vae_encode=None, pixel_values=reference_image, vq_model=self.vqvae)
            reference_image_hidden_states = reference_image_hidden_states.reshape(-1, height // self.vae_scale_factor, width // self.vae_scale_factor) # 处理和训练保持了一致
            if reference_image_hidden_states.shape[0] == 1:
                reference_image_hidden_states = reference_image_hidden_states.expand(batch_size, -1, -1)
            elif reference_image_hidden_states.shape[0] * num_images_per_prompt == batch_size:
                reference_image_hidden_states = reference_image_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            if composite_source:
                # Expand RGB only AFTER encoding: a shared reference is still
                # VQ-encoded once, regardless of the number of output images.
                source_pixel_values = _expand_image_batch(
                    source_pixel_values, batch_size, num_images_per_prompt, "Source image",
                )

            if mask_image is not None:
                source_tokens = reference_image_hidden_states.clone()
                latents = source_tokens.clone()
                token_edit_mask = token_edit_mask.reshape(
                    token_edit_mask.shape[0], latents.shape[-2], latents.shape[-1]
                ).bool().to(latents.device)
                if token_edit_mask.shape[0] == 1 and latents.shape[0] > 1:
                    token_edit_mask = token_edit_mask.expand(latents.shape[0], -1, -1)
                if token_edit_mask.shape != latents.shape:
                    raise ValueError(
                        f"Mask token grid {tuple(token_edit_mask.shape)} does not match "
                        f"source token grid {tuple(latents.shape)}"
                    )
                target_region_mask = token_edit_mask
                if sparse_target and self.transformer.config.supervise_target_halo:
                    grid_size = (tuple(size // 2 for size in latents.shape[-2:])
                                 if self.transformer.down_block.downsample is not None else tuple(latents.shape[-2:]))
                    halo = (self.transformer.config.sparse_target_halo
                            if sparse_target_halo is None else sparse_target_halo)
                    token_edit_mask = expand_edit_mask_for_halo(
                        target_region_mask, tuple(latents.shape[-2:]), grid_size, halo,
                    )
                latents[token_edit_mask] = self.scheduler.config.mask_token_id
                starting_mask_ratio = token_edit_mask.float().mean()
            
        if reference_image is not None:
            for name, module in self.transformer.named_modules():
                if not name.endswith(".attn"):
                    continue
                if reference_strength != 1:
                    module.reference_strength_factor = torch.ones(1, 1) * reference_strength
                elif hasattr(module, "reference_strength_factor"):
                    delattr(module, "reference_strength_factor")

        intermediate_latents = {}
        # Static IDs/masks and the full CFG reference input are prepared once.
        # Cached forwards omit this reference input entirely; source VQ tokens
        # stay available to the sampler for preservation outside the generation mask.
        img_ids = _prepare_latent_image_ids(
            latents.shape[-2], latents.shape[-1], latents.device, latents.dtype,
            up_sample=self.transformer.down_block.downsample is None,
        )
        text_length = (encoder_hidden_states.shape[1] if '_' not in self.transformer.text_encoder_architecture
                       else get_text_encoder_length(self.transformer.text_encoder_architecture, return_main=True))
        txt_ids = torch.zeros(text_length, 3, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        model_edit_mask = None
        if sparse_target:
            # Gather uses the original ROI and adds its halo exactly once;
            # the sampler uses the expanded generation mask when trained so.
            model_edit_mask = torch.cat([target_region_mask] * 2) if guidance_scale > 1.0 else target_region_mask
        reference_input, reference_image_ids = None, None
        if reference_image is not None:
            reference_input = (torch.cat([reference_image_hidden_states] * 2)
                               if guidance_scale > 1.0 else reference_image_hidden_states)
            reference_image_ids = _prepare_latent_image_ids(
                reference_input.shape[-2], reference_input.shape[-1], reference_input.device, reference_input.dtype,
                up_sample=self.transformer.down_block.downsample is None,
            )

        reference_kv_cache = None
        condition_kv_cache = None
        cache_capture_step = None
        capture_threshold = (min(reference_kv_cache_threshold, self.transformer.config.condition_cache_threshold)
                             if cache_conditioning else reference_kv_cache_threshold)
        # Attention/local-guidance reports require live text queries. Recompute
        # the correctly masked prefix for these calls, instead of fabricating
        # attention maps for query rows absent from a cache-only forward.
        live_text_queries = (return_attention or self.local_guidance > 0 or return_target_step_images is not None
                             or any(getattr(p, "capture_attention", False)
                                    for p in self.transformer.attn_processors.values()))
        cache_enabled = ((use_condition_kv_cache and not live_text_queries) if cache_conditioning
                         else use_reference_kv_cache)
        if cache_enabled and reference_image is not None and capture_threshold > 0:
            # Never spend memory capturing a cache if no later step can reuse it.
            cache_capture_step = next((
                step for step in range(max(0, len(self.scheduler.timesteps) - 1))
                if mask_ratios[step] <= capture_threshold
            ), None)
            if cache_capture_step is not None:
                if cache_conditioning:
                    condition_kv_cache = ConditionKVCache()
                else:
                    reference_kv_cache = ReferenceKVCache()
        self.last_reference_cache_info = {
            "enabled": reference_kv_cache is not None,
            "inference_shift": self.scheduler.inference_shift,
            "threshold": reference_kv_cache_threshold,
            "capture_step": cache_capture_step, "reuse_steps": [],
        }
        self.last_condition_cache_info = {
            "enabled": condition_kv_cache is not None,
            "inference_shift": self.scheduler.inference_shift,
            "trained_prefix_mask": cache_conditioning,
            "phase_threshold": self.transformer.config.condition_cache_threshold if cache_conditioning else None,
            "threshold": capture_threshold,
            "capture_step": cache_capture_step if condition_kv_cache is not None else None,
            "reuse_steps": [], "cached_branches": ["text", "reference"] if condition_kv_cache is not None else [],
            "disabled_for_live_text_queries": bool(cache_conditioning and live_text_queries),
        }
        if condition_kv_cache is not None:
            self.last_reference_cache_info = self.last_condition_cache_info

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            timesteps_iter = enumerate(self.scheduler.timesteps)

            for i, timestep in timesteps_iter:
                if guidance_scale > 1.0:
                    model_input = torch.cat([latents] * 2)
                else:
                    model_input = latents

                step_mask_ratio = mask_ratios[i]
                # New checkpoints derive model time from step_mask_ratio.
                # Legacy checkpoints retain their original step-ID time units.
                model_timestep = None
                if self.transformer.config.time_conditioning == "legacy":
                    model_timestep = torch.full(
                        (model_input.shape[0],), self.scheduler.conditioning_timesteps[i],
                        device=model_input.device,
                        dtype=torch.long,
                    )

                # image edit
                if reference_image is not None:
                    step_cache = (reference_kv_cache if cache_capture_step is not None
                                  and i >= cache_capture_step else None)
                    step_condition_cache = (condition_kv_cache if cache_capture_step is not None
                                            and i >= cache_capture_step else None)
                    reuse_reference = step_cache is not None and step_cache.ready
                    reuse_condition = step_condition_cache is not None and step_condition_cache.ready
                    if reuse_reference:
                        self.last_reference_cache_info["reuse_steps"].append(i)
                    if reuse_condition:
                        self.last_condition_cache_info["reuse_steps"].append(i)
                    model_output = self.transformer(
                        hidden_states = model_input,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        encoder_hidden_states=None if reuse_condition else encoder_hidden_states,
                        img_ids = img_ids,
                        txt_ids = None if reuse_condition else txt_ids,
                        timestep=model_timestep,
                        reference_image_hidden_states=None if (reuse_reference or reuse_condition) else reference_input,
                        reference_image_ids=None if (reuse_reference or reuse_condition) else reference_image_ids,
                        reference_kv_cache=step_cache,
                        condition_kv_cache=step_condition_cache,
                        mask_ratio=step_mask_ratio,
                        joint_attention_kwargs={"scale": lora_scale},
                        lora_part_enable=lora_part_enable,
                        sparse_target=sparse_target,
                        sparse_target_halo=sparse_target_halo,
                        target_edit_mask=model_edit_mask,
                    )
                    if step_cache is not None and not reuse_reference:
                        self.last_reference_cache_info.update({
                            "reference_length": step_cache.reference_length,
                            "num_layers": step_cache.num_layers,
                        })
                    if step_condition_cache is not None and not reuse_condition:
                        self.last_condition_cache_info.update({
                            "text_length": step_condition_cache.text_length,
                            "reference_length": step_condition_cache.reference_length,
                            "num_layers": step_condition_cache.num_layers,
                        })
                else:
                    # text to image
                    model_output = self.transformer(
                        hidden_states = model_input,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        encoder_hidden_states=encoder_hidden_states,
                        img_ids = img_ids,
                        txt_ids = txt_ids,
                        timestep=model_timestep,
                        sparse_target=False,
                        mask_ratio=step_mask_ratio,
                    )

                if guidance_scale > 1.0:
                    uncond_logits, cond_logits = model_output.chunk(2)  # uncond comes before concat
                    model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)

                if self.local_guidance > 0 or return_target_step_images is not None:
                    current_attention, local_stacked, _ = tokens_to_attn(
                        pipe=self, global_enable=False, prompt=prompt
                    )
                    local_scores = rescale_scores(local_stacked) # [bs, seq]
                    local_scores = smooth_local_scores(local_scores, method='adaptive', strength=1.0) # smoothen
                else:
                    current_attention, local_scores = None, None

                if return_attention and i == len(self.scheduler.timesteps) - 1:
                    self.last_attention_analysis = extract_image_to_text_attention(
                        transformer=self.transformer,
                        tokenizer=[self.tokenizer, self.tokenizer_t5]
                        if self.tokenizer_t5 is not None else self.tokenizer,
                        prompt=prompt,
                        layers=self.attention_enable_blocks,
                        keywords=attention_keywords,
                        guidance_enabled=guidance_scale > 1.0,
                    )

                latents = self.scheduler.step(
                    model_output=model_output,  # torch.Size([1, 8192, 32, 32])
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                    starting_mask_ratio=starting_mask_ratio,
                    # local
                    local_guidance=self.local_guidance,
                    local_scores=local_scores, # [batch, seq]
                    ref_latents=reference_image_hidden_states if reference_image is not None else None, # torch.Size([1, 32, 32])
                    edit_mask=token_edit_mask if edit_only_diffusion else None,
                    source_tokens=source_tokens if edit_only_diffusion else None,
                ).prev_sample   # torch.Size([1, 32, 32])

                if edit_only_diffusion:
                    latents = torch.where(token_edit_mask, latents, source_tokens)
                    if not torch.equal(latents[~token_edit_mask], source_tokens[~token_edit_mask]):
                        raise AssertionError("Preserve-region tokens changed during edit-only sampling")

                # See Attention: Decode intermediate images at target steps
                if return_target_step_images is not None:
                    step_idx = i
                    if step_idx in return_target_step_images:
                        intermediate_latents[step_idx] = latents
                        self.current_attention_list[step_idx] = current_attention
                        self.selected_mask_token[step_idx] = self.scheduler.mask_selected
                        self.local_scores_list[step_idx] = local_scores

                if (i == len(self.scheduler.timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                )):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, timestep, latents)
        
        # Release potentially large per-layer K/V before VQ decoding. Processors
        # never own the cache, so it cannot leak into the next request or training.
        if reference_kv_cache is not None:
            reference_kv_cache.clear()
        if condition_kv_cache is not None:
            condition_kv_cache.clear()

        if edit_only_diffusion:
            # Export the actual token generation mask for reports, rather than
            # incorrectly showing only the smaller requested output region.
            self.last_generation_mask = token_edit_mask.detach().cpu()
            original_token_mask = target_region_mask.detach().cpu()
            self.last_edit_region_info = {
                "sparse_target_halo": (
                    self.transformer.config.sparse_target_halo if sparse_target_halo is None
                    else sparse_target_halo
                ) if sparse_target else 0,
                "vq_grid_size": list(token_edit_mask.shape[-2:]),
                "original_token_mask_ratio": original_token_mask.float().mean().item(),
                "generation_token_mask_ratio": self.last_generation_mask.float().mean().item(),
                "preserve_unmasked_pixels": bool(composite_source and output_type != "latent"),
                "output_compositing": (
                    "original_pixel_mask" if composite_source and output_type != "latent" else "none"
                ),
            }
        decode_preservation = {
            "preserve_image": source_pixel_values if composite_source else None,
            "pixel_edit_mask": pixel_edit_mask,
        }
        if output_type == "latent":
            output = latents
        else:
            output = self.safe_decode_latents(
                latents=latents, batch_size=batch_size, height=height, width=width,
                output_type=output_type, **decode_preservation,
            )
        if return_attention:
            self.transformer.clear_attention_hooks()
            self.attention_enable_blocks = previous_attention_blocks
        
        # See Attention: Decode intermediate images at target steps
        if return_target_step_images:
            intermediate_output = [self.safe_decode_latents(latents=l, batch_size=batch_size, height=height, 
                                                            width=width, output_type=output_type,
                                                            **decode_preservation) for l in intermediate_latents.values()]

        self.maybe_free_model_hooks()

        # See Attention: Return results with intermediate images if requested
        if return_target_step_images:
            if not return_dict:
                return (output, intermediate_output)
            return ImagePipelineOutput(output), ImagePipelineOutput(intermediate_output)
        else:
            if not return_dict:
                return (output,)
            return ImagePipelineOutput(output)
