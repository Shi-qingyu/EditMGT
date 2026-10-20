# Copyright 2025 EditMGT Team. All rights reserved.
import torch
from diffusers import VQModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, AutoModelForCausalLM, AutoTokenizer
from src.scheduler import Scheduler
from src.pipeline import Pipeline
from src.transformer import Transformer2DModel


def set_inference_cache_threshold(transformer, threshold=0.8):
    """Apply the requested prefix phase after loading full or LoRA weights."""
    transformer = getattr(transformer, "_orig_mod", transformer)
    if threshold is None or not getattr(transformer.config, "cache_conditioning", False):
        return
    if not hasattr(transformer, "_loaded_condition_cache_threshold"):
        transformer._loaded_condition_cache_threshold = transformer.config.condition_cache_threshold
    transformer.enable_condition_cache_training(threshold=threshold)


def init_edit_mgt(
    device,
    enable_bf16=False,
    base_model_path='WeiChow/EditMGT',
    transformer_subfolder='editmgt',
    transformer_path=None,
    condition_cache_threshold=0.8,
):
    text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
        base_model_path, subfolder="text_encoder"
    )
    tokenizer_clip = CLIPTokenizer.from_pretrained(
        base_model_path, subfolder="tokenizer"
    )
    model_gemma = AutoModelForCausalLM.from_pretrained(base_model_path, subfolder="llm_encoder", output_hidden_states=True)
    tokenizer_gemma = AutoTokenizer.from_pretrained(base_model_path, subfolder="llm_encoder")
    text_encoder = [text_encoder_clip,model_gemma]
    tokenizer = [tokenizer_clip,tokenizer_gemma]
    vq_model = VQModel.from_pretrained(
        base_model_path, subfolder="vqvae",
    ) 
    scheduler = Scheduler.from_pretrained(
        base_model_path, subfolder="scheduler",
    )
    transformer_source = transformer_path or base_model_path
    transformer_source_subfolder = None if transformer_path else transformer_subfolder
    if enable_bf16:
        text_encoder = [t.to(torch.bfloat16) for t in text_encoder]
        vq_model = vq_model.to(torch.bfloat16)
        # model = Transformer2DModel.from_pretrained(base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        model = Transformer2DModel.from_pretrained(
            transformer_source, subfolder=transformer_source_subfolder, torch_dtype=torch.bfloat16
        )
    else:
        # model = Transformer2DModel.from_pretrained(base_model_path, subfolder="transformer")
        model = Transformer2DModel.from_pretrained(
            transformer_source, subfolder=transformer_source_subfolder
        )
    
    # A checkpoint may still store the historical 0.3 phase. Set the runtime
    # phase before the pipeline caps capture by the model's configured phase.
    set_inference_cache_threshold(model, condition_cache_threshold)
    pipe = Pipeline(
        transformer=model,
        tokenizer=tokenizer[0],
        text_encoder=text_encoder[0],
        vqvae=vq_model,
        scheduler=scheduler,
        text_encoder_t5=text_encoder[1],
        tokenizer_t5=tokenizer[1]
    )
    pipe = pipe.to(device)

    return pipe
