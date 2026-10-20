"""Match inference noise conditioning to the time units used in training."""
import math

import pytest
import torch
from diffusers import VQModel
from PIL import Image

from src.pipeline import Pipeline
from src.scheduler import Scheduler
from test_sparse_target import deterministic_cpu, tiny_model


@pytest.mark.parametrize("time_mode", ["legacy", "mask_ratio"])
@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("num_steps", [1, 4, 8])
def test_pipeline_time_matches_training_noise_level(time_mode, with_reference, num_steps):
    model = tiny_model(sparse=False).eval()
    model.set_time_conditioning(time_mode)
    vq = VQModel(block_out_channels=(16,), latent_channels=4, vq_embed_dim=4,
                 num_vq_embeddings=31, norm_num_groups=4).eval()
    pipe = Pipeline(vq, None, None, model, Scheduler(mask_token_id=31), None, None)
    pipe.set_progress_bar_config(disable=True)
    embedded_times = []
    handle = model.time_text_embed.register_forward_pre_hook(
        lambda module, arguments: embedded_times.append(arguments[0].detach().clone())
    )
    prompt_embeds, text = torch.randn(1, 8), torch.randn(1, 3, 24)
    reference_args = dict(
        reference_image=[Image.new("RGB", (16, 16), (80, 140, 50))],
        mask_image=torch.ones(1, 1, 16, 16), edit_only_diffusion=True,
    ) if with_reference else {}
    try:
        pipe(
            prompt_embeds=prompt_embeds, encoder_hidden_states=text,
            negative_prompt_embeds=torch.zeros_like(prompt_embeds),
            negative_encoder_hidden_states=torch.zeros_like(text),
            height=16, width=16, num_inference_steps=num_steps, guidance_scale=2.0,
            output_type="latent", use_reference_kv_cache=False, **reference_args,
        )
    finally:
        handle.remove()

    target_times = embedded_times[::2] if with_reference else embedded_times
    assert len(target_times) == num_steps
    for step, actual in enumerate(target_times):
        # train/train.py passes r*1000 in legacy mode and r in mask_ratio
        # mode. The Transformer multiplies either input by 1000 internally.
        ratio = math.cos(math.pi * step / (2 * num_steps))
        training_ratio = torch.full((2,), ratio, dtype=torch.float32)
        training_timestep = training_ratio * 1000 if time_mode == "legacy" else training_ratio
        torch.testing.assert_close(actual, training_timestep * 1000, rtol=1e-6, atol=0)
    if with_reference:
        assert len(embedded_times) == 2 * num_steps
        assert all(time.eq(0).all() for time in embedded_times[1::2])
    if time_mode == "legacy":
        torch.testing.assert_close(
            torch.tensor(pipe.last_inference_schedule_info["legacy_conditioning_timesteps"]),
            torch.stack(target_times)[:, 0] / 1000,
        )
    else:
        assert pipe.last_inference_schedule_info["legacy_conditioning_timesteps"] is None
