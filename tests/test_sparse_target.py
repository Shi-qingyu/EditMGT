import json
from unittest.mock import patch

import pytest
import torch
from diffusers.loaders import LoraLoaderMixin
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from src.sparse_utils import SparseTargetLayout, prepare_sparse_target_layout
from src.transformer import Transformer2DModel, load_lora_adapter
from src.v2_utils import apply_branch_rotary_emb
from diffusers.models.embeddings import apply_rotary_emb


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(12)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(sparse=True, downsample=True):
    return Transformer2DModel(
        num_layers=2, num_single_layers=2, attention_head_dim=12,
        num_attention_heads=2, joint_attention_dim=24, pooled_projection_dim=8,
        axes_dims_rope=(4, 4, 4), vocab_size=32, codebook_size=31,
        downsample=downsample, upsample=downsample, sparse_target=sparse,
        sparse_target_halo=0,
    )


def inputs(batch=2, downsample=True):
    h, w = (3, 4) if downsample else (6, 8)
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ids = torch.stack((torch.zeros_like(rows), rows, cols), dim=-1).reshape(-1, 3)
    mask = torch.zeros(batch, 6, 8, dtype=torch.bool)
    mask[0, 1, 3] = True
    if batch > 1:
        mask[1, 2:6, 4:8] = True
    return dict(
        hidden_states=torch.randint(0, 32, (batch, 6, 8)),
        reference_image_hidden_states=torch.randint(0, 31, (batch, 6, 8)),
        encoder_hidden_states=torch.randn(batch, 3, 24),
        pooled_projections=torch.randn(batch, 8), micro_conds=torch.randn(batch, 5),
        timestep=torch.rand(batch), img_ids=ids, reference_image_ids=ids.clone(),
        txt_ids=torch.zeros(3, 3), target_edit_mask=mask,
    )


def test_layout_projects_any_edit_and_keeps_per_sample_active_sets():
    mask = torch.zeros(3, 6, 8, dtype=torch.bool)
    mask[0, 1, 3] = True
    mask[1, 2:6, 4:8] = True
    layout = prepare_sparse_target_layout(mask, (6, 8), (3, 4), 3, mask.device, halo=0)
    assert layout.indices[0, layout.valid[0]].tolist() == [1]
    assert layout.indices[1, layout.valid[1]].tolist() == [6, 7, 10, 11]
    assert layout.valid.sum(1).tolist() == [1, 4, 0]
    assert layout.shape.tolist() == [[1, 1], [2, 2], [0, 0]]
    halo_layout = prepare_sparse_target_layout(mask[:1], (6, 8), (3, 4), 1, mask.device, halo=1)
    assert halo_layout.indices.tolist() == [[0, 1, 2, 4, 5, 6]]
    assert halo_layout.shape.tolist() == [[2, 3]]


def test_scatter_preserves_background_and_does_not_write_padding():
    args = inputs()
    layout = prepare_sparse_target_layout(args["target_edit_mask"], (6, 8), (3, 4), 2, torch.device("cpu"), 0)
    base = torch.randn(2, 12, 5, requires_grad=True)
    sparse = layout.gather(base).detach().requires_grad_()
    output = layout.scatter(sparse + 7, base)
    assert torch.equal(output[0, 0], base[0, 0])  # padding indices all point at zero
    assert torch.equal(output[0, 1], sparse[0, 0] + 7)
    output.sum().backward()
    assert sparse.grad[layout.valid].eq(1).all()
    assert sparse.grad[~layout.valid].eq(0).all()
    assert base.grad[0, 1].eq(0).all()
    assert base.grad[0, 0].eq(1).all()


def test_every_block_receives_sparse_target_full_reference_and_original_rope():
    model = tiny_model().eval()
    args = inputs()
    # Batched coordinates must not be replaced with the first sample's IDs.
    args["img_ids"] = args["img_ids"][None].expand(2, -1, -1).clone()
    args["img_ids"][1, :, 1:] += 10
    lengths, rope_ids = [], []
    handles = []
    for block in [*model.transformer_blocks, *model.single_transformer_blocks]:
        handles.append(block.register_forward_pre_hook(
            lambda module, positional, kwargs: lengths.append((
                kwargs["hidden_states"].shape[1], kwargs["reference_image_hidden_states"].shape[1]
            )), with_kwargs=True,
        ))
    handles.append(model.pos_embed.register_forward_pre_hook(lambda module, positional: rope_ids.append(positional[0].clone())))
    original_scatter = SparseTargetLayout.scatter
    scatter_count = []

    def check_scatter(layout, sparse, full):
        scatter_count.append(1)
        result = original_scatter(layout, sparse, full)
        active = torch.zeros(full.shape[:2], dtype=torch.bool)
        for b in range(full.shape[0]):
            active[b, layout.indices[b, layout.valid[b]]] = True
        assert torch.equal(result[~active], full[~active])
        return result

    with patch.object(SparseTargetLayout, "scatter", check_scatter), torch.no_grad():
        output = model(**args)
    for handle in handles:
        handle.remove()
    assert output.shape == (2, 31, 6, 8)
    assert lengths == [(4, 12), (4, 12), (7, 12), (7, 12)]
    assert len(scatter_count) == 1
    joint_ids = rope_ids[0].reshape(2, 7, 3)
    reference_ids = rope_ids[1].reshape(2, 12, 3)
    assert joint_ids[:, :3, 0].eq(0).all()
    assert joint_ids[0, 3].tolist() == [2, 0, 1]
    assert joint_ids[1, 3:, 0].eq(2).all()
    assert torch.equal(joint_ids[1, 3:, 1:], args["img_ids"][1, [6, 7, 10, 11], 1:])
    assert reference_ids[..., 0].eq(1).all()
    assert torch.equal(reference_ids[..., 1:], args["reference_image_ids"][None, :, 1:].expand(2, -1, -1))


@pytest.mark.parametrize("downsample", [True, False])
def test_ragged_batch_matches_independent_samples_with_reference_bias(downsample):
    model = tiny_model(downsample=downsample).eval()
    args = inputs(downsample=downsample)
    for block in [*model.transformer_blocks, *model.single_transformer_blocks]:
        block.attn.reference_strength_factor = torch.tensor([[1.25]])
    with torch.no_grad():
        batched = model(**args)
        for index in range(2):
            single = {
                key: value[index:index + 1] if key not in ("img_ids", "txt_ids", "reference_image_ids") else value
                for key, value in args.items()
            }
            torch.testing.assert_close(batched[index:index + 1], model(**single), rtol=2e-5, atol=2e-5)


def test_full_mask_sparse_equals_dense_and_batched_rope_equals_shared_rope():
    model = tiny_model().eval()
    args = inputs()
    args["target_edit_mask"] = torch.ones_like(args["target_edit_mask"])
    with torch.no_grad():
        torch.testing.assert_close(model(**args), model(**args, sparse_target=False))
    q = torch.randn(2, 2, 12, 12)
    rope = model.pos_embed(args["img_ids"])
    batched_rope = tuple(value[None].expand(2, -1, -1) for value in rope)
    torch.testing.assert_close(apply_branch_rotary_emb(q, batched_rope), apply_rotary_emb(q, rope), rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_sparse_backward_trains_shape_conditioning_and_all_blocks(checkpointing):
    model = tiny_model().train()
    if checkpointing:
        model.enable_gradient_checkpointing()
    args = inputs()
    observed_shape_encoding = []
    handle = model.sparse_shape_embed.register_forward_pre_hook(
        lambda module, positional: observed_shape_encoding.append(positional[0].detach())
    )
    logits = model(**args)
    labels = torch.randint(0, 31, (2, 6, 8))
    labels[~args["target_edit_mask"]] = -100
    torch.nn.functional.cross_entropy(logits, labels).backward()
    handle.remove()
    assert observed_shape_encoding[0].shape == (2, 512)
    assert not torch.equal(observed_shape_encoding[0][0], observed_shape_encoding[0][1])
    assert model.sparse_shape_embed.linear_2.weight.grad.abs().sum() > 0
    for block in [*model.transformer_blocks, *model.single_transformer_blocks]:
        assert block.attn.to_q.weight.grad.abs().sum() > 0


def test_empty_edit_mask_is_finite_and_keeps_entire_background():
    model = tiny_model().eval()
    args = inputs()
    args["target_edit_mask"].zero_()
    with torch.no_grad():
        assert torch.isfinite(model(**args)).all()


def test_checkpoint_round_trip_and_dense_checkpoint_upgrade(tmp_path):
    model = tiny_model(sparse=False).eval()
    dense_path = tmp_path / "dense"
    model.save_pretrained(dense_path)
    restored = Transformer2DModel.from_pretrained(dense_path)
    assert restored.sparse_shape_embed is None
    old_weight = restored.cond_embed.linear_1.weight.detach().clone()
    restored.enable_sparse_target(halo=0)
    torch.testing.assert_close(restored.cond_embed.linear_1.weight, old_weight, rtol=0, atol=0)
    assert restored.sparse_shape_embed.linear_2.weight.count_nonzero() == 0
    sparse_path = tmp_path / "sparse"
    restored.save_pretrained(sparse_path)
    restored_again = Transformer2DModel.from_pretrained(sparse_path).eval()
    assert restored_again.config.sparse_target
    assert restored_again.config.token_type_rope
    assert restored_again.config.sparse_target_halo == 0
    args = inputs()
    with torch.no_grad():
        torch.testing.assert_close(restored(**args), restored_again(**args))


@pytest.mark.parametrize("compiled", [False, True])
def test_sparse_lora_saves_and_restores_trainable_shape_module(tmp_path, compiled):
    model = tiny_model(sparse=False)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.enable_sparse_target(halo=0)
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"], modules_to_save=["sparse_shape_embed"])
    model.add_adapter(config)
    trained = model.sparse_shape_embed.modules_to_save["default"]
    assert all(parameter.requires_grad for parameter in trained.parameters())
    with torch.no_grad():
        trained.linear_2.weight.fill_(0.1)
        model.transformer_blocks[0].attn.to_q.lora_B["default"].weight.fill_(0.2)
    save_model = torch.compile(model, backend="eager") if compiled else model
    LoraLoaderMixin.save_lora_weights(tmp_path, unet_lora_layers=get_peft_model_state_dict(save_model))
    (tmp_path / "lora_config.json").write_text(json.dumps(config.to_dict(), default=list))
    (tmp_path / "sparse_target_config.json").write_text(json.dumps({"sparse_target_halo": 0}))
    restored = tiny_model(sparse=False)
    restored.load_state_dict(base_state)
    load_model = torch.compile(restored, backend="eager") if compiled else restored
    load_lora_adapter(load_model, str(tmp_path))
    torch.testing.assert_close(restored.sparse_shape_embed.modules_to_save["default"].linear_2.weight, trained.linear_2.weight)
    torch.testing.assert_close(
        restored.transformer_blocks[0].attn.to_q.lora_B["default"].weight,
        model.transformer_blocks[0].attn.to_q.lora_B["default"].weight,
    )


def test_compiled_model_handles_changing_active_lengths():
    model = tiny_model().eval()
    compiled = torch.compile(model, backend="eager")
    args = inputs()
    with torch.no_grad():
        torch.testing.assert_close(compiled(**args), model(**args))
        args["target_edit_mask"][:, 0:4, 0:4] = True
        torch.testing.assert_close(compiled(**args), model(**args))


def test_sparse_attention_analysis_restores_original_grid():
    from src.attention_utils import extract_image_to_text_attention
    from transformers import BatchEncoding

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return BatchEncoding({
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3),
                "special_tokens_mask": torch.zeros(1, 3),
            })

        def convert_ids_to_tokens(self, ids):
            return [str(value) for value in ids]

    model = tiny_model().eval()
    model.register_attention_hooks([0, 2])
    for block in [*model.transformer_blocks, *model.single_transformer_blocks]:
        block.attn.reference_strength_factor = torch.tensor([[1.25]])
    with torch.no_grad():
        model(**inputs(batch=1))
    result = extract_image_to_text_attention(model, Tokenizer(), "add a giraffe", [0, 2], guidance_enabled=False)
    assert result["token_grid_size"] == [3, 4]
    assert result["all_tokens"].shape == (12,)
    assert result["all_tokens"].nonzero().flatten().tolist() == [1]
    assert result["sequence_layout"]["generation_active_indices"] == [1]


@pytest.mark.parametrize("guidance_scale", [1.0, 2.0])
def test_pipeline_preserves_source_at_all_steps_and_duplicates_cfg_mask(guidance_scale):
    from PIL import Image
    from diffusers import VQModel
    from src.pipeline import Pipeline
    from src.scheduler import Scheduler
    from src.v2_utils import prepare_cond_token
    from src.dataset_utils import process_image

    model = tiny_model(sparse=False).eval()
    vq = VQModel(block_out_channels=(16,), latent_channels=4, vq_embed_dim=4,
                 num_vq_embeddings=31, norm_num_groups=4).eval()
    pipe = Pipeline(vq, None, None, model, Scheduler(mask_token_id=31), None, None)
    pipe.set_progress_bar_config(disable=True)
    source = Image.new("RGB", (16, 16), (80, 140, 50))
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:6, 6:8] = 1
    frames, observed_masks = [], []
    handle = model.register_forward_pre_hook(
        lambda module, positional, kwargs: observed_masks.append(kwargs.get("target_edit_mask")), with_kwargs=True,
    )
    prompt_embeds, text = torch.randn(1, 8), torch.randn(1, 3, 24)
    common = dict(
        prompt_embeds=prompt_embeds, encoder_hidden_states=text, reference_image=[source],
        negative_prompt_embeds=torch.zeros_like(prompt_embeds), negative_encoder_hidden_states=torch.zeros_like(text),
        height=16, width=16, num_inference_steps=3, guidance_scale=guidance_scale,
        output_type="latent", temperature=0.0,
    )
    output = pipe(**common, mask_image=mask, sparse_target=True, sparse_target_halo=0,
                  callback=lambda step, time, tokens: frames.append(tokens.clone())).images
    with torch.no_grad():
        source_tokens = prepare_cond_token(None, process_image(source, 16)["image"][None], vq).reshape(1, 16, 16)
    preserve = ~mask[:, 0].bool()
    assert model.config.sparse_target
    assert len(frames) == 3
    for tokens in [*frames, output]:
        assert torch.equal(tokens[preserve], source_tokens[preserve])
    for observed in observed_masks:
        assert observed.shape[0] == (2 if guidance_scale > 1 else 1)
        if guidance_scale > 1:
            assert torch.equal(observed[0], observed[1])
    # A sparse checkpoint can still run dense, unmasked mask-discovery inference.
    pipe(**common)
    assert observed_masks[-1] is None
    handle.remove()
