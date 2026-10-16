# Copyright 2025 EditMGT Team. All rights reserved.
import argparse
import copy
import logging
import math
import os
import json
from pathlib import Path
from datetime import timedelta
import sys
sys.path.append(os.getcwd())
import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
import diffusers.optimization
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs, set_seed
from termcolor import cprint
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, default_collate
from diffusers import EMAModel
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import is_wandb_available
from src.v2_utils import save_checkpoint, prepare_cond_token
from src.transformer import Transformer2DModel, get_text_encoder_length, load_lora_adapter
from src.pipeline import _prepare_latent_image_ids
from src.dataset_utils import HuggingFaceDataset, HuggingFaceEditDataset, HDFSParquetDataset, HDFSEditParquetDataset, \
                                LocalEditParquetDataset, tokenize_prompt, encode_prompt, move_batch_prompt_device
from src.v2_model import init_text_encoder, init_base_model, negative_prompt, get_pipeline
from src.mask_utils import (
    prepare_edit_only_corruption,
    resize_edit_mask_to_tokens,
    save_mask_debug_visualization,
)
from src.inter_edit_dataset import DEFAULT_INTER_EDIT_ROOT, InterEditDataset
from src.magicbrush_dataset import MagicBrushDataset
from src.mixed_edit_dataset import MixedEditDataset
from src.timestep_sampling import MaskRatioSampler
from src.sparse_utils import expand_edit_mask_for_halo
from src.mask_schedule import validate_inference_shift
from src.train_validation import (
    prepare_inter_edit_validation_cases,
    run_inter_edit_validation,
    synchronized_main_process_call,
    validation_model_state,
)

logger = get_logger(__name__, log_level="INFO")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NCCL_DEBUG'] = 'ERROR'


def make_mask_ratio_sampler(args):
    return MaskRatioSampler(
        distribution=args.timestep_sampling,
        logit_mean=args.timestep_logit_mean,
        logit_std=args.timestep_logit_std,
        shift=args.timestep_shift,
        cosine_mix=args.timestep_cosine_mix,
        low_noise_mix=args.timestep_low_noise_mix,
        low_noise_threshold=args.condition_cache_threshold,
        min_mask_ratio=args.min_masking_rate,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_encoder_architecture",
        type=str,
        default="CLIP",
        required=False,
        help="The architecture of the text encoder. One of ['CLIP', 'open_clip', 'flan-t5-base','Qwen2-0.5B','gemini-2b',long_CLIP_T5_base','CLIP_T5_base', 'CLIP_Gemma']",
    )
    parser.add_argument(
        "--instance_dataset",
        type=str,
        default=None,
        required=False,
        help="The dataset to use for training. One of ['MSCOCO600K', 'PickaPicV2']",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    ) # Collov-Labs/Monetico for 512; MeissonFlow/Meissonic for 1024
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--transformer_subfolder",
        type=str,
        default="transformer",
        help="Transformer weights subfolder (use 'editmgt' to fine-tune the released editing checkpoint).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="muse_training",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Save validation images every X optimizer steps; 0 disables validation.",
    )
    parser.add_argument("--validation_data_dir", type=str, default=None,
                        help="Inter-Edit root for test visualizations (defaults to the training data root).")
    parser.add_argument("--validation_num_images", type=int, default=5,
                        help="Fixed test cases, choosing different edit types first.")
    parser.add_argument("--validation_sample_indices", type=int, nargs="+", default=None,
                        help="Explicit test row indices; overrides --validation_num_images.")
    parser.add_argument("--validation_num_inference_steps", type=int, default=18)
    parser.add_argument("--validation_inference_shift", type=validate_inference_shift, default=1.0,
                        help="Inference-only mask schedule shift for fixed test previews; independent of training timestep_shift.")
    parser.add_argument("--validation_guidance_scale", type=float, default=6.0)
    parser.add_argument("--validation_reference_strength", type=float, default=1.0)
    parser.add_argument("--validation_disable_reference_kv_cache", action="store_true",
                        help="Recompute the full reference at every validation inference step.")
    parser.add_argument("--validation_reference_kv_cache_threshold", type=float, default=0.3,
                        help="Capture reference K/V when the scheduled remaining editable mask fraction is <= this value.")
    parser.add_argument("--validation_seed", type=int, default=42)
    parser.add_argument("--validation_at_start", action="store_true",
                        help="Also visualize the initial/resumed model before training updates.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'Logging integration: "wandb" (default), "tensorboard", "comet_ml", or "all". '
            'Use "none" to disable external trackers; local validation images are still saved.'
        ),
    )
    parser.add_argument("--validation_prompts", type=str, nargs="*")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--split_vae_encode", type=int, required=False, default=None)
    parser.add_argument("--split_manifest", type=str, default=None)
    parser.add_argument("--dataset_split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--dataset_language", choices=["all", "en", "zh"], default="all",
        help="Inter-Edit prompt language for BOTH the training loader and official test visualizations.",
    )
    parser.add_argument(
        "--dataset_cache_dir", type=str, default=None,
        help="Writable tar member-index cache for InterEditDataset (default: data/inter_edit_tar_index).",
    )
    parser.add_argument(
        "--magicbrush_data_dir", type=str, default=None,
        help="Optional local MagicBrush release to mix with Inter-Edit; only train-*.parquet is used.",
    )
    parser.add_argument(
        "--magicbrush_probability", type=float, default=0.2,
        help="Per-example MagicBrush sampling probability when mixing; Inter-Edit receives the remaining weight.",
    )
    parser.add_argument("--magicbrush_read_batch_size", type=int, default=4,
                        help="Bound the number of image-containing parquet rows materialized together per worker.")
    parser.add_argument("--min_masking_rate", type=float, default=0.0)
    parser.add_argument(
        "--timestep_sampling", choices=["cosine", "uniform", "logit_normal"], default="cosine",
        help="Training distribution over editable-token mask ratios; cosine preserves the original uniform-progress sampler.",
    )
    parser.add_argument("--timestep_logit_mean", type=float, default=0.0,
                        help="Mean of the normal variable before sigmoid; positive favors more masking.")
    parser.add_argument("--timestep_logit_std", type=float, default=1.0)
    parser.add_argument("--timestep_shift", type=float, default=1.0,
                        help="Main-component mask-ratio shift s*r/(1+(s-1)*r); s>1 favors higher noise.")
    parser.add_argument("--timestep_cosine_mix", type=float, default=0.0,
                        help="Probability of an unshifted original-cosine sample to retain endpoint coverage.")
    parser.add_argument("--timestep_low_noise_mix", type=float, default=0.0,
                        help="Probability of a uniform sample between min_masking_rate and condition_cache_threshold.")
    parser.add_argument(
        "--time_conditioning", choices=["legacy", "mask_ratio"], default="legacy",
        help="Use normalized mask ratio for matching train/infer time embeddings, or preserve old-checkpoint time units.",
    )
    parser.add_argument(
        "--mask_loss_reduction", choices=["token", "sample"], default="token",
        help="Average CE over all supervised tokens (legacy) or normalize each example before averaging.",
    )
    parser.add_argument("--cond_dropout_prob", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.", required=False)
    parser.add_argument("--use_lora", action="store_true", help="Fine tune the model using LoRa")
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--compile_model", action="store_true",
        help="Opt in to torch.compile; default is eager training for variable-length sparse targets.",
    )
    parser.add_argument(
        "--compile_backend", choices=["inductor", "aot_eager", "eager"], default="inductor",
        help="Backend used only with --compile_model. aot_eager/eager are compiler diagnostic options.",
    )
    parser.add_argument("--prompt_prefix", type=str, required=False, default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--train_edit_model", default=False, help='Train T2I model is False; Train edit model is True')
    parser.add_argument(
        "--edit_only_diffusion",
        action="store_true",
        help="Corrupt/supervise the edit-generation region (including halo when enabled); preserve tokens come from source.",
    )
    parser.add_argument(
        "--sparse_target", action="store_true",
        help="Use full reference + sparse edit/halo target tokens; enables --edit_only_diffusion.",
    )
    parser.add_argument(
        "--sparse_target_halo", type=int, default=2,
        help="Target-mask dilation radius on the Transformer grid after downsampling.",
    )
    parser.add_argument(
        "--supervise_target_halo", action=argparse.BooleanOptionalAction, default=True,
        help="Corrupt/supervise the expanded active VQ region and regenerate it in inference; disable for context-only halo.",
    )
    parser.add_argument(
        "--cache_conditioning", action="store_true",
        help="Train low-noise target-independent text/reference prefixes with fixed modulation for joint KV reuse.",
    )
    parser.add_argument(
        "--condition_cache_threshold", type=float, default=0.3,
        help="Scheduled edit-mask ratio below which prefix isolation/fixed context modulation apply.",
    )
    parser.add_argument(
        "--mask_debug_dir",
        type=str,
        default=None,
        help="Optionally save one source/mask/token-mask alignment panel here.",
    )
    args = parser.parse_args()
    if args.report_to.lower() == "none":
        args.report_to = None
    if args.validation_steps < 0:
        parser.error("--validation_steps must be non-negative (0 disables validation)")
    if args.validation_num_images < 1 or args.validation_num_inference_steps < 1:
        parser.error("Validation image count and inference steps must be positive")
    if not 0.0 <= args.validation_reference_kv_cache_threshold <= 1.0:
        parser.error("--validation_reference_kv_cache_threshold must be between 0 and 1")
    if args.sparse_target_halo < 0:
        parser.error("--sparse_target_halo must be non-negative")
    if args.dataset_language != "all" and args.instance_dataset != "InterEditDataset":
        parser.error("--dataset_language is supported only for InterEditDataset")
    if args.magicbrush_data_dir and (
        args.instance_dataset != "InterEditDataset" or args.dataset_split != "train" or not args.train_edit_model
    ):
        parser.error("MagicBrush mixing requires InterEditDataset, --dataset_split train and --train_edit_model")
    if not 0.0 < args.magicbrush_probability < 1.0 or args.magicbrush_read_batch_size < 1:
        parser.error("MagicBrush probability must be in (0,1) and its read batch size must be positive")
    if args.sparse_target:
        args.edit_only_diffusion = True
    if not 0.0 < args.condition_cache_threshold <= 1.0:
        parser.error("--condition_cache_threshold must be in (0,1]")
    if args.cache_conditioning and not args.train_edit_model:
        parser.error("--cache_conditioning requires --train_edit_model")
    if args.cache_conditioning and args.min_masking_rate > args.condition_cache_threshold:
        parser.error("--min_masking_rate would exclude the entire cache-conditioning phase")
    if args.edit_only_diffusion and not args.train_edit_model:
        parser.error("--edit_only_diffusion/--sparse_target requires --train_edit_model")
    try:
        make_mask_ratio_sampler(args)
    except ValueError as error:
        parser.error(str(error))

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    return args

def initialize_local_cuda_device():
    """Bind each DDP worker before NCCL/Accelerator can initialize CUDA."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        return
    visible_count = torch.cuda.device_count()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", local_rank + 1))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if local_rank >= visible_count or local_world_size > visible_count:
        # Accelerate 1.0.1 otherwise wraps local_rank modulo visible_count,
        # which can place several workers on the same GPU.
        raise RuntimeError(
            f"Invalid DDP GPU mapping: LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={local_world_size}, "
            f"visible CUDA devices={visible_count}, CUDA_VISIBLE_DEVICES={visible_devices!r}. "
            "Launch at most one worker per visible GPU."
        )
    print(
        f"[rank {os.environ.get('RANK', local_rank)}] LOCAL_RANK={local_rank}, "
        f"device=cuda:{local_rank}, visible GPUs={visible_count}, "
        f"CUDA_VISIBLE_DEVICES={visible_devices}", flush=True,
    )
    torch.cuda.set_device(local_rank)


def main(args):
    wandb = None
    mask_ratio_sampler = make_mask_ratio_sampler(args)
    timestep_sampling_config = {
        "version": 1,
        "time_conditioning": args.time_conditioning,
        "mask_loss_reduction": args.mask_loss_reduction,
        "sampler": mask_ratio_sampler.to_dict(),
    }
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    try:
        initialize_local_cuda_device()
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            # Compilation is controlled explicitly below. Do not let a launcher
            # configuration silently re-enable Inductor for the eager default.
            dynamo_backend="no",
            # Other ranks wait while rank zero reads test tar indexes or generates
            # preview images; allow a cold shared filesystem more than 10 minutes.
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
        )
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        raise RuntimeError(
            "CUDA ran out of memory during device/distributed initialization, before loading any models. "
            f"RANK={os.environ.get('RANK', '0')}, LOCAL_RANK={os.environ.get('LOCAL_RANK', '<unset>')}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "Check GPU memory/process ownership and worker-to-GPU mapping. "
            "Reducing the training batch size does not reduce this initialization allocation."
        ) from error

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        tracker_init_kwargs = {}
        if args.report_to in ("wandb", "all") and is_wandb_available():
            import wandb

            wandb_kwargs = {"name": Path(args.output_dir).name}
            # Let the SDK use the account's configured default entity when
            # WANDB_ENTITY is absent; an explicit team/user remains supported.
            wandb_entity = os.environ.get("WANDB_ENTITY")
            if wandb_entity:
                wandb_kwargs["entity"] = wandb_entity
            if args.wandb_id:
                wandb_kwargs.update(id=args.wandb_id, resume="allow")
            tracker_init_kwargs["wandb"] = wandb_kwargs
        # Initialize W&B exactly once, through Accelerate, so training metrics
        # and visualization images belong to the same run and finish together.
        accelerator.init_trackers(
            os.environ.get("WANDB_PROJECT") or "editmgt",
            config=vars(copy.deepcopy(args)), init_kwargs=tracker_init_kwargs,
        )

    if args.seed is not None:
        set_seed(args.seed)

    validation_cases = None
    validation_output_dir = Path(args.output_dir) / "validation"
    if args.validation_steps > 0 and args.train_edit_model:
        validation_data_dir = args.validation_data_dir or (
            (args.instance_data_dir or DEFAULT_INTER_EDIT_ROOT) if args.instance_dataset == "InterEditDataset" else DEFAULT_INTER_EDIT_ROOT
        )
        if Path(validation_data_dir).name == "train" and (Path(validation_data_dir).parent / "test").is_dir():
            validation_data_dir = str(Path(validation_data_dir).parent)
        validation_cases = synchronized_main_process_call(
            accelerator,
            lambda: prepare_inter_edit_validation_cases(
                validation_data_dir, validation_output_dir, args.resolution,
                num_images=args.validation_num_images, sample_indices=args.validation_sample_indices,
                cache_dir=args.dataset_cache_dir, language=args.dataset_language,
            ),
            description="Loading Inter-Edit test visualization cases",
        )
        if accelerator.is_main_process:
            logger.info("Fixed Inter-Edit test visualization sample IDs: %s", [case["sample_id"] for case in validation_cases])
            logger.info("Inter-Edit train/test language filter: %s", args.dataset_language)

    text_encoder, tokenizer, text_att_dim, text_proj_dim = init_text_encoder(text_encoder_architecture=args.text_encoder_architecture,
                                pretrained_model_name_or_path=args.pretrained_model_name_or_path, 
                                revision=args.revision, variant=args.variant, return_dim=True)
    vq_model, scheduler = init_base_model(pretrained_model_name_or_path=args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant)

    if isinstance(text_encoder, list):
        text_encoder[0].eval()
        text_encoder[0].requires_grad_(False)
        text_encoder[1].eval()
        text_encoder[1].requires_grad_(False)
    else:
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    vq_model.requires_grad_(False)

    if args.text_encoder_architecture == 'CLIP':
        model = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, 
                                        subfolder=args.transformer_subfolder, low_cpu_mem_usage=False, device_map=None)
    else:
        model = Transformer2DModel(
            patch_size = 1,
            in_channels = 64,
            num_layers = 14,
            num_single_layers = 28,
            attention_head_dim = 128,
            num_attention_heads = 8,
            joint_attention_dim = text_att_dim,
            pooled_projection_dim = text_proj_dim,
            guidance_embeds = False,
            axes_dims_rope = (16, 56, 56),
            downsample= args.resolution==1024,
            upsample= args.resolution==1024,
            text_encoder_architecture=args.text_encoder_architecture,
            connector_type='linear' if 'CLIP' != args.text_encoder_architecture else None,
        )
        model_tmp = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder=args.transformer_subfolder,
            low_cpu_mem_usage=False, device_map=None)
        state_dict = model_tmp.state_dict()

        def load_state_dict_with_skip(model, state_dict):
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
            model.load_state_dict(filtered_state_dict, strict=False)

        load_state_dict_with_skip(model, state_dict)

        del model_tmp 

    if args.sparse_target:
        model.enable_sparse_target(halo=args.sparse_target_halo, supervise_halo=args.supervise_target_halo)
    if args.cache_conditioning:
        model.enable_condition_cache_training(threshold=args.condition_cache_threshold)
    model.set_time_conditioning(args.time_conditioning)

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            modules_to_save=["sparse_shape_embed"] if args.sparse_target else None,
        )
        model.add_adapter(lora_config)

    model.train()

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.use_ema:
        ema = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
            model_cls= Transformer2DModel,
            model_config=model.config,
        )

    if args.compile_model:
        if args.gradient_checkpointing:
            import torch._dynamo as dynamo

            dynamo.config.optimize_ddp = False
        logger.info("Opt-in torch.compile backend=%s, dynamic=True", args.compile_backend)
        model = torch.compile(model, backend=args.compile_backend, dynamic=True)
    else:
        logger.info("Training in eager mode; torch.compile is disabled")

    def unwrap_training_model(module):
        unwrapped = accelerator.unwrap_model(module)
        return getattr(unwrapped, "_orig_mod", unwrapped)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            with open(os.path.join(output_dir, "timestep_sampling_config.json"), "w") as f:
                json.dump(timestep_sampling_config, f, indent=2)
            transformer_lora_layers_to_save = None

            for model_ in models:
                transformer_to_save = unwrap_training_model(model_)
                if isinstance(transformer_to_save, Transformer2DModel):
                    if args.use_lora:
                        transformer_lora_layers_to_save = get_peft_model_state_dict(transformer_to_save)
                    else:
                        transformer_to_save.save_pretrained(os.path.join(output_dir, "transformer"))
                else:
                    raise ValueError(f"unexpected save model: {model_.__class__}")

                weights.pop()

            if transformer_lora_layers_to_save is not None:
                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=transformer_lora_layers_to_save,
                    text_encoder_lora_layers=None,
                )
                class LoraConfigEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, set):
                            return list(obj)
                        return super().default(obj)
                with open(os.path.join(output_dir, "lora_config.json"), 'w') as f:
                    json.dump(lora_config.to_dict(), f, indent=2, cls=LoraConfigEncoder)
                with open(os.path.join(output_dir, "time_conditioning_config.json"), "w") as f:
                    json.dump({"time_conditioning": args.time_conditioning}, f)
                if args.sparse_target:
                    with open(os.path.join(output_dir, "sparse_target_config.json"), "w") as f:
                        json.dump({
                            "sparse_target_halo": args.sparse_target_halo,
                            "supervise_target_halo": args.supervise_target_halo,
                        }, f)
                if args.cache_conditioning:
                    with open(os.path.join(output_dir, "condition_cache_config.json"), "w") as f:
                        json.dump({"condition_cache_threshold": args.condition_cache_threshold}, f)

            if args.use_ema:
                ema.save_pretrained(os.path.join(output_dir, "ema_model"))

    def load_model_hook(models, input_dir):
        transformer = None
        sampling_config_path = Path(input_dir) / "timestep_sampling_config.json"
        if sampling_config_path.is_file():
            with sampling_config_path.open() as f:
                saved_sampling_config = json.load(f)
            if saved_sampling_config != timestep_sampling_config:
                raise ValueError(
                    "Checkpoint timestep sampling/time conditioning/loss reduction differs from this run. "
                    "Resume with matching options, or start a new output directory for the changed objective."
                )
        elif (args.time_conditioning != "legacy" or args.mask_loss_reduction != "token" or
              args.timestep_sampling != "cosine" or args.timestep_shift != 1.0 or
              args.timestep_cosine_mix != 0.0 or args.timestep_low_noise_mix != 0.0):
            raise ValueError(
                "This checkpoint predates configurable time sampling. Resume it in legacy cosine/token mode, "
                "or start a separate run for normalized mask-ratio conditioning and the new sampler."
            )

        while len(models) > 0:
            model_ = unwrap_training_model(models.pop())
            if isinstance(model_, Transformer2DModel):
                if args.use_lora:
                    time_config_path = Path(input_dir) / "time_conditioning_config.json"
                    saved_time_mode = "legacy"
                    if time_config_path.is_file():
                        with time_config_path.open() as f:
                            saved_time_mode = json.load(f)["time_conditioning"]
                    if model_.config.time_conditioning != saved_time_mode:
                        raise ValueError("LoRA checkpoint time conditioning differs from --time_conditioning")
                    transformer = model_
                else:
                    load_model = Transformer2DModel.from_pretrained(os.path.join(input_dir, "transformer"),low_cpu_mem_usage=False,device_map=None)
                    if model_.config.time_conditioning != load_model.config.time_conditioning:
                        raise ValueError("Checkpoint time conditioning differs from --time_conditioning")
                    if (model_.config.cache_conditioning != load_model.config.cache_conditioning or
                        (model_.config.cache_conditioning and
                         model_.config.condition_cache_threshold != load_model.config.condition_cache_threshold)):
                        raise ValueError(
                            "Checkpoint cache-conditioning settings differ from this run; "
                            "use matching --cache_conditioning/--condition_cache_threshold or a separate output directory."
                        )
                    # Diffusers checkpoints store the original model's keys.
                    # Load into that model regardless of compile/DDP wrappers.
                    model_.load_state_dict(load_model.state_dict())
                    del load_model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer is not None:
            transformer = load_lora_adapter(transformer, input_dir)
            if args.sparse_target:
                # Explicit training options govern the new objective even when
                # continuing a previous context-only LoRA checkpoint.
                transformer.enable_sparse_target(
                    halo=args.sparse_target_halo, supervise_halo=args.supervise_target_halo,
                )

        if args.use_ema:
            ema_model_path = os.path.join(input_dir, "ema_model")
            # If you skip loading EMA, it is equivalent to starting a new EMA. 
            # It is used for training that was not EMA before and switching to EMA training.
            if os.path.exists(ema_model_path) and os.path.isdir(ema_model_path):
                try:
                    load_from = EMAModel.from_pretrained(ema_model_path, model_cls=Transformer2DModel)
                    ema.load_state_dict(load_from.state_dict())
                    del load_from
                    cprint("EMA model loaded successfully", 'red')
                except Exception as e:
                    cprint(f"Failed to load EMA model: {e}. Continuing without EMA state.", 'red')
            else:
                cprint(f"EMA model not found at {ema_model_path}. Continuing without EMA state.", 'red')

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    if args.use_8bit_adam:
        from importlib.metadata import PackageNotFoundError, version
        from packaging.version import Version

        try:
            bnb_version = version("bitsandbytes")
        except PackageNotFoundError as error:
            raise ImportError(
                "8-bit AdamW requires bitsandbytes. Install bitsandbytes==0.48.2 into this "
                "environment, or use USE_8BIT_ADAM=0 with train/train.sh for PyTorch AdamW."
            ) from error
        if (torch.version.cuda is not None and Version(torch.version.cuda) >= Version("13.0")
                and Version(bnb_version) < Version("0.48.0")):
            raise RuntimeError(
                f"bitsandbytes {bnb_version} does not provide CUDA 13 kernels, but PyTorch uses "
                f"CUDA {torch.version.cuda}. Install bitsandbytes==0.48.2, or select "
                "USE_8BIT_ADAM=0 for native PyTorch AdamW. No optimizer updates have been performed."
            )
        try:
            import bitsandbytes as bnb
            from bitsandbytes.cextension import lib as bnb_library
        except ImportError as error:
            raise ImportError(
                "Cannot import bitsandbytes CUDA optimizer support. Install bitsandbytes==0.48.2 "
                "into this environment, or select USE_8BIT_ADAM=0 for PyTorch AdamW."
            ) from error
        if bnb_library is None or getattr(bnb_library, "compiled_with_cuda", True) is False:
            raise RuntimeError(
                f"bitsandbytes {bnb_version} could not load a CUDA optimizer library for "
                f"PyTorch CUDA {torch.version.cuda}. Check its native-library loading messages "
                "and any BNB_CUDA_VERSION override, or select USE_8BIT_ADAM=0. "
                "A CPU-only bitsandbytes library cannot update this model's CUDA parameters."
            )

        optimizer_cls = bnb.optim.AdamW8bit
        optimizer_kwargs = {}
        logger.info("Optimizer: bitsandbytes AdamW8bit, bitsandbytes=%s, CUDA=%s", bnb_version, torch.version.cuda)
    else:
        optimizer_cls = torch.optim.AdamW
        # Avoid the extra parameter-sized temporary buffers of the foreach
        # implementation when using full-precision optimizer states.
        optimizer_kwargs = {"foreach": False}
        logger.info("Optimizer: torch.optim.AdamW (foreach=False)")

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.adam_weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
        **optimizer_kwargs,
    )

    logger.info("Creating dataloaders and lr_scheduler")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable_params = sum(p.numel() for p in trainable_params)
    cprint(f"Number of Model parameters: {sum(p.numel() for p in model.parameters()) // 1024 // 1024} M", 'cyan')
    cprint(f"Number of trainable parameters: {num_trainable_params//1024//1024} M", 'cyan')

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if args.instance_dataset == "HDFSParquetDataset":
        dataset = HDFSParquetDataset(
            root_dir=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
    elif args.instance_dataset == "HDFSEditParquetDataset":
        dataset = HDFSEditParquetDataset(
            root_dir=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            require_edit_mask=args.edit_only_diffusion,
        )
    elif args.instance_dataset == "LocalEditParquetDataset":
        if args.split_manifest is None:
            raise ValueError("LocalEditParquetDataset requires --split_manifest")
        dataset = LocalEditParquetDataset(
            root_dir=args.instance_data_dir,
            split_manifest=args.split_manifest,
            split=args.dataset_split,
            tokenizer=tokenizer,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            require_edit_mask=args.edit_only_diffusion,
        )
    elif args.instance_dataset == "InterEditDataset":
        dataset = InterEditDataset(
            root_dir=args.instance_data_dir, split=args.dataset_split,
            tokenizer=tokenizer, size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
            rank=accelerator.process_index, world_size=accelerator.num_processes,
            cache_dir=args.dataset_cache_dir, seed=args.seed if args.seed is not None else 42,
            language=args.dataset_language,
        )
        logger.info("Inter-Edit rank %s: %s %s samples, language=%s (before empty-mask filtering)",
                    accelerator.process_index, len(dataset), args.dataset_split, args.dataset_language)
        if args.magicbrush_data_dir:
            inter_edit_dataset = dataset
            magicbrush_dataset = MagicBrushDataset(
                root_dir=args.magicbrush_data_dir, split="train",
                tokenizer=tokenizer, size=args.resolution,
                text_encoder_architecture=args.text_encoder_architecture,
                rank=accelerator.process_index, world_size=accelerator.num_processes,
                seed=args.seed if args.seed is not None else 42, language=args.dataset_language,
                cache_dir=Path(args.dataset_cache_dir) / "magicbrush" if args.dataset_cache_dir else None,
                read_batch_size=args.magicbrush_read_batch_size,
            )
            dataset = MixedEditDataset(
                datasets=[inter_edit_dataset, magicbrush_dataset], names=["Inter-Edit", "MagicBrush"],
                weights=[1.0 - args.magicbrush_probability, args.magicbrush_probability],
                rank=accelerator.process_index, seed=args.seed if args.seed is not None else 42,
            )
            mixture_dir = Path(args.output_dir) / "data_mixture"
            mixture_dir.mkdir(parents=True, exist_ok=True)
            mixture_info = {
                "rank": accelerator.process_index, "world_size": accelerator.num_processes,
                "language_filter": args.dataset_language,
                "count_scope": "rank-local, language-filtered, before empty-mask exclusions",
                "sources": [
                    {"name": "Inter-Edit", "root": str(inter_edit_dataset.reader.root), "split": "train",
                     "sample_count": len(inter_edit_dataset), "probability": 1.0 - args.magicbrush_probability},
                    {"name": "MagicBrush", "root": str(magicbrush_dataset.root.resolve()), "split": "train",
                     "sample_count": len(magicbrush_dataset), "probability": args.magicbrush_probability},
                ],
                "validation": {"dataset": "Inter-Edit", "split": "test", "language_filter": args.dataset_language},
            }
            (mixture_dir / f"rank_{accelerator.process_index:04d}.json").write_text(
                json.dumps(mixture_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            logger.info("Mixing Inter-Edit/MagicBrush with probabilities %.2f/%.2f; MagicBrush rank-local rows=%s",
                        1.0 - args.magicbrush_probability, args.magicbrush_probability, len(magicbrush_dataset))
    elif args.instance_dataset == 'HuggingFaceDataset':
        # Inter-Edit and local parquet readers do not depend on HF datasets.
        from datasets import load_dataset

        dataset = HuggingFaceDataset(
            hf_dataset=load_dataset(args.instance_data_dir, split="train"),
            tokenizer=tokenizer,
            image_key='image_file',
            prompt_key='edit_instruction',
            prompt_prefix=args.prompt_prefix,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
        )
    elif args.instance_dataset == 'HuggingFaceEditDataset':
        from datasets import load_dataset

        dataset = HuggingFaceEditDataset(
            hf_dataset=load_dataset(args.instance_data_dir, split="replace"), # example: read Bin1117/AnyEdit
            tokenizer=tokenizer,
            image_key='image_file',
            prompt_key='edit_instruction',
            prompt_prefix=args.prompt_prefix,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
            require_edit_mask=args.edit_only_diffusion,
        )
    else:
        assert False

    if args.instance_dataset in ["HDFSParquetDataset", "HDFSEditParquetDataset", "LocalEditParquetDataset", "InterEditDataset"]:
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            collate_fn=default_collate,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            collate_fn=default_collate,
            pin_memory=True,
        )

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )

    logger.info("Preparing model, optimizer and dataloaders")

    if args.instance_dataset == "InterEditDataset":
        # Rank/worker ownership is already handled by the dataset. Preparing the
        # loader (including its optional MagicBrush mixture) again would shard
        # it a second time and lose coverage.
        # The training loop moves each input tensor to accelerator.device.
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    else:
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader
        )

    train_dataloader.num_batches = len(train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if isinstance(text_encoder, list):
        text_encoder[0].to(device=accelerator.device, dtype=weight_dtype)
        text_encoder[1].to(device=accelerator.device, dtype=weight_dtype)
    else:
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)

    vq_model.to(device=accelerator.device)

    if args.use_ema:
        ema.to(accelerator.device)

    with torch.no_grad():
        empty_embeds, empty_clip_embeds = encode_prompt(
            text_encoder, tokenize_prompt(tokenizer, "", args.text_encoder_architecture, device=accelerator.device, non_blocking=True), args.text_encoder_architecture
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info("  Timestep/noise sampling = %s", timestep_sampling_config)

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                resume_from_checkpoint = None

        if resume_from_checkpoint is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
        else:
            cprint('resume successfully!', 'red')
            accelerator.print(f"Resuming from checkpoint {resume_from_checkpoint}")

    if resume_from_checkpoint is None:
        global_step = 0
        first_epoch = 0
    else:
        accelerator.load_state(resume_from_checkpoint)  # load opt and model
        global_step = int(os.path.basename(resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "timestep_sampling_config.json"), "w") as f:
            json.dump(timestep_sampling_config, f, indent=2)

    # This is to solve the inconsistent tensor device issue
    if args.use_ema:
        ema.shadow_params = [p.to(accelerator.device) for p in ema.shadow_params]

    if args.edit_only_diffusion and not args.train_edit_model:
        raise ValueError("--edit_only_diffusion requires --train_edit_model and source/reference images")
    mask_debug_saved = False

    last_validation_step = -1

    def generate_validation(step):
        # Use the same weights without triggering a new torch.compile inference
        # graph for each sparse sequence length.
        validation_model = unwrap_training_model(model)
        logger.info("Generating validation images at optimizer step %s", step)
        pipe = None
        try:
            with validation_model_state(validation_model, ema if args.use_ema else None), accelerator.autocast():
                pipe = get_pipeline(
                    text_encoder_architecture=args.text_encoder_architecture,
                    transformer=validation_model, tokenizer=tokenizer, text_encoder=text_encoder,
                    vq_model=vq_model, scheduler=type(scheduler).from_config(scheduler.config),
                )
                if args.train_edit_model:
                    report = run_inter_edit_validation(
                        pipe, validation_cases, validation_output_dir, step,
                        num_inference_steps=args.validation_num_inference_steps,
                        inference_shift=args.validation_inference_shift,
                        guidance_scale=args.validation_guidance_scale,
                        reference_strength=args.validation_reference_strength,
                        seed=args.validation_seed, edit_only_diffusion=args.edit_only_diffusion,
                        sparse_target=args.sparse_target, sparse_target_halo=args.sparse_target_halo,
                        negative_prompt=negative_prompt, use_ema=args.use_ema,
                        use_reference_kv_cache=not args.validation_disable_reference_kv_cache,
                        reference_kv_cache_threshold=args.validation_reference_kv_cache_threshold,
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log({
                            "validation/inter_edit": [wandb.Image(panel, caption=caption)
                                                      for panel, caption in zip(report["panels"], report["captions"])],
                            "validation/contact_sheet": wandb.Image(report["contact_sheet"]),
                            "validation/seconds": report["seconds"],
                        }, step=step)
                    logger.info("Saved Inter-Edit validation to %s (%.1fs)", report["step_dir"], report["seconds"])
                else:
                    prompts = args.validation_prompts or ["A scenic mountain landscape."]
                    pipe.set_progress_bar_config(disable=True)
                    images = pipe(
                        prompt=prompts, height=args.resolution, width=args.resolution,
                        negative_prompt=[negative_prompt] * len(prompts),
                        guidance_scale=args.validation_guidance_scale,
                        num_inference_steps=args.validation_num_inference_steps,
                        generator=torch.Generator(device=accelerator.device).manual_seed(args.validation_seed),
                    ).images
                    step_dir = validation_output_dir / f"step_{step:06d}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    for index, image in enumerate(images):
                        image.save(step_dir / f"generated_{index:03d}.png")
                    if wandb is not None and wandb.run is not None:
                        wandb.log({"validation/generated_images": [wandb.Image(image, caption=prompt)
                                  for image, prompt in zip(images, prompts)]}, step=step)
        finally:
            del pipe
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()

    def visualize_training_step(step):
        nonlocal last_validation_step
        if args.validation_steps > 0 and step != last_validation_step:
            synchronized_main_process_call(
                accelerator, lambda: generate_validation(step),
                description=f"Training visualization at optimizer step {step}",
            )
            last_validation_step = step

    if args.validation_at_start:
        visualize_training_step(global_step)
    
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            torch.cuda.empty_cache()
            with torch.no_grad():
                micro_conds = batch["micro_conds"].to(accelerator.device, non_blocking=True)
                pixel_values = batch["image"].to(accelerator.device, non_blocking=True)

                image_tokens = prepare_cond_token(split_vae_encode=args.split_vae_encode, 
                                        pixel_values=pixel_values, vq_model=vq_model) # 512: [bs, 1024]

                batch_size, seq_len = image_tokens.shape

                # The reference branch remains unchanged, but its clean source tokens
                # are also used to construct preserve regions in edit-only mode.
                if args.train_edit_model:
                    reference_values = batch["reference_image"].to(accelerator.device, non_blocking=True)
                    source_tokens = prepare_cond_token(
                        split_vae_encode=args.split_vae_encode,
                        pixel_values=reference_values,
                        vq_model=vq_model,
                    )

                mask_prob = mask_ratio_sampler.sample(batch_size, device=image_tokens.device)

                mask_id = accelerator.unwrap_model(model).config.vocab_size - 1
                batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)

                if args.edit_only_diffusion:
                    token_side = int(math.isqrt(seq_len))
                    if token_side * token_side != seq_len:
                        raise ValueError(f"Expected a square VQ token grid, got {seq_len} tokens")
                    token_edit_mask = resize_edit_mask_to_tokens(
                        batch["edit_mask"].to(image_tokens.device, non_blocking=True),
                        (token_side, token_side),
                    )
                    token_generation_mask = token_edit_mask
                    if args.sparse_target and args.supervise_target_halo:
                        grid_side = (token_side // 2 if unwrap_training_model(model).down_block.downsample is not None
                                     else token_side)
                        token_generation_mask = expand_edit_mask_for_halo(
                            token_edit_mask, (token_side, token_side), (grid_side, grid_side), args.sparse_target_halo,
                        )
                    edit_token_count = token_generation_mask.sum(dim=-1)
                    if (edit_token_count == 0).any():
                        raise ValueError("Encountered an empty edit mask after token-grid resizing")

                    input_ids, labels, corruption_mask, loss_mask = prepare_edit_only_corruption(
                        target_tokens=image_tokens,
                        source_tokens=source_tokens,
                        token_edit_mask=token_generation_mask,
                        mask_probability=mask_prob,
                        mask_token_id=mask_id,
                    )
                    preserve_supervised_count = (loss_mask & ~token_generation_mask).sum()
                    assert preserve_supervised_count.item() == 0

                    edit_token_ratio = token_edit_mask.float().mean()
                    generation_token_ratio = token_generation_mask.float().mean()
                    halo_supervised_count = (loss_mask & ~token_edit_mask).sum()
                    corrupted_edit_token_ratio = (
                        corruption_mask.sum() / edit_token_count.sum().clamp(min=1)
                    )
                    supervised_token_count = loss_mask.sum()

                    if args.mask_debug_dir and accelerator.is_main_process and not mask_debug_saved:
                        save_mask_debug_visualization(
                            reference_values[0], batch["edit_mask"][0], token_generation_mask[0],
                            (token_side, token_side),
                            os.path.join(args.mask_debug_dir, "mask_alignment.png"),
                        )
                        mask_debug_saved = True
                else:
                    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
                    mask = batch_randperm < num_token_masked.unsqueeze(-1)
                    input_ids = torch.where(mask, mask_id, image_tokens)
                    labels = torch.where(mask, image_tokens, -100)

                with torch.no_grad():
                    batch["prompt_input_ids"] = move_batch_prompt_device(prompts=batch["prompt_input_ids"], device=accelerator.device, 
                                                                        split=isinstance(text_encoder, list), non_blocking=True)
                    encoder_hidden_states, cond_embeds = encode_prompt(
                            text_encoder, batch["prompt_input_ids"], args.text_encoder_architecture
                    )

                if args.cond_dropout_prob > 0.0:
                    assert encoder_hidden_states is not None

                    batch_size = encoder_hidden_states.shape[0]

                    mask = (
                        torch.zeros((batch_size, 1, 1), device=encoder_hidden_states.device).float().uniform_(0, 1)
                        < args.cond_dropout_prob
                    )

                    empty_embeds_ = empty_embeds.expand(batch_size, -1, -1)
                    encoder_hidden_states = torch.where(
                        (encoder_hidden_states * mask).bool(), encoder_hidden_states, empty_embeds_
                    )

                    empty_clip_embeds_ = empty_clip_embeds.expand(batch_size, -1)
                    cond_embeds = torch.where((cond_embeds * mask.squeeze(-1)).bool(), cond_embeds, empty_clip_embeds_)

                bs = input_ids.shape[0]
                vae_scale_factor = 2 ** (len(vq_model.config.block_out_channels) - 1)
                resolution = args.resolution // vae_scale_factor
                input_ids = input_ids.reshape(bs, resolution, resolution)

                # --- edit
                if args.train_edit_model:
                    reference_image_hidden_states = source_tokens.reshape(bs, resolution, resolution)
                # ----

            with torch.no_grad():
                batch["prompt_input_ids"] = move_batch_prompt_device(prompts=batch["prompt_input_ids"], device=accelerator.device, 
                                                                    split=isinstance(text_encoder, list), non_blocking=True)
                encoder_hidden_states, cond_embeds = encode_prompt(
                    text_encoder, batch["prompt_input_ids"], args.text_encoder_architecture
                )

            # Train Step
            with accelerator.accumulate(model):
                codebook_size = accelerator.unwrap_model(model).config.codebook_size
                img_ids = _prepare_latent_image_ids(input_ids.shape[-2], input_ids.shape[-1],
                                                    input_ids.device, input_ids.dtype,
                                                    up_sample=accelerator.unwrap_model(model).down_block.downsample is None)

                if not isinstance(text_encoder, list):
                    txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device = input_ids.device, dtype = input_ids.dtype)
                else:
                    txt_ids = torch.zeros(get_text_encoder_length(args.text_encoder_architecture, return_main=True),3).to(device = input_ids.device, dtype = input_ids.dtype)

                if args.train_edit_model:
                    reference_image_ids = _prepare_latent_image_ids(reference_image_hidden_states.shape[-2],
                                                        reference_image_hidden_states.shape[-1],
                                                        reference_image_hidden_states.device,
                                                        reference_image_hidden_states.dtype,
                                                        up_sample=accelerator.unwrap_model(model).down_block.downsample is None)

                logits = (
                    model(
                        hidden_states=input_ids, # should be (batch size, channel, height, width)
                        encoder_hidden_states=encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                        micro_conds=micro_conds, # 
                        pooled_projections=cond_embeds, # should be (batch_size, projection_dim)
                        img_ids = img_ids,
                        txt_ids = txt_ids,
                        timestep = mask_prob if args.time_conditioning == "mask_ratio" else mask_prob * 1000,
                        mask_ratio=mask_prob,
                        sparse_target=args.sparse_target,
                        target_edit_mask=token_edit_mask if args.sparse_target else None,
                        # ----- edit
                        **({
                            'reference_image_hidden_states': reference_image_hidden_states,
                            'reference_image_ids': reference_image_ids,
                            'lora_part_enable': True
                        } if args.train_edit_model else {})
                    )
                    .reshape(bs, codebook_size, -1)
                    .permute(0, 2, 1)
                    .reshape(-1, codebook_size)
                )

                if args.mask_loss_reduction == "sample":
                    token_loss = F.cross_entropy(
                        logits, labels.reshape(-1), ignore_index=-100, reduction="none",
                    ).reshape(batch_size, -1)
                    supervised_per_sample = (labels.reshape(batch_size, -1) != -100).sum(dim=-1)
                    sample_loss = token_loss.sum(dim=-1) / supervised_per_sample.clamp(min=1)
                    # Avoid silently weighting high-noise or large edits more
                    # simply because they contain more supervised tokens.
                    loss = sample_loss.sum() / (supervised_per_sample > 0).sum().clamp(min=1)
                else:
                    loss = F.cross_entropy(
                        logits, labels.reshape(-1), ignore_index=-100, reduction="mean",
                    )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                noise_stats = torch.stack((
                    mask_prob.mean(),
                    (mask_prob <= args.condition_cache_threshold).float().mean(),
                ))
                avg_noise_stats = accelerator.gather(noise_stats.unsqueeze(0)).mean(dim=0)
                avg_masking_rate = avg_noise_stats[0]
                avg_low_noise_fraction = avg_noise_stats[1]

                accelerator.backward(loss)

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema.step(model.parameters())

                if (global_step + 1) % args.logging_steps == 0:
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "low_noise_sample_fraction": avg_low_noise_fraction.item(),
                    }
                    if args.edit_only_diffusion:
                        logs.update({
                            "edit_token_ratio": edit_token_ratio.item(),
                            "generation_token_ratio": generation_token_ratio.item(),
                            "halo_supervised_token_count": halo_supervised_count.item(),
                            "corrupted_edit_token_ratio": corrupted_edit_token_ratio.item(),
                            "supervised_token_count": supervised_token_count.item(),
                            "preserve_supervised_token_count": preserve_supervised_count.item(),
                        })
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                        + (f" Edit ratio: {edit_token_ratio.item():0.4f} "
                           f"Corrupted edit ratio: {corrupted_edit_token_ratio.item():0.4f} "
                           f"Supervised: {supervised_token_count.item()} "
                           f"Preserve supervised: {preserve_supervised_count.item()}"
                           if args.edit_only_diffusion else "")
                    )

                if (global_step + 1) % args.checkpointing_steps == 0:
                    save_checkpoint(args, accelerator, global_step + 1, logger)

                if args.validation_steps > 0 and (global_step + 1) % args.validation_steps == 0:
                    visualize_training_step(global_step + 1)

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break

            if accelerator.sync_gradients:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        # End for

    # Also save the last trained state when the run ends between intervals.
    visualize_training_step(global_step)
    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(args, accelerator, global_step, logger)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = unwrap_training_model(model)
        if args.use_ema:
            ema.copy_to(model.parameters())
        if args.use_lora:
            save_model_hook([model], [None], args.output_dir)
        else:
            model.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main(parse_args())
