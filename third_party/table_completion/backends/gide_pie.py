"""GIDE released pipeline adapted to PIE-Bench's paired captions and subjects."""

import json
import re
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / "third_party/GIDE"
LUMINA = ROOT / "third_party/Lumina-DiMOO/models/Lumina-DiMOO"
SAM3 = REPO / "models/sam3/sam3.pt"
SAM2 = REPO / "models/sam2-hiera-large/sam2_hiera_large.pt"
PIE = ROOT / "data/PIE-Bench/mapping_file.json"


def subjects(row):
    words = (row.get("blended_word") or "").split()
    if len(words) == 2:
        return words[0], words[1]
    source_brackets = re.findall(r"\[([^]]+)\]", row["original_prompt"])
    target_brackets = re.findall(r"\[([^]]+)\]", row["editing_prompt"])
    if source_brackets and target_brackets:
        return source_brackets[0], target_brackets[0]
    if row["editing_type_id"] == "8":
        return "background", "background"
    if row["editing_type_id"] == "9":
        return "scene", "scene"
    instruction = row["editing_instruction"]
    match = re.search(r"(?:the|a|an) ([\w -]{2,45}?)(?: with | to | from | into | on | in |\.|$)", instruction, re.I)
    phrase = match.group(1).strip() if match else "object"
    return phrase, phrase


class Backend:
    steps = 64
    resolution = 512

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(REPO / "evaluation"))
        sys.path.insert(0, str(REPO / "sam3_repo"))
        sys.path.insert(0, str(REPO / "sam2_repo"))
        from transformers import AutoTokenizer
        from diffusers import VQModel
        from model import LLaDAForMultiModalGeneration
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from evaluate_ours import get_editing_args, get_masking_args
        from t2i_generation import t2i_generate_image

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(LUMINA), trust_remote_code=True, local_files_only=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(
            str(LUMINA), torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True,
        ).eval()
        self.vq = VQModel.from_pretrained(str(LUMINA), subfolder="vqvae", local_files_only=True).to(self.device).eval()
        sam2_model = build_sam2(
            "configs/sam2/sam2_hiera_l.yaml", ckpt_path=str(SAM2), device=str(self.device),
        )
        self.sam2 = SAM2ImagePredictor(sam2_model)
        sam3_model = build_sam3_image_model(
            checkpoint_path=str(SAM3), load_from_HF=False, device=str(self.device),
        )
        self.sam3 = Sam3Processor(sam3_model, device=str(self.device))
        self.get_editing_args = get_editing_args
        self.get_masking_args = get_masking_args
        self.edit = t2i_generate_image
        self.metadata = json.loads(PIE.read_text(encoding="utf-8"))

    def prepare_source(self, source: Image.Image) -> Image.Image:
        return source.resize((512, 512), Image.Resampling.BICUBIC)

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(self, source: Image.Image, task: dict, seed: int) -> Image.Image:
        if task["key"] not in self.metadata:
            raise ValueError("GIDE released code requires source/target prompt metadata")
        row = self.metadata[task["key"]]
        source_prompt = row["original_prompt"].replace("[", "").replace("]", "")
        target_prompt = row["editing_prompt"].replace("[", "").replace("]", "")
        source_subject, target_subject = subjects(row)
        edit_type = {"2": "add", "3": "remove"}.get(row["editing_type_id"], "replace")
        with tempfile.TemporaryDirectory(prefix="gide-pie-", dir="/tmp") as scratch:
            scratch_path = Path(scratch)
            cache = str(scratch_path / "cache")
            output = str(scratch_path / "result.jpg")
            if edit_type == "remove":
                args = self.get_editing_args(source, source_prompt, target_prompt, cache,
                                             output, lamda1_ratio=0.2)
                args.seed = seed
                return self.edit(
                    args, self.tokenizer, self.model, self.vq,
                    if_edit=True, if_inversion=False, device=str(self.device),
                    use_attention_control=True, sam3_processor=self.sam3,
                    subject_token=source_subject, edit_type=edit_type,
                    sam2_predictor=self.sam2, mask_mode="sam3_text",
                ).convert("RGB")
            mask_args = self.get_masking_args(source, source_prompt, cache, str(scratch_path / "masking_map.png"))
            mask_args.seed = seed
            latents_mask = self.edit(
                mask_args, self.tokenizer, self.model, self.vq,
                if_edit=False, if_inversion=True, device=str(self.device),
                use_attention_control=True, sam3_processor=self.sam3,
                subject_token=source_subject, edit_type=edit_type,
                sam2_predictor=self.sam2, mask_mode="sam3_text",
            )
            args = self.get_editing_args(source, target_prompt, target_prompt, cache,
                                         output, lamda1_ratio=0.2,
                                         target_subject_token=target_subject)
            args.seed = seed
            args.inpaint_ms_minus_mt = not (source_subject == target_subject and edit_type == "replace")
            return self.edit(
                args, self.tokenizer, self.model, self.vq,
                if_edit=True, if_inversion=False, device=str(self.device),
                use_attention_control=True, sam3_processor=self.sam3,
                subject_token=target_subject, latents_mask=latents_mask,
                edit_type=edit_type, sam2_predictor=self.sam2,
            ).convert("RGB")
