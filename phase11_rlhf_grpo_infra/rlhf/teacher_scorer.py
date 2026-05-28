"""Teacher scorer for OPD: load llava-hf/llama3-llava-next-8b-hf once, expose
score(image, prompt_text, response_text) -> (response_logits[T_resp,V], response_ids[T_resp]).

V is the teacher's lm_head out (128320 for this model; we slice to 128256 in opd_loss).
The student is responsible for getting its OWN logits at the same response positions;
this scorer only owns the teacher side.
"""
from __future__ import annotations
import os
from typing import Tuple, Optional
import torch

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

TEACHER_ID = "llava-hf/llama3-llava-next-8b-hf"


class TeacherScorer:
    def __init__(self, model_id: str = TEACHER_ID, device: str = "cuda:0",
                 dtype: torch.dtype = torch.bfloat16,
                 max_memory: dict | None = None):
        """Load the teacher VLM once.

        Args:
            model_id: HF model id.
            device: target device when single-GPU (e.g. ``"cuda:0"``).
                Ignored when ``max_memory`` is set (then HF accelerate
                spreads layers across the keyed devices).
            dtype: model dtype.
            max_memory: optional accelerate ``max_memory`` dict
                (``{device_idx: "10GiB", ...}``). When provided,
                ``device_map="auto"`` + this map controls layer
                placement — used by the OPD launcher to put the
                teacher on the otherwise-idle GPUs (cuda:5-7 mapped
                into the trainer process's logical 1-3 via
                ``CUDA_VISIBLE_DEVICES=0,5,6,7``) instead of
                stacking it on top of the student's card.
        """
        # Loader routes to the right model + processor based on config:
        #   * llava-hf/llama3-llava-next-8b-hf → LlavaNextProcessor +
        #     LlavaNextForConditionalGeneration (CLIP-336, hi-res tiling)
        #   * TIGER-Lab/Mantis-8B-siglip-llama3 → LlavaProcessor (composed
        #     manually because Mantis ships no processor_config.json, so
        #     AutoProcessor falls back to bare LlamaTokenizer) +
        #     LlavaForConditionalGeneration (SigLIP-so400m-384, single-image)
        # Both processors expose ``.tokenizer`` + ``.apply_chat_template``
        # and accept ``(images=..., text=...)`` keyword call; both forwards
        # return ``logits[B, T, V]``. The rest of this class is arch-agnostic.
        from transformers import (
            AutoProcessor, AutoTokenizer, AutoImageProcessor,
            AutoModelForImageTextToText,
        )
        try:
            self.proc = AutoProcessor.from_pretrained(model_id)
            # Some VLM repos (e.g. Mantis) ship no processor_config.json, so
            # AutoProcessor silently degrades to a bare tokenizer. Detect
            # and fall back to manual LlavaProcessor composition.
            if not hasattr(self.proc, "image_processor"):
                raise RuntimeError("AutoProcessor returned a tokenizer-only object")
        except Exception:
            from transformers import LlavaProcessor, AutoConfig
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            image_processor = AutoImageProcessor.from_pretrained(model_id)
            # LlavaProcessor needs patch_size + vision_feature_select_strategy
            # to compute num_image_tokens at __call__ time. Pull from the
            # model's config.json (no separate processor_config.json on
            # Mantis-style repos).
            mcfg = AutoConfig.from_pretrained(model_id)
            patch_size = getattr(mcfg.vision_config, "patch_size", 14)
            vfs = getattr(mcfg, "vision_feature_select_strategy", "default")
            self.proc = LlavaProcessor(
                image_processor=image_processor, tokenizer=tokenizer,
                patch_size=patch_size,
                vision_feature_select_strategy=vfs,
            )
        if max_memory is not None:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=dtype,
                device_map="auto", max_memory=max_memory,
            )
            first_param_dev = next(self.model.parameters()).device
            self.device = str(first_param_dev)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device,
            )
            self.device = device
        self.model.eval()
        self.dtype = dtype

    @torch.no_grad()
    def score(self, image, prompt_text: str, response_text: str
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward (image, prompt+response) once; return logits + ids at response positions.

        Args:
            image: PIL.Image, on-disk path, or base64 ``data:image/...`` URL.
            prompt_text: either a fully chat-templated prompt (must contain
                ``<|start_header_id|>``, the LLaMA-3 marker) OR a bare user
                question text. In the latter case (the OPD trainer path),
                this method wraps it with the teacher's own
                ``apply_chat_template`` so the ``<image>`` placeholder
                lands in the right position for LLaVA-NeXT's image-feature
                splicing.
            response_text: the student's generated text (raw decoded).

        Returns:
            response_logits: [T_resp, V_teacher] (bf16, no grad)
            response_ids:    [T_resp] long (the teacher-tokenized response ids; for labels)
        """
        from PIL import Image
        if isinstance(image, str):
            if image.startswith("data:image"):
                # base64 data URL — that's what SGLang uses when the
                # launcher passes images inline (see ``_async_main_opd``
                # in run_grpo_llava_kimi.py — base64 avoids the SHM IPC
                # race that was the v16 GRPO blocker). The SGLang
                # generator stores the data URL verbatim in
                # ``Episode.image_path``, so the trainer sees it here.
                import base64
                import io
                _, b64 = image.split(",", 1)
                image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")
        # Auto-wrap bare user text with the LLaVA-NeXT chat template if
        # the caller passed a plain user question (no LLaMA-3 header
        # markers). The OPD trainer reaches us with just the user
        # prompt (decoded from Episode.prompt_token_ids with
        # skip_special_tokens=True, which strips the original <image>
        # placeholder). Wrapping here puts the teacher's own <image>
        # marker in the right spot for LlavaNext's image-token splicing.
        if "<|start_header_id|>" not in prompt_text:
            try:
                conv = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text.strip()},
                ]}]
                prompt_text = self.proc.apply_chat_template(
                    conv, add_generation_prompt=True,
                )
            except (ValueError, AttributeError):
                # Some VLMs (e.g. Mantis) ship a LlavaProcessor without a
                # chat_template — fall back to the standard Llama-3-instruct
                # format with LLaVA's <image> placeholder before the user
                # question.
                prompt_text = (
                    "<|begin_of_text|>"
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                    f"<image>\n{prompt_text.strip()}"
                    "<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
        resp_ids = self.proc.tokenizer(
            response_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]                                    # [T_resp]
        full = self.proc(images=image, text=prompt_text + response_text,
                         return_tensors="pt").to(self.device, self.dtype)
        full["input_ids"] = full["input_ids"].long()
        if "attention_mask" in full:
            full["attention_mask"] = full["attention_mask"].long()
        logits = self.model(**full).logits[0]             # [T_full, V]
        T_resp = resp_ids.shape[0]
        return logits[-T_resp:, :].detach(), resp_ids


def _smoke():
    s = TeacherScorer()
    img = "/workspace/gqa_rl/images/n161313.jpg"
    # mirror chat template the runtime uses
    conv = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe the image briefly."}]}]
    prompt = s.proc.apply_chat_template(conv, add_generation_prompt=True)
    response = "A snowboarder in mid-air against a blue sky."
    logits, ids = s.score(img, prompt, response)
    print(f"[scorer] response_logits {tuple(logits.shape)}  response_ids {tuple(ids.shape)}")
    print(f"[scorer] dtype={logits.dtype} device={logits.device} sample-logits[0,:5]={logits[0,:5].float().tolist()}")
    # second call reuses loaded model (the win)
    logits2, _ = s.score(img, prompt, "Another short response here.")
    print(f"[scorer] second-call OK shape={tuple(logits2.shape)}  (model reused)")
    print("[scorer] SMOKE PASSED")


if __name__ == "__main__":
    _smoke()
