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
                 dtype: torch.dtype = torch.bfloat16):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        self.proc = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        self.model.eval()
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def score(self, image, prompt_text: str, response_text: str
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward (image, prompt+response) once; return logits + ids at response positions.

        Args:
            image: PIL.Image or path str.
            prompt_text: already-formatted prompt (e.g. via apply_chat_template).
            response_text: the student's generated text (raw decoded).

        Returns:
            response_logits: [T_resp, V_teacher] (bf16, no grad)
            response_ids:    [T_resp] long (the teacher-tokenized response ids; for labels)
        """
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
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
