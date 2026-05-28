"""Stage B smoke: load llava-hf/llama3-llava-next-8b-hf (HF transformers), verify
(1) it loads + generates a coherent caption on a known image, and
(2) we can extract response-position logits with shape [T_resp, V_teacher >= 128256]
    aligned to Llama-3 token ids — the bytes the OPD step will consume.
"""
import os, torch
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

TEACHER = "llava-hf/llama3-llava-next-8b-hf"
IMG = "/workspace/gqa_rl/images/n161313.jpg"   # known GQA image (Q: "is it overcast?")


def main():
    from PIL import Image
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    print("[B] loading processor + model (bf16 on cuda:0)...", flush=True)
    proc = LlavaNextProcessor.from_pretrained(TEACHER)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    print(f"[B] teacher vocab (lm_head out): {model.lm_head.out_features}", flush=True)
    print(f"[B] tokenizer vocab: {len(proc.tokenizer)}", flush=True)

    image = Image.open(IMG).convert("RGB")
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe the image briefly."},
    ]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)
    # processor returns int tensors for ids; keep dtype only for pixel_values
    inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()

    # (1) generate caption — sanity
    print("[B] generating caption (greedy, 40 tok)...", flush=True)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    new_ids = gen[0, inputs["input_ids"].shape[1]:]
    caption = proc.decode(new_ids, skip_special_tokens=True)
    print(f"[B] CAPTION: {caption!r}", flush=True)

    # (2) score gate: forward (prompt + a fixed response) → response logits shape
    response = "A red double-decker bus drives down a city street."
    resp_ids = proc.tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids
    full_text = prompt + response
    full = proc(images=image, text=full_text, return_tensors="pt").to(model.device, torch.bfloat16)
    full["input_ids"] = full["input_ids"].long()
    if "attention_mask" in full:
        full["attention_mask"] = full["attention_mask"].long()
    with torch.no_grad():
        out = model(**full)
    logits = out.logits[0]            # [T, V]
    T_resp = resp_ids.shape[1]
    response_logits = logits[-T_resp:, :]
    print(f"[B] full logits shape: {tuple(logits.shape)}", flush=True)
    print(f"[B] response_logits:   {tuple(response_logits.shape)}  (T_resp={T_resp}, V>=128256 needed)", flush=True)
    assert response_logits.shape[-1] >= 128256, "teacher vocab < 128256 — alignment broken"
    print("[B] SMOKE PASSED — teacher loads, captions sanely, response-logits shape is correct", flush=True)


if __name__ == "__main__":
    main()
