"""GQA testdev greedy eval of the OPD TEACHER (llava-hf/llama3-llava-next-8b-hf).

Gives PR14's evidence triangle:
    SFT baseline  (Kimi-AttnRes 447M after stage-2 SFT)  : 12.3%
    OPD student   (this run, see eval_summary.md)         : TBD
    Teacher       (LLaVA-NeXT-8B, this script)            : TBD ← what we measure here

Without the teacher upper-bound we can't tell whether the OPD student is
(a) plateauing because the teacher itself can't do GQA, or
(b) plateauing because cross-VLM distillation transfer is lossy.

Reuses TeacherScorer (same load path as the OPD training) so the teacher
state matches what the student was distilled from. Greedy generation,
same answer-grading rule (gold-content-anywhere) as gqa_eval.py.

Run AFTER the D-5 orchestrator completes — running while the trainer's
actor holds the teacher on cuda:5-7 would either OOM (another teacher
copy on the same cards) or contend for the HF model weights.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time

import torch

# Use the same TeacherScorer code path that the training run uses.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from teacher_scorer import TeacherScorer  # noqa: E402

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

GQA = os.environ.get("GQA_JSON", "/workspace/gqa_rl/gqa_testdev.json")
IMGDIR = os.environ.get("GQA_IMG_DIR", "/workspace/gqa_rl")
N = int(os.environ.get("N_EVAL", "300"))

_ART = {"a", "an", "the"}


def _norm(t: str) -> str:
    toks = [x for x in re.findall(r"[a-z0-9']+", t.lower()) if x not in _ART]
    return " ".join(toks)


def _correct(completion: str, gold: str) -> bool:
    """Same grading as gqa_eval.py: gold phrase appears, or all gold tokens present."""
    g = _norm(gold)
    full = _norm(completion)
    toks = full.split()
    if not g:
        return False
    return (" " + g + " ") in (" " + full + " ") or set(g.split()).issubset(set(toks))


def main():
    print(f"[teacher-eval] loading teacher (max_memory across cuda:5-7)…", flush=True)
    # Same max_memory layout as the OPD training (cuda:5-7 = logical 0-2
    # when this script runs standalone with full visibility).
    # In a standalone process CUDA_VISIBLE_DEVICES defaults to "all 8 cards".
    # We pin teacher to phys cuda:5-7 to avoid touching the student GPU if
    # that's reused for other work, and match the training-time layout.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
        # Within this process: logical cuda:0/1/2 = phys 5/6/7
        max_mem = {0: "12GiB", 1: "12GiB", 2: "12GiB"}
    else:
        # Caller has constrained visibility; let HF auto-place.
        max_mem = None

    scorer = TeacherScorer(
        max_memory=max_mem,
        dtype=torch.bfloat16,
    )
    print(f"[teacher-eval] loaded on {scorer.device}", flush=True)

    recs = json.load(open(GQA))
    random.Random(0).shuffle(recs)
    recs = recs[:N]
    print(f"[teacher-eval] {len(recs)} GQA testdev questions, greedy max_new=16",
          flush=True)

    # Build chat-template prompt once (parametrised by question per record).
    proc = scorer.proc
    SYS_INSTR = ("You are a helpful vision assistant. Answer the question about "
                 "the image in 1-3 words.")

    correct = 0
    samples = []
    t0 = time.time()
    for i, r in enumerate(recs):
        img_path = os.path.join(IMGDIR, r["image"])
        question = r["question"]
        gold = r["answer"]

        # Wrap with LLaVA-NeXT chat template.
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"{SYS_INSTR}\n\n{question}"},
        ]}]
        prompt = proc.apply_chat_template(conv, add_generation_prompt=True)

        # Load image (no data URL handling needed — these are local files).
        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        inputs = proc(images=image, text=prompt, return_tensors="pt").to(
            scorer.device, scorer.dtype,
        )
        inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].long()

        with torch.no_grad():
            gen = scorer.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=proc.tokenizer.eos_token_id,
            )
        new_ids = gen[0, inputs["input_ids"].shape[1]:]
        answer = proc.decode(new_ids, skip_special_tokens=True).strip()

        ok = _correct(answer, gold)
        correct += int(ok)
        if i < 12:
            samples.append((question, gold, answer[:60], ok))
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(recs) - i - 1) / rate
            print(f"[teacher-eval] {i+1}/{len(recs)} acc={correct/(i+1):.3f} "
                  f"({rate:.1f} q/s, ETA {eta/60:.1f} min)",
                  flush=True)

    final = correct / len(recs)
    print(f"\n===== TEACHER GQA testdev greedy accuracy: "
          f"{correct}/{len(recs)} = {final:.4f} =====", flush=True)
    print("samples (Q | gold | pred | ok):", flush=True)
    for q, g, p, ok in samples:
        print(f"  [{'OK' if ok else 'XX'}] {q[:50]!r} | {g!r} | {p!r}", flush=True)


if __name__ == "__main__":
    main()
