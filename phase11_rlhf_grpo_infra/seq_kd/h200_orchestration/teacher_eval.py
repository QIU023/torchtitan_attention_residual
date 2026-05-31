"""Teacher (Qwen3-VL-30B-AWQ) eval on the SAME benchmark records/prompts/scorer
as the student, for an apples-to-apples baseline/student/teacher triangle.

CRITICAL: max_pixels=1003520 (same as distillation generation) — full-res would
inflate the teacher and break comparability.

Runs in the vLLM venv. Reuses each score module's _load_records/_prompt_builder/
_image_loader/_score so the teacher is parsed identically to the student.
Usage: teacher_eval.py <bench: mmbench|gqa|pope> [LIMIT]
"""
import os, sys, json
os.environ.setdefault("HF_HOME", "/home/.hf_home")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# eval_data dirs (same as student re-score)
os.environ.setdefault("GQA_DIR", "/home/.hf_home/eval_data/gqa")
os.environ.setdefault("MMB_DIR", "/home/.hf_home/eval_data/mmbench/en")
os.environ.setdefault("POPE_DIR", "/home/.hf_home/eval_data/pope")
sys.path.insert(0, "/home/torchtitan_attention_residual")

# Shim: score modules do `from eval_benchmarks.eval_common import run_benchmark`,
# and eval_common imports torch + train_mm + torchtitan (the STUDENT inference path,
# not needed for teacher eval). Inject a stub so the import succeeds without pulling
# the whole torchtitan stack into the vLLM venv.
import types
_ec = types.ModuleType("phase5_vlm_multimodal_sft.eval_benchmarks.eval_common")
_ec.run_benchmark = lambda *a, **k: None
sys.modules["phase5_vlm_multimodal_sft.eval_benchmarks.eval_common"] = _ec

bench = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MOD = {
    "mmbench": "phase5_vlm_multimodal_sft.eval_benchmarks.score_mmbench",
    "gqa": "phase5_vlm_multimodal_sft.eval_benchmarks.score_gqa",
    "pope": "phase5_vlm_multimodal_sft.eval_benchmarks.score_pope",
}[bench]
m = __import__(MOD, fromlist=["_load_records", "_prompt_builder", "_image_loader", "_score"])

# load records (handle both return styles: list, or (records, imgs))
lr = m._load_records(limit=limit or None)
if isinstance(lr, tuple):
    records, _imgs = lr[0], lr[1]
    # gqa: _image_loader reads from module global _REC_IMG_CACHE; populate it
    if hasattr(m, "_REC_IMG_CACHE") and isinstance(_imgs, dict):
        m._REC_IMG_CACHE.update(_imgs)
else:
    records = lr
print(f"[teacher_eval:{bench}] {len(records)} records", flush=True)

# gt extractor (varies by bench: 'gt' for mmbench, 'answer'/'label' for others)
def get_gt(r):
    for k in ("gt", "answer", "label", "gold"):
        if k in r:
            return r[k]
    return None

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
MODEL = sorted(__import__("glob").glob(
    "/home/.hf_home/hub/models--QuantTrio--Qwen3-VL-30B-A3B-Instruct-AWQ/snapshots/*/"))[0]
import vllm.model_executor.layers.rotary_embedding.common as _rc
_orig = _rc.find_spec
_rc.find_spec = lambda n, *a, **k: None if n == "flash_attn" else _orig(n, *a, **k)
llm = LLM(model=MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.90,
          max_model_len=8192, limit_mm_per_prompt={"image": 1},
          mm_processor_kwargs={"max_pixels": 1003520}, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=128)

OUT = f"/home/seqkd_overnight/teacher_eval/{bench}"
os.makedirs(OUT, exist_ok=True)
fout = open(f"{OUT}/preds_rank0.jsonl", "w")
CHUNK = 256
preds = []
for c0 in range(0, len(records), CHUNK):
    chunk = records[c0:c0 + CHUNK]
    vinputs, metas = [], []
    for r in chunk:
        try:
            img = m._image_loader(r)
        except Exception:
            continue
        q = m._prompt_builder(r)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        vinputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
        metas.append(r)
    outs = llm.generate(vinputs, sp)
    for o, r in zip(outs, metas):
        rec = {"id": str(r.get("id", "")), "pred": o.outputs[0].text.strip(), "gt": get_gt(r)}
        preds.append(rec)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fout.flush()
    print(f"[teacher_eval:{bench}] {len(preds)}/{len(records)}", flush=True)
fout.close()

res = m._score(preds)
json.dump(res, open(f"{OUT}/result.json", "w"), indent=2)
print(f"[teacher_eval:{bench}] RESULT", json.dumps({k: res[k] for k in res if k in
      ("accuracy", "f1", "primary_score", "n_scored", "parse_rate")}), flush=True)
