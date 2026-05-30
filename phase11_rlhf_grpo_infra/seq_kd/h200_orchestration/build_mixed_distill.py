"""Per-task-length seq-KD dataset: short/MC answers keep ORIGINAL (crisp),
long/caption answers use TEACHER rewrite. Avoids the blanket-verbose regression
that cost MMBench 9.3pp (short-answer/MC distribution got washed out).

Align orig<->teacher by (image + first human question) — 99% unique, position
order differs between the two files. Turns are matched within a conversation by
index (same human turns, only gpt answers differ).
"""
import json, re, hashlib, sys
from collections import Counter
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/torchtitan_attention_residual/torchtitan/assets/hf/Llama-3.1-8B")
ORIG = "/home/.hf_home/LLaVA-Instruct/llava_v1_5_mix665k.json"
TEACH = "/home/torchtitan_attention_residual/phase11_rlhf_grpo_infra/seq_kd/distilled_mix665k_full_OLDBOX.json"
OUT = "/home/torchtitan_attention_residual/phase11_rlhf_grpo_infra/seq_kd/distilled_mix665k_TASKMIX.json"

SHORT_TOK = 10  # original answer <= this many tokens -> keep original (crisp)

def key(r):
    img = r.get("image", "")
    h = ""
    for m in r["conversations"]:
        if m["from"] == "human":
            h = m["value"].replace("<image>", "").strip()[:200]; break
    return hashlib.md5((str(img) + "||" + h).encode()).hexdigest()

def is_short(v):
    s = v.strip()
    if re.match(r"^[A-D][\.\)]?\s*$", s):  # bare MC letter
        return True
    return len(tok(v, add_special_tokens=False)["input_ids"]) <= SHORT_TOK

print("loading...", flush=True)
orig = json.load(open(ORIG))
teach = json.load(open(TEACH))

# build teacher lookup by key; for dup keys keep first (1% ambiguous -> orig fallback)
kt = {}
kc = Counter()
for r in teach:
    k = key(r); kc[k] += 1
    if k not in kt: kt[k] = r

out = []
stats = {"keep_orig": 0, "use_teach": 0, "no_teacher_match": 0, "dup_key_fallback": 0, "rows": 0}
for r in orig:
    k = key(r)
    trow = kt.get(k)
    ambiguous = kc[k] > 1
    new_convs = []
    for i, m in enumerate(r["conversations"]):
        if m["from"] != "gpt":
            new_convs.append(m); continue
        # decide per gpt turn
        if is_short(m["value"]):
            new_convs.append(m); stats["keep_orig"] += 1          # crisp original
        elif trow is not None and not ambiguous and i < len(trow["conversations"]) \
                and trow["conversations"][i]["from"] == "gpt":
            new_convs.append(trow["conversations"][i]); stats["use_teach"] += 1   # teacher rewrite
        else:
            new_convs.append(m)                                    # fallback: original
            stats["no_teacher_match" if trow is None else "dup_key_fallback"] += 1
    rec = {"id": r.get("id"), "conversations": new_convs}
    if "image" in r: rec["image"] = r["image"]
    out.append(rec); stats["rows"] += 1

json.dump(out, open(OUT, "w"), ensure_ascii=False)
gpt_total = stats["keep_orig"] + stats["use_teach"] + stats["no_teacher_match"] + stats["dup_key_fallback"]
print(f"rows={stats['rows']} -> {OUT}", flush=True)
print(f"gpt turns: keep_orig(short/MC)={stats['keep_orig']} ({stats['keep_orig']/gpt_total*100:.0f}%)  "
      f"use_teacher(long)={stats['use_teach']} ({stats['use_teach']/gpt_total*100:.0f}%)  "
      f"orig_fallback={stats['no_teacher_match']+stats['dup_key_fallback']} ({(stats['no_teacher_match']+stats['dup_key_fallback'])/gpt_total*100:.0f}%)", flush=True)
