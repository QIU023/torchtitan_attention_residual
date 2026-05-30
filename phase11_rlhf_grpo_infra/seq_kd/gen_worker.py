"""seq-KD teacher generation worker — ONE single-GPU vLLM replica (TP=1).

Reads a shard of llava_v1_5_mix665k.json, regenerates every assistant ("gpt")
turn with the Qwen3-VL teacher conditioned on the ORIGINAL preceding dialogue
(human turns + original gpt turns), and writes the conversation back out with
teacher responses substituted — same schema as the input, so the existing
LlavaInstructSFTDataset can consume it directly.

Why condition on ORIGINAL context (not teacher's own prior turns): keeps every
turn independent → the whole shard flattens into one big batch, and avoids
teacher drift compounding across a multi-turn dialogue.

5090 has no NVLink → we do NOT use tensor parallelism. Each GPU runs a full
AWQ-4bit copy (TP=1); the launcher starts 8 of these on 8 GPUs (data parallel),
zero inter-GPU comms.

Resumable: skips conversations whose id already appears in the output JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Run the v1 EngineCore in-process (TP=1) so our in-process flash_attn find_spec
# monkeypatch below actually applies to model construction (otherwise the patch
# is lost in the spawned EngineCore subprocess).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from PIL import Image


def log(*a):
    print(f"[shard{os.environ.get('SHARD_ID','?')} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_requests(convs, image_root):
    """Flatten conversations into per-gpt-turn generation requests.

    Returns (requests, index) where requests[i] = {messages, image_path} and
    index[i] = (conv_idx, turn_idx) so we can write the response back.
    """
    requests, index = [], []
    for ci, s in enumerate(convs):
        img = s.get("image")
        msgs = s["conversations"]
        history = []  # accumulated original messages as chat turns
        for ti, m in enumerate(msgs):
            role = m["from"]
            text = m["value"]
            if role == "human":
                # strip the literal <image> placeholder; image attaches via mm
                clean = text.replace("<image>", "").strip()
                history.append(("user", clean))
            else:  # gpt — this is a slot we regenerate
                # context = everything before this turn
                requests.append({"context": list(history), "image": img})
                index.append((ci, ti))
                history.append(("assistant", text))  # original for next ctx
    return requests, index


def to_vllm_messages(context, has_image):
    """context = list of (role, text). Attach image to the FIRST user turn."""
    out = []
    img_used = False
    for role, text in context:
        if role == "user" and has_image and not img_used:
            out.append({"role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": text}]})
            img_used = True
        else:
            out.append({"role": role, "content": text})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap conversations (smoke)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=2000, help="conversations per gen batch")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="vLLM scheduler concurrency cap; 0=vLLM default. "
                         "Higher saturates GPU on short-answer batches (distill "
                         "answers are p50=106 tok), big throughput win.")
    ap.add_argument("--max-pixels", type=int, default=1003520,
                    help="Qwen3-VL image-token cap: pixels/(28*28) tokens. "
                         "1003520 -> ~1280 vision tokens. Default Qwen max_pixels "
                         "is ~16x larger and lets high-res images blow past "
                         "max_model_len (one such prompt crashes the whole batch).")
    args = ap.parse_args()
    os.environ["SHARD_ID"] = str(args.shard_id)

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    # flash-attn 4 (namespace pkg, only flash_attn.cute) lacks the v2-style
    # flash_attn.ops.triton.rotary layout that vLLM's ApplyRotaryEmb.__init__
    # blindly imports when find_spec("flash_attn") is truthy. Neutralize that
    # probe in-process so vLLM uses its correct pure-torch forward_native rotary.
    import vllm.model_executor.layers.rotary_embedding.common as _rc
    _orig_find_spec = _rc.find_spec

    def _fs(name, *a, **k):
        if name == "flash_attn":
            return None
        return _orig_find_spec(name, *a, **k)

    _rc.find_spec = _fs

    # ---- load shard ----
    data = json.load(open(args.input_json))
    shard = [s for i, s in enumerate(data) if i % args.num_shards == args.shard_id]
    if args.limit:
        shard = shard[: args.limit]
    log(f"shard has {len(shard)} conversations (of {len(data)} total)")

    # ---- resume by POSITION (not id) ----
    # mix665k has heavily duplicated ids (275k dups), so id-based dedup is
    # unsafe. The shard is processed in deterministic order, one output line
    # per conversation, flushed per-chunk — so a crash leaves a clean chunk
    # boundary and #lines == #conversations already done. Skip that many.
    n_done = 0
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as fh:
            n_done = sum(1 for _ in fh)
    if n_done:
        shard = shard[n_done:]
        log(f"resume: {n_done} already written, {len(shard)} remaining")
    if not shard:
        log("nothing to do"); return

    # ---- load teacher (AWQ, single GPU) ----
    log(f"loading {args.model} (AWQ TP=1)…")
    _llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": args.max_pixels},
        trust_remote_code=True,
        enforce_eager=False,
    )
    if args.max_num_seqs > 0:
        _llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**_llm_kwargs)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    _tok = getattr(processor, "tokenizer", processor)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    # Skip prompts whose (text-only) token count alone would blow the context.
    # Image expansion adds more on top in vLLM, so reserve a generous budget.
    # Over-long prompts (huge multi-turn contexts) otherwise crash the whole
    # vLLM batch and kill the shard mid-run.
    prompt_token_cap = args.max_model_len - args.max_new_tokens - 2048
    n_skip_long = 0
    log(f"teacher loaded; prompt_token_cap={prompt_token_cap}")

    fout = open(args.out_jsonl, "a")
    t0 = time.time()
    n_conv_done = 0
    for c0 in range(0, len(shard), args.chunk):
        chunk = shard[c0 : c0 + args.chunk]
        reqs, index = build_requests(chunk, args.image_root)
        # build vLLM inputs
        vinputs, valid_index = [], []
        for r, idx in zip(reqs, index):
            has_img = r["image"] is not None
            msgs = to_vllm_messages(r["context"], has_img)
            prompt = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            if len(_tok(prompt, add_special_tokens=False)["input_ids"]) > prompt_token_cap:
                # too-long multi-turn context — skip this gpt turn (keeps the
                # original label as fallback in the write-back below)
                n_skip_long += 1
                continue
            mm = {}
            if has_img:
                ip = os.path.join(args.image_root, r["image"])
                try:
                    mm["image"] = Image.open(ip).convert("RGB")
                except Exception as e:
                    log(f"img fail {ip}: {e}"); continue
            entry = {"prompt": prompt}
            if mm:
                entry["multi_modal_data"] = mm
            vinputs.append(entry)
            valid_index.append(idx)

        outs = llm.generate(vinputs, sp)
        # map responses back per conversation
        resp_by_conv = {}
        for o, (ci, ti) in zip(outs, valid_index):
            resp_by_conv.setdefault(ci, {})[ti] = o.outputs[0].text.strip()

        for ci, s in enumerate(chunk):
            tmap = resp_by_conv.get(ci, {})
            new_convs = []
            for ti, m in enumerate(s["conversations"]):
                if m["from"] == "gpt" and ti in tmap:
                    new_convs.append({"from": "gpt", "value": tmap[ti]})
                else:
                    new_convs.append(m)
            rec = {"id": s.get("id"), "conversations": new_convs}
            if "image" in s:
                rec["image"] = s["image"]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_conv_done += len(chunk)
        rate = n_conv_done / (time.time() - t0)
        eta = (len(shard) - n_conv_done) / rate / 60 if rate > 0 else -1
        log(f"{n_conv_done}/{len(shard)} conv ({rate:.1f}/s, ETA {eta:.0f}min)")

    fout.close()
    log(f"DONE {n_conv_done} conversations in {(time.time()-t0)/60:.1f}min "
        f"(skipped {n_skip_long} over-long gpt turns → original label kept)")


if __name__ == "__main__":
    main()
