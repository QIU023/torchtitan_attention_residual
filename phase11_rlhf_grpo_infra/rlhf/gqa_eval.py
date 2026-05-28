#!/usr/bin/env python3
"""GREEDY GQA accuracy eval of the LLaVA-SFT'd 447M Kimi-AttnRes VLM (stage2
step-5200, DCP->HF). Grounds the "how good is this model" question with a real
exact-match number (vs the temp-0.8 RL rollouts which degenerate). torch_native
decode to avoid the bf16-MLA NaN. Runs N questions, prints accuracy + samples.
NOTE: Engine spawns subprocs -> all work under __main__."""
import os, json, base64, re, random

os.environ.setdefault("ATTNRES_MLA_FP32_FALLBACK", "1")
os.environ.setdefault("SGLANG_DISABLE_SHM_MM", "1")
os.environ.setdefault("SGLANG_FP8_IGNORED_LAYERS",
                      "attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts")

HF = os.environ.get(
    "HF_MODEL_PATH",
    "/workspace/torchtitan_attention_residual/phase11_rlhf_grpo_infra/hf/stage2_447m_step5200",
)
GQA = os.environ.get("GQA_JSON", "/workspace/gqa_rl/gqa_testdev.json")
IMGDIR = os.environ.get("GQA_IMG_DIR", "/workspace/gqa_rl")
N = int(os.environ.get("N_EVAL", "500"))

_ART = {"a", "an", "the"}
def _norm(t):
    toks = [x for x in re.findall(r"[a-z0-9']+", t.lower()) if x not in _ART]
    return " ".join(toks)

def _correct(completion, gold):
    g = _norm(gold); full = _norm(completion); toks = full.split()
    if not g:
        return False
    # gold answer present (phrase or all tokens) — matches the RL reward's notion
    return (" " + g + " ") in (" " + full + " ") or set(g.split()).issubset(set(toks))


def main():
    import sglang.srt.configs.kimi_attn_res_vl  # noqa: F401
    from sglang.srt.entrypoints.engine import Engine
    import asyncio

    recs = json.load(open(GQA))
    random.Random(0).shuffle(recs)
    recs = recs[:N]
    print(f"[eval] {len(recs)} GQA questions; booting engine...", flush=True)
    eng = Engine(model_path=HF, skip_tokenizer_init=False, tp_size=1, dtype="bfloat16",
                 mem_fraction_static=0.6, attention_backend="flashinfer",
                 decode_attention_backend="torch_native", linear_attn_backend="triton",
                 disable_cuda_graph=True, log_level="error", base_gpu_id=0)
    SYS = ("You are a helpful vision assistant. Answer the question about the image "
           "with a single short word or phrase.")
    sp = {"temperature": 0.0, "max_new_tokens": 16}

    async def run():
        correct = 0; samples = []
        for i, r in enumerate(recs):
            with open(os.path.join(IMGDIR, r["image"]), "rb") as f:
                durl = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
            prompt = f"{SYS}\n\n<image>\nUser: {r['question']}\nAssistant:"
            out = await eng.async_generate(prompt, image_data=durl, sampling_params=sp)
            txt = (out[0] if isinstance(out, list) else out)
            txt = txt.get("text", "") if isinstance(txt, dict) else str(txt)
            ok = _correct(txt, r["answer"])
            correct += int(ok)
            if i < 12:
                samples.append((r["question"], r["answer"], txt.strip()[:60], ok))
            if (i + 1) % 100 == 0:
                print(f"[eval] {i+1}/{len(recs)} running acc={correct/(i+1):.3f}", flush=True)
        print(f"\n===== GQA testdev greedy accuracy: {correct}/{len(recs)} = {correct/len(recs):.4f} =====")
        print("samples (Q | gold | pred | ok):")
        for q, g, p, ok in samples:
            print(f"  [{'OK' if ok else 'XX'}] {q[:45]!r} | {g!r} | {p!r}")
    asyncio.run(run())
    eng.shutdown()


if __name__ == "__main__":
    main()
