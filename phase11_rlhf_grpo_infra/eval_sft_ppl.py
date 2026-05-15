"""Quick PPL/loss eval for the SFT VLM ckpt on its training distribution.

Loads vlm_sft_3ep via SGLang, samples N captions from LLaVA-Pretrain, and
computes per-token log-prob of the gold caption conditioned on the
image + prompt. Reports mean NLL and PPL. A healthy SFT'd model on its
own training distribution should hit NLL < 0.7 (PPL < 2.0) for the
caption tokens. Anything > 3 PPL means SFT didn't converge or the ckpt
is corrupted.

This is the "is the model actually SFT'd?" sanity check that yesterday's
broken qualitative gate (OR vs AND) was supposed to catch.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

try:
    import sglang.srt.configs.kimi_attn_res_vl  # noqa: F401
    import sglang.srt.models.attn_res_vl_overlay  # noqa: F401
except ImportError:
    pass


_SYSTEM_PROMPT = (
    "You are a helpful vision assistant. Describe the image in one short\n"
    "sentence (5 to 30 words). Begin with a capital letter and end with a\n"
    "period."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument(
        "--json-path",
        type=Path,
        default=Path("/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"),
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/.hf_home/LLaVA-Pretrain"),
    )
    p.add_argument("--num-samples", type=int, default=100)
    args = p.parse_args()

    rng = random.Random(42)
    with open(args.json_path) as f:
        records = json.load(f)
    # Records have fields like {"image": "00000/...jpg", "conversations": [...]}.
    # The "gpt" turn (assistant) is the gold caption for LLaVA-Pretrain.
    selected = rng.sample(records, args.num_samples)

    from sglang.srt.entrypoints.engine import Engine

    print(f"[ppl] booting SGLang on {args.model_path}")
    t0 = time.perf_counter()
    e = Engine(
        model_path=str(args.model_path),
        tp_size=1,
        dtype="bfloat16",
        attention_backend="flashinfer",
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        disable_radix_cache=True,
    )
    print(f"[ppl] engine ready in {time.perf_counter()-t0:.1f}s")

    nlls = []
    n_failed = 0
    for i, rec in enumerate(selected):
        img_rel = rec.get("image")
        if img_rel is None:
            continue
        img_path = args.images_dir / img_rel
        if not img_path.exists():
            n_failed += 1
            continue
        convs = rec.get("conversations") or []
        # Find first gpt/assistant turn.
        gold = None
        for c in convs:
            if c.get("from") in ("gpt", "assistant"):
                gold = c.get("value") or ""
                break
        if not gold:
            n_failed += 1
            continue
        # Strip <image> tags etc from the user prompt; use a generic caption ask.
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n<image>\nUser: Describe the image briefly.\n"
            f"Assistant: {gold}"
        )
        # Use SGLang's generate with max_new_tokens=1 to force a forward pass
        # and request input log probs (covers the gold tokens which are part
        # of the prompt).
        out = e.generate(
            prompt=prompt,
            image_data=str(img_path),
            sampling_params={"temperature": 0.0, "max_new_tokens": 1, "stop": []},
            return_logprob=True,
            logprob_start_len=0,
        )
        # SGLang returns input_token_logprobs as list of (logprob, token_id, token_text).
        meta = out.get("meta_info") or {}
        in_logprobs = meta.get("input_token_logprobs") or []
        # Heuristic: skip the system+user portion and only score the gold caption
        # tokens. Approximate by taking the tail equal to ~len(gold.split()) * 2
        # tokens (rough subword expansion). For a 5-30 word caption, that's
        # ~10-60 tokens; this is good enough to compute relative NLL.
        n_gold_approx = max(8, min(60, len(gold.split()) * 2))
        tail = in_logprobs[-n_gold_approx:] if in_logprobs else []
        if not tail:
            n_failed += 1
            continue
        # Each entry is (logprob, token_id, token_text); first one in SGLang
        # may be None because the very first token has no preceding context.
        token_lps = [t[0] for t in tail if t and t[0] is not None]
        if not token_lps:
            n_failed += 1
            continue
        mean_lp = sum(token_lps) / len(token_lps)
        nll = -mean_lp
        nlls.append(nll)
        if i < 5 or i % 20 == 0:
            print(
                f"[ppl] i={i:3d} n_tok={len(token_lps):3d} mean_lp={mean_lp:+.3f} "
                f"NLL={nll:.3f} gold[:60]={gold[:60]!r}"
            )

    if not nlls:
        print(f"[ppl] FATAL: no valid samples (failed={n_failed})")
        e.shutdown()
        return 2
    mean_nll = sum(nlls) / len(nlls)
    ppl = math.exp(mean_nll)
    print(
        f"\n[ppl] N={len(nlls)} mean_NLL={mean_nll:.4f} PPL={ppl:.3f} "
        f"(failed={n_failed})"
    )
    print(
        "[ppl] Health threshold: NLL<0.7 / PPL<2.0 for healthy SFT on training "
        "distribution"
    )
    if mean_nll < 0.7:
        print("[ppl] VERDICT: HEALTHY")
    elif mean_nll < 1.5:
        print("[ppl] VERDICT: MARGINAL — undertrained")
    else:
        print("[ppl] VERDICT: BROKEN — SFT did not converge")
    e.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
