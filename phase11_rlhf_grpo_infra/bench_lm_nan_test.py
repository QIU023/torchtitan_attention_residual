"""LM-only NaN/garbage test for Kimi-Linear AttnRes at low precision.

Boots SGLang Engine on the LM (no VL overlay), generates a few text-only
completions, checks for NaN / garbage / empty outputs. Run one dtype per
invocation (Engine instances don't compose cleanly on same GPUs).

Usage:
    python phase11_rlhf_grpo_infra/bench_lm_nan_test.py \
        --model-path phase10_ckpt_dcp_to_hf/hf_step9700_paperalign_C \
        --dtype bfloat16
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

# Match production env knobs from bench_inference_dtype.py
os.environ.setdefault("ATTNRES_MLA_FP32_FALLBACK", "1")
os.environ.setdefault(
    "SGLANG_FP8_IGNORED_LAYERS",
    "attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts",
)

# Pre-import LM-only overlay (no VL)
try:
    import sglang.srt.models.attn_res_overlay  # noqa: F401
except ImportError:
    pass

_GARBAGE_PREFIX = re.compile(r"^[^A-Za-z0-9\s]{3,}")


def check_output(text: str) -> tuple[bool, str]:
    """Returns (is_bad, reason)."""
    if not text:
        return True, "empty"
    if "nan" in text.lower() or "NaN" in text:
        return True, "literal nan in output"
    if "�" in text:  # replacement char = bad UTF-8 decode
        return True, "replacement char (bad utf-8)"
    if _GARBAGE_PREFIX.match(text):
        return True, f"garbage prefix {text[:6]!r}"
    # Check for any non-printable chars besides whitespace
    nonprint = sum(1 for c in text if not (c.isprintable() or c.isspace()))
    if nonprint > len(text) * 0.1:
        return True, f"{nonprint}/{len(text)} non-printable"
    return False, ""


PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Once upon a time, there was a",
    "Python is a programming language that",
    "The best way to learn machine learning is",
    "In the year 2050, humans will",
    "def fibonacci(n):",
    "Q: What is the speed of light?\nA:",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--quantization", default="",
                   help="'', 'fp8', 'fp8_e4m3' etc.")
    p.add_argument("--attention-backend", default="flashinfer")
    p.add_argument("--decode-attention-backend", default="torch_native",
                   help="torch_native avoids the flashinfer-MLA NaN trap on AttnRes.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    from sglang.srt.entrypoints.engine import Engine

    engine_kwargs = dict(
        model_path=str(args.model_path),
        tp_size=1,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="warning",
        disable_radix_cache=True,
    )
    decode_backend = args.decode_attention_backend or args.attention_backend
    engine_kwargs["decode_attention_backend"] = decode_backend
    if decode_backend == "torch_native" or args.attention_backend == "torch_native":
        engine_kwargs["disable_cuda_graph"] = True
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization

    label = f"dtype={args.dtype} quant={args.quantization or 'none'} decode={decode_backend}"
    print(f"[lm-nan] config: {label}", flush=True)
    print(f"[lm-nan] booting Engine on {args.model_path}", flush=True)
    t0 = time.perf_counter()
    e = Engine(**engine_kwargs)
    boot_s = time.perf_counter() - t0
    print(f"[lm-nan] engine ready in {boot_s:.1f}s", flush=True)

    n_total = len(PROMPTS)
    n_bad = 0
    print(f"[lm-nan] running {n_total} prompts (greedy, max_new={args.max_new_tokens})", flush=True)
    for i, prompt in enumerate(PROMPTS):
        t1 = time.perf_counter()
        out = e.generate(
            prompt=prompt,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
                "stop": [],
            },
        )
        dt = time.perf_counter() - t1
        text = out.get("text", "").strip()
        meta = out.get("meta_info", {}) or {}
        n_tok = (
            meta.get("completion_tokens")
            or meta.get("output_tokens")
            or len(text.split())
        )
        is_bad, reason = check_output(text)
        mark = "-" if is_bad else "+"
        if is_bad:
            n_bad += 1
        print(
            f"[lm-nan] {mark} p{i:02d} dt={dt:.2f}s tok={n_tok} "
            f"prompt={prompt!r} → text={text[:100]!r}"
            + (f" [BAD:{reason}]" if is_bad else ""),
            flush=True,
        )

    print(f"\n[lm-nan] SUMMARY: {label}")
    print(f"[lm-nan]   bad/total: {n_bad}/{n_total}")
    print(f"[lm-nan]   boot: {boot_s:.1f}s")
    rc = 0 if n_bad == 0 else 1
    print(f"[lm-nan]   exit: {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
