"""Sglang Engine validation suite for Kimi-Linear AttnRes LM.

Runs many in-process tests on a single Engine boot to amortize cost.

Tests covered (V1-V15 from the gap list; V3/V4/V9/V12 handled separately):
    V1  numerical parity (logit KL vs bf16 reference)
    V2  MoE routing stability (return_routed_experts)
    V5  long context (4K/8K input)
    V6  long generation (max_new=2K/4K NaN drift)
    V7  throughput/latency profiling
    V8  batch size sweep
    V10 soak test (configurable duration)
    V11 edge cases (special tokens, unicode, empty)
    V13 streaming generation
    V14 chat template (synthetic)
    V15 constrained decoding (regex/grammar)

Usage:
    python run_validation_suite.py \\
        --model-path .../hf_step9700_paperalign_C \\
        --dtype bfloat16 [--quantization fp8] \\
        --output-dir results/bf16 \\
        --soak-seconds 60

Tier 1 (always run): V1, V5, V6, V7, V8, V11, V13
Tier 2 (--tier 2): + V2, V10, V14, V15
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("ATTNRES_MLA_FP32_FALLBACK", "1")
os.environ.setdefault(
    "SGLANG_FP8_IGNORED_LAYERS",
    "attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts",
)

try:
    import sglang.srt.models.attn_res_overlay  # noqa: F401
except ImportError:
    pass


def _result(name, status, **kwargs):
    """Structured result dict for JSON dump."""
    return {"test": name, "status": status, **kwargs}


# -----------------------------------------------------------------------------
# Prompt fixtures
# -----------------------------------------------------------------------------
TIER1_PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Once upon a time, there was a",
    "Python is a programming language that",
    "The best way to learn machine learning is",
    "In the year 2050, humans will",
    "def fibonacci(n):",
    "Q: What is the speed of light?\nA:",
]

# 100-prompt set for parity (V1)
def parity_prompts():
    seeds = [
        "The capital of",
        "In a galaxy far away,",
        "The recipe for",
        "Once upon a time,",
        "The first principle of",
        "When the rain falls,",
        "A long time ago in",
        "The secret to happiness is",
        "Scientists have discovered",
        "The most important thing",
    ]
    out = []
    for s in seeds:
        for tail in [" $X is", " $X was", " $X will be", " $X means",
                     " $X are", " $X can be", " $X requires",
                     " $X depends on", " $X involves", " $X creates"]:
            out.append(s + tail.replace("$X", "this"))
    return out  # 100 prompts


def edge_case_prompts():
    return [
        ("",                                                              "empty"),
        ("a",                                                             "single_char"),
        ("Hello",                                                         "single_word"),
        ("你好,今天天气真好。",                                            "chinese"),
        ("こんにちは、元気ですか?",                                         "japanese"),
        ("Mixed 中文 with English and 日本語 numbers 123",                "mixed_lang"),
        ("🚀 Rocket emoji + 🌟 star",                                    "emoji"),
        ("\n\n\n\n",                                                      "whitespace_only"),
        ("a" * 200,                                                       "repeated_char"),
        ("[BOS] system: you are a helpful assistant [EOS]",              "bracket_tokens"),
    ]


# -----------------------------------------------------------------------------
# Engine boot helper
# -----------------------------------------------------------------------------
def boot_engine(model_path, dtype, quantization, tp_size=1, max_total_tokens=None):
    from sglang.srt.entrypoints.engine import Engine
    kwargs = dict(
        model_path=str(model_path),
        tp_size=tp_size,
        dtype=dtype,
        attention_backend="flashinfer",
        decode_attention_backend="torch_native",
        linear_attn_backend="triton",
        trust_remote_code=True,
        log_level="error",
        disable_radix_cache=True,
        disable_cuda_graph=True,  # torch_native decode lacks cuda graph
    )
    if quantization:
        kwargs["quantization"] = quantization
    if max_total_tokens:
        kwargs["max_total_tokens"] = max_total_tokens
    t0 = time.perf_counter()
    e = Engine(**kwargs)
    boot_s = time.perf_counter() - t0
    return e, boot_s


# -----------------------------------------------------------------------------
# Output sanity check
# -----------------------------------------------------------------------------
_GARBAGE = re.compile(r"^[^A-Za-z0-9一-鿿\s]{3,}")


def is_bad(text: str) -> tuple[bool, str]:
    if "nan" in text.lower() or "NaN" in text:
        return True, "literal_nan"
    if "�" in text:
        return True, "bad_utf8"
    if text and _GARBAGE.match(text):
        return True, f"garbage_prefix({text[:6]!r})"
    return False, ""


# -----------------------------------------------------------------------------
# V1: Numerical parity (logit KL via top-k logprobs)
# -----------------------------------------------------------------------------
def v01_numerical_parity(engine, dtype_label, ref_logprobs=None):
    """Returns dict: per-prompt top-k logprobs (or None for ref dtype).

    Strategy: greedy decode 1 prompt → ask for top-20 logprobs per token →
    compare to reference (bf16) via KL divergence on softmax of those top-20.

    If ref_logprobs is None: this run IS the reference; just collect logprobs.
    Otherwise: compute KL(this || ref) per token, report mean/p95/max.
    """
    prompts = parity_prompts()
    collected = []
    for p in prompts:
        out = engine.generate(
            prompt=p,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1, "stop": []},
            return_logprob=True,
            top_logprobs_num=20,
        )
        # Output token logprob structure (sglang format):
        #   meta_info.output_top_logprobs: List[List[(logprob, token_id, str)]]
        meta = out.get("meta_info", {}) or {}
        topk = meta.get("output_top_logprobs") or []
        if topk:
            collected.append(topk[0])  # first (and only) output position
        else:
            collected.append(None)

    if ref_logprobs is None:
        return _result("V1", "ref_collected", dtype=dtype_label,
                       n_prompts=len(prompts), data=collected)

    # Compute KL(this || ref) for each prompt, using top-20 logprob distribution
    kls = []
    for cur, ref in zip(collected, ref_logprobs):
        if cur is None or ref is None:
            continue
        # cur/ref are lists of (logprob, token_id, str)
        # Restrict to the intersection of token_ids for fair compare
        ref_d = {tid: lp for lp, tid, _ in ref}
        cur_d = {tid: lp for lp, tid, _ in cur}
        common = set(ref_d) & set(cur_d)
        if len(common) < 5:
            continue
        # Renormalize to a softmax over the common token set
        ref_lp = [ref_d[t] for t in common]
        cur_lp = [cur_d[t] for t in common]
        # Convert to probabilities (sglang gives natural log)
        ref_max = max(ref_lp); cur_max = max(cur_lp)
        ref_e = [math.exp(lp - ref_max) for lp in ref_lp]
        cur_e = [math.exp(lp - cur_max) for lp in cur_lp]
        ref_z = sum(ref_e); cur_z = sum(cur_e)
        ref_p = [e / ref_z for e in ref_e]
        cur_p = [e / cur_z for e in cur_e]
        kl = sum(c * (math.log(max(c, 1e-12)) - math.log(max(r, 1e-12)))
                 for c, r in zip(cur_p, ref_p))
        kls.append(kl)
    if not kls:
        return _result("V1", "no_valid_kls", dtype=dtype_label)
    kls.sort()
    n = len(kls)
    return _result("V1", "ok", dtype=dtype_label,
                   n_compared=n,
                   kl_mean=sum(kls) / n,
                   kl_p50=kls[n // 2],
                   kl_p95=kls[int(n * 0.95)],
                   kl_max=kls[-1])


# -----------------------------------------------------------------------------
# V2: MoE routing stability
# -----------------------------------------------------------------------------
def v02_moe_routing(engine, dtype_label):
    """Run a few prompts with return_routed_experts; if it works, count
    expert usage distribution. Also check that determinism (greedy decode)
    gives identical expert routing across two runs of same prompt."""
    try:
        out1 = engine.generate(
            prompt="The recipe for chocolate cake requires:",
            sampling_params={"temperature": 0.0, "max_new_tokens": 8, "stop": []},
            return_routed_experts=True,
        )
        out2 = engine.generate(
            prompt="The recipe for chocolate cake requires:",
            sampling_params={"temperature": 0.0, "max_new_tokens": 8, "stop": []},
            return_routed_experts=True,
        )
    except Exception as exc:
        return _result("V2", "unsupported", dtype=dtype_label, error=str(exc)[:200])
    meta1 = out1.get("meta_info", {}) or {}
    routed1 = meta1.get("output_routed_experts") or meta1.get("routed_experts")
    meta2 = out2.get("meta_info", {}) or {}
    routed2 = meta2.get("output_routed_experts") or meta2.get("routed_experts")
    if routed1 is None:
        return _result("V2", "no_routing_data", dtype=dtype_label,
                       text1=out1.get("text", "")[:50])
    same = routed1 == routed2
    # Flatten to count expert usage
    flat = []
    def _flatten(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                _flatten(e)
        elif isinstance(x, int):
            flat.append(x)
    _flatten(routed1)
    if not flat:
        return _result("V2", "empty_routing", dtype=dtype_label)
    usage = {}
    for e in flat:
        usage[e] = usage.get(e, 0) + 1
    return _result("V2", "ok", dtype=dtype_label,
                   deterministic=same,
                   n_routed=len(flat),
                   unique_experts=len(usage),
                   most_used=max(usage.values()) / len(flat))


# -----------------------------------------------------------------------------
# V5: Long context
# -----------------------------------------------------------------------------
def v05_long_context(engine, dtype_label):
    """Feed long context (4K, 8K, 16K tokens approx via repeated text)
    and check no crash, no NaN."""
    results = []
    base_text = (
        "The history of mathematics begins with the development of counting "
        "systems in ancient civilizations. Sumerian and Babylonian "
        "mathematicians, working some four thousand years ago, made "
        "remarkable advances in arithmetic and algebra. ")
    targets = [(4 * 1024, "4K"), (8 * 1024, "8K"), (16 * 1024, "16K")]
    for n_chars, label in targets:
        prompt = (base_text * (n_chars // len(base_text) + 1))[:n_chars]
        try:
            t0 = time.perf_counter()
            out = engine.generate(
                prompt=prompt,
                sampling_params={"temperature": 0.0, "max_new_tokens": 16, "stop": []},
            )
            dt = time.perf_counter() - t0
            text = out.get("text", "").strip()
            bad, reason = is_bad(text)
            results.append({"label": label, "n_chars": n_chars, "dt_s": dt,
                            "text_tail": text[-80:], "bad": bad, "reason": reason})
        except Exception as exc:
            results.append({"label": label, "n_chars": n_chars,
                            "error": str(exc)[:200]})
    return _result("V5", "ok" if all("error" not in r for r in results) else "partial",
                   dtype=dtype_label, results=results)


# -----------------------------------------------------------------------------
# V6: Long generation NaN drift
# -----------------------------------------------------------------------------
def v06_long_generation(engine, dtype_label):
    """max_new=2K, 4K — check no NaN/repetition collapse."""
    results = []
    for max_new in [1024, 2048, 4096]:
        try:
            t0 = time.perf_counter()
            out = engine.generate(
                prompt="Tell me a long story about an explorer.",
                sampling_params={"temperature": 0.0, "max_new_tokens": max_new,
                                 "stop": []},
            )
            dt = time.perf_counter() - t0
            text = out.get("text", "").strip()
            meta = out.get("meta_info", {}) or {}
            n_tok = meta.get("completion_tokens") or 0
            bad, reason = is_bad(text)
            # Check for repetition collapse: max-length window with same char
            collapse = False
            if len(text) > 200:
                # find a 50-char window where 80% is the same char
                for i in range(0, len(text) - 50, 50):
                    win = text[i:i + 50]
                    if win:
                        most = max(win.count(c) for c in set(win))
                        if most > 40:
                            collapse = True
                            break
            results.append({"max_new": max_new, "n_tok_out": n_tok, "dt_s": dt,
                            "text_tail": text[-60:],
                            "tok_per_s": n_tok / dt if dt > 0 else 0,
                            "bad": bad, "reason": reason, "collapse": collapse})
        except Exception as exc:
            results.append({"max_new": max_new, "error": str(exc)[:200]})
    return _result("V6", "ok" if all("error" not in r for r in results) else "partial",
                   dtype=dtype_label, results=results)


# -----------------------------------------------------------------------------
# V7: Throughput / latency
# -----------------------------------------------------------------------------
def v07_throughput(engine, dtype_label):
    """Greedy decode 8 prompts × 64 new tokens. Report tok/s + per-prompt latency."""
    times, toks = [], []
    for p in TIER1_PROMPTS:
        t0 = time.perf_counter()
        out = engine.generate(
            prompt=p,
            sampling_params={"temperature": 0.0, "max_new_tokens": 64, "stop": []},
        )
        dt = time.perf_counter() - t0
        meta = out.get("meta_info", {}) or {}
        n_tok = meta.get("completion_tokens") or 0
        times.append(dt)
        toks.append(n_tok)
    total_t = sum(times); total_tok = sum(toks)
    return _result("V7", "ok", dtype=dtype_label,
                   total_tokens=total_tok,
                   total_seconds=total_t,
                   tok_per_s=total_tok / total_t if total_t > 0 else 0,
                   per_prompt_s_mean=total_t / len(times),
                   per_prompt_s_max=max(times))


# -----------------------------------------------------------------------------
# V8: Batch size sweep
# -----------------------------------------------------------------------------
def v08_batch_sweep(engine, dtype_label):
    """Send increasing batch sizes simultaneously; report total throughput."""
    results = []
    for bs in [1, 4, 8, 16, 32]:
        prompts = (TIER1_PROMPTS * (bs // len(TIER1_PROMPTS) + 1))[:bs]
        try:
            t0 = time.perf_counter()
            out = engine.generate(
                prompt=prompts,
                sampling_params={"temperature": 0.0, "max_new_tokens": 64, "stop": []},
            )
            dt = time.perf_counter() - t0
            outs = out if isinstance(out, list) else [out]
            total_tok = sum(
                (o.get("meta_info", {}) or {}).get("completion_tokens", 0)
                for o in outs)
            results.append({"bs": bs, "dt_s": dt, "total_tokens": total_tok,
                            "tok_per_s": total_tok / dt if dt > 0 else 0})
        except Exception as exc:
            results.append({"bs": bs, "error": str(exc)[:200]})
    return _result("V8", "ok" if all("error" not in r for r in results) else "partial",
                   dtype=dtype_label, results=results)


# -----------------------------------------------------------------------------
# V10: Soak test
# -----------------------------------------------------------------------------
def v10_soak(engine, dtype_label, duration_s=60):
    """Run continuous inference for `duration_s`, monitor NaN drift + tok/s."""
    t_start = time.perf_counter()
    n_prompts, n_bad, total_tok = 0, 0, 0
    last_text_tail = ""
    while time.perf_counter() - t_start < duration_s:
        p = TIER1_PROMPTS[n_prompts % len(TIER1_PROMPTS)]
        try:
            out = engine.generate(
                prompt=p,
                sampling_params={"temperature": 0.0, "max_new_tokens": 32, "stop": []},
            )
        except Exception as exc:
            return _result("V10", "crash", dtype=dtype_label, n_prompts=n_prompts,
                           error=str(exc)[:200])
        text = out.get("text", "").strip()
        meta = out.get("meta_info", {}) or {}
        total_tok += meta.get("completion_tokens", 0)
        bad, _ = is_bad(text)
        if bad:
            n_bad += 1
        last_text_tail = text[-40:]
        n_prompts += 1
    total_t = time.perf_counter() - t_start
    return _result("V10", "ok", dtype=dtype_label,
                   duration_s=total_t,
                   n_prompts=n_prompts,
                   n_bad=n_bad,
                   total_tokens=total_tok,
                   tok_per_s=total_tok / total_t,
                   last_text_tail=last_text_tail)


# -----------------------------------------------------------------------------
# V11: Edge cases
# -----------------------------------------------------------------------------
def v11_edge_cases(engine, dtype_label):
    results = []
    for prompt, label in edge_case_prompts():
        try:
            out = engine.generate(
                prompt=prompt,
                sampling_params={"temperature": 0.0, "max_new_tokens": 24, "stop": []},
            )
            text = out.get("text", "").strip()
            bad, reason = is_bad(text)
            results.append({"label": label, "text": text[:80], "bad": bad,
                            "reason": reason})
        except Exception as exc:
            results.append({"label": label, "error": str(exc)[:200]})
    return _result("V11", "ok" if all("error" not in r for r in results) else "partial",
                   dtype=dtype_label, results=results)


# -----------------------------------------------------------------------------
# V13: Streaming generation
# -----------------------------------------------------------------------------
def v13_streaming(engine, dtype_label):
    """Verify stream=True yields token-by-token output, no NaN."""
    try:
        stream = engine.generate(
            prompt="Write a haiku about the moon:",
            sampling_params={"temperature": 0.0, "max_new_tokens": 32, "stop": []},
            stream=True,
        )
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        if not chunks:
            return _result("V13", "no_chunks", dtype=dtype_label)
        final_text = chunks[-1].get("text", "") if isinstance(chunks[-1], dict) else ""
        bad, reason = is_bad(final_text)
        return _result("V13", "ok", dtype=dtype_label,
                       n_chunks=len(chunks),
                       final_text=final_text[:80],
                       bad=bad, reason=reason)
    except Exception as exc:
        return _result("V13", "error", dtype=dtype_label, error=str(exc)[:200])


# -----------------------------------------------------------------------------
# V14: Chat template (synthetic — base model has none, inject simple)
# -----------------------------------------------------------------------------
def v14_chat_template(engine, dtype_label):
    """Apply a simple multi-turn template; just verify server doesn't crash."""
    template = (
        "System: You are a helpful assistant.\n"
        "User: What is 2+2?\n"
        "Assistant: 4\n"
        "User: What is 3+3?\n"
        "Assistant:"
    )
    try:
        out = engine.generate(
            prompt=template,
            sampling_params={"temperature": 0.0, "max_new_tokens": 16, "stop": []},
        )
        text = out.get("text", "").strip()
        bad, reason = is_bad(text)
        return _result("V14", "ok", dtype=dtype_label,
                       text=text[:100], bad=bad, reason=reason)
    except Exception as exc:
        return _result("V14", "error", dtype=dtype_label, error=str(exc)[:200])


# -----------------------------------------------------------------------------
# V15: Constrained decoding (regex constraint)
# -----------------------------------------------------------------------------
def v15_constrained(engine, dtype_label):
    """Use sglang xgrammar/regex constraint if available."""
    try:
        # Force output to match a JSON number pattern
        out = engine.generate(
            prompt="Output just a number for 17 + 25:",
            sampling_params={
                "temperature": 0.0, "max_new_tokens": 16, "stop": [],
                "regex": r"\d+",
            },
        )
        text = out.get("text", "").strip()
        # constraint should make output start with digits
        valid = bool(re.match(r"^\d+$", text)) if text else False
        return _result("V15", "ok", dtype=dtype_label,
                       text=text[:60], regex_pass=valid)
    except Exception as exc:
        return _result("V15", "unsupported", dtype=dtype_label, error=str(exc)[:200])


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--quantization", default="",
                   help="'', 'fp8', 'fp8_e4m3'")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ref-logprobs",
                   help="Path to bf16 ref logprobs JSON for V1 KL compare")
    p.add_argument("--soak-seconds", type=int, default=60)
    p.add_argument("--tier", type=int, default=2,
                   help="1 = Tier 1 only; 2 = all")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Skip these test IDs, e.g. V10 V15")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype_label = f"{args.dtype}_{args.quantization or 'none'}_tp{args.tp_size}"

    # Boot engine once
    print(f"[suite] boot {dtype_label}...", flush=True)
    try:
        engine, boot_s = boot_engine(args.model_path, args.dtype,
                                     args.quantization, args.tp_size)
    except Exception as exc:
        traceback.print_exc()
        rec = _result("BOOT", "fail", dtype=dtype_label, error=str(exc)[:200])
        (args.output_dir / f"{dtype_label}_results.json").write_text(json.dumps([rec], indent=2))
        return 2
    print(f"[suite] engine ready in {boot_s:.1f}s", flush=True)

    results = [_result("BOOT", "ok", dtype=dtype_label, boot_s=boot_s)]

    # Load ref logprobs for V1
    ref_lp = None
    if args.ref_logprobs and Path(args.ref_logprobs).exists():
        ref_data = json.loads(Path(args.ref_logprobs).read_text())
        ref_lp = ref_data.get("data")
        print(f"[suite] loaded {len(ref_lp)} ref logprobs from {args.ref_logprobs}", flush=True)

    # Sequence
    test_seq = [
        ("V1",  lambda: v01_numerical_parity(engine, dtype_label, ref_lp)),
        ("V7",  lambda: v07_throughput(engine, dtype_label)),
        ("V8",  lambda: v08_batch_sweep(engine, dtype_label)),
        ("V11", lambda: v11_edge_cases(engine, dtype_label)),
        ("V13", lambda: v13_streaming(engine, dtype_label)),
        ("V5",  lambda: v05_long_context(engine, dtype_label)),
        ("V6",  lambda: v06_long_generation(engine, dtype_label)),
        ("V14", lambda: v14_chat_template(engine, dtype_label)),
        ("V15", lambda: v15_constrained(engine, dtype_label)),
        ("V2",  lambda: v02_moe_routing(engine, dtype_label)),
        ("V10", lambda: v10_soak(engine, dtype_label, args.soak_seconds)),
    ]

    for vid, fn in test_seq:
        if vid in args.skip:
            print(f"[suite] {vid} skipped", flush=True)
            continue
        if args.tier < 2 and vid in {"V2", "V10", "V14", "V15"}:
            print(f"[suite] {vid} skipped (tier 1)", flush=True)
            continue
        try:
            t0 = time.perf_counter()
            r = fn()
            r["wall_s"] = round(time.perf_counter() - t0, 2)
            results.append(r)
            print(f"[suite] {vid} {r.get('status','?')} ({r['wall_s']:.1f}s)", flush=True)
        except Exception as exc:
            traceback.print_exc()
            results.append(_result(vid, "exception", dtype=dtype_label,
                                   error=str(exc)[:200]))

    # Dump full results
    out_path = args.output_dir / f"{dtype_label}_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[suite] wrote {out_path}", flush=True)

    # If we're the bf16 ref run, also dump the V1 data for the next runs
    if not args.quantization and args.dtype == "bfloat16" and not args.ref_logprobs:
        for r in results:
            if r.get("test") == "V1":
                ref_out = args.output_dir / "v01_ref_bf16.json"
                ref_out.write_text(json.dumps(r, indent=2))
                print(f"[suite] V1 ref → {ref_out}", flush=True)

    # Cleanup
    try:
        engine.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
