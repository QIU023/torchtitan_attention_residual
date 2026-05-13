"""4-way SGLang AttnRes inference benchmark.

Compares four execution modes on the canonical aligned 447M Kimi
AttnRes ckpt at TP=1 (algorithm cost only) and TP=8 (algorithm cost
+ comm fabric):

* vanilla     — SGLANG_ATTN_RES_BYPASS=1: skip every aggregation, run
                pure PreNorm. Same model class, same ckpt, same
                env-compat patches; only the AttnRes algorithm itself
                is removed.
* naive       — SGLANG_ATTN_RES_NAIVE_PATH=1: every layer re-reads
                every committed block (the naive single-pass aggregator
                from the Zhihu blog).
* two-phase   — default: Phase 1 batched once per block + Phase 2
                online-softmax merge per layer. The optimisation under
                test.
* shard       — default + SGLANG_ATTN_RES_SEQ_SHARD=1: two-phase plus
                the sequence-dim TP shard with reduce-scatter+all-gather
                comm fusion. Only meaningful at TP>1.

Workload: 1 prompt × 1024-token prefill + 256-token decode, 5 timed
runs after 2 warmup runs. Reports mean ± stdev for prefill TTFT and
decode tokens/sec.

Usage:
    python3 phase11/bench_attn_res.py --model phase11/hf/lm_base \
        --tp 1
    python3 phase11/bench_attn_res.py --model phase11/hf/lm_base \
        --tp 8

Run on a free GPU box (training must not be active). Each engine boot
takes ~30 s; full bench at TP=8 takes ~10 min.
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


_MODES = (
    # (name, env-vars to set)
    ("vanilla",   {"SGLANG_ATTN_RES_BYPASS": "1"}),
    ("naive",     {"SGLANG_ATTN_RES_NAIVE_PATH": "1"}),
    ("two-phase", {}),
    ("shard",     {"SGLANG_ATTN_RES_SEQ_SHARD": "1"}),
)

_RUNNER_TEMPLATE = '''
import json, os, statistics, sys, time
import sglang as sgl

PREFILL_LEN  = {prefill_len}
DECODE_LEN   = {decode_len}
N_WARMUP     = {n_warmup}
N_TIMED      = {n_timed}
TP_SIZE      = {tp_size}

# Build a deterministic prompt of the requested prefill length.
PROMPT_IDS = list(range(1, PREFILL_LEN + 1))

e = sgl.Engine(
    model_path={model_path!r},
    skip_tokenizer_init=True,
    tp_size=TP_SIZE,
    dtype="bfloat16",
    mem_fraction_static=0.85,
    disable_cuda_graph={disable_cuda_graph},
    disable_piecewise_cuda_graph={disable_piecewise_cuda_graph},
    log_level="error",
    attention_backend="flashinfer",
    linear_attn_backend="triton",
)

# Warmup runs (compile, kernel autotune, JIT cache).
for _ in range(N_WARMUP):
    out = e.generate(
        input_ids=[PROMPT_IDS],
        sampling_params={{"max_new_tokens": DECODE_LEN, "temperature": 0}},
    )

# Timed runs.
prefill_ttfts = []
decode_tps = []
for _ in range(N_TIMED):
    t0 = time.perf_counter()
    out = e.generate(
        input_ids=[PROMPT_IDS],
        sampling_params={{"max_new_tokens": 1, "temperature": 0}},
    )
    t1 = time.perf_counter()
    prefill_ttfts.append((t1 - t0) * 1000.0)  # ms

    t0 = time.perf_counter()
    out = e.generate(
        input_ids=[PROMPT_IDS],
        sampling_params={{"max_new_tokens": DECODE_LEN, "temperature": 0}},
    )
    t1 = time.perf_counter()
    # Subtract prefill TTFT from end-to-end to isolate decode time.
    full_e2e = (t1 - t0) * 1000.0
    decode_only_ms = max(1.0, full_e2e - prefill_ttfts[-1])
    decode_tps.append(DECODE_LEN / (decode_only_ms / 1000.0))

e.shutdown()

result = {{
    "prefill_ttft_ms_mean": statistics.mean(prefill_ttfts),
    "prefill_ttft_ms_stdev": statistics.stdev(prefill_ttfts) if len(prefill_ttfts) > 1 else 0.0,
    "decode_tps_mean": statistics.mean(decode_tps),
    "decode_tps_stdev": statistics.stdev(decode_tps) if len(decode_tps) > 1 else 0.0,
    "prefill_ttft_raw": prefill_ttfts,
    "decode_tps_raw": decode_tps,
}}
print("\\nBENCH_RESULT_JSON " + json.dumps(result), flush=True)
'''


def _run_one_mode(name: str, env_vars: dict, args) -> dict | None:
    """Spawn an isolated subprocess for one bench mode and parse its result."""
    code = _RUNNER_TEMPLATE.format(
        model_path=args.model,
        prefill_len=args.prefill,
        decode_len=args.decode,
        n_warmup=args.warmup,
        n_timed=args.timed,
        tp_size=args.tp,
        disable_cuda_graph=str(args.disable_cuda_graph),
        disable_piecewise_cuda_graph=str(args.disable_cuda_graph),
    )
    env = dict(os.environ)
    # Clear any inherited AttnRes env so each mode runs cleanly.
    for k in (
        "SGLANG_ATTN_RES_BYPASS",
        "SGLANG_ATTN_RES_NAIVE_PATH",
        "SGLANG_ATTN_RES_SEQ_SHARD",
    ):
        env.pop(k, None)
    env.update(env_vars)

    print(f"\n{'='*60}")
    print(f"MODE: {name}   env: {env_vars or '{}'}")
    print(f"{'='*60}", flush=True)

    proc = subprocess.run(
        ["python3", "-c", code],
        env=env,
        cwd="/sgl-workspace/sglang",
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        print("STDOUT (tail):")
        print("\n".join(proc.stdout.splitlines()[-20:]))
        print("STDERR (tail):")
        print("\n".join(proc.stderr.splitlines()[-20:]))
        return None

    # Parse the BENCH_RESULT_JSON marker.
    import json
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT_JSON "):
            return json.loads(line[len("BENCH_RESULT_JSON "):])
    print("No BENCH_RESULT_JSON marker found in stdout.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF ckpt dir")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--prefill", type=int, default=1024,
                    help="prompt length in tokens")
    ap.add_argument("--decode", type=int, default=256,
                    help="number of new tokens to decode")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timed", type=int, default=5)
    ap.add_argument("--disable-cuda-graph", action="store_true",
                    help="run with --disable-cuda-graph (eager mode); default leaves cuda graph ON")
    ap.add_argument("--out", default="phase11/bench_results.json")
    args = ap.parse_args()

    if args.tp == 1:
        # shard mode is a no-op at TP=1 (helpers degenerate to identity);
        # skip to save bench wall.
        modes = [(n, e) for (n, e) in _MODES if n != "shard"]
    else:
        modes = list(_MODES)

    results = {"args": vars(args), "modes": {}}
    for name, env in modes:
        r = _run_one_mode(name, env, args)
        if r is not None:
            results["modes"][name] = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results -> {out}\n")

    # Pretty-print summary.
    print(f"{'Mode':<12} {'TTFT (ms)':>16} {'Decode tps':>16}")
    print("-" * 50)
    for name in [m[0] for m in modes]:
        if name not in results["modes"]:
            print(f"{name:<12} {'(failed)':>16} {'':>16}")
            continue
        r = results["modes"][name]
        print(
            f"{name:<12} "
            f"{r['prefill_ttft_ms_mean']:>10.2f} ± {r['prefill_ttft_ms_stdev']:<3.2f} "
            f"{r['decode_tps_mean']:>10.2f} ± {r['decode_tps_stdev']:<3.2f}"
        )

    if "vanilla" in results["modes"] and "two-phase" in results["modes"]:
        v = results["modes"]["vanilla"]
        tp = results["modes"]["two-phase"]
        ttft_overhead_pct = (tp["prefill_ttft_ms_mean"] / v["prefill_ttft_ms_mean"] - 1) * 100
        tps_overhead_pct = (1 - tp["decode_tps_mean"] / v["decode_tps_mean"]) * 100
        print(
            f"\nAttnRes (two-phase) overhead vs vanilla: "
            f"prefill +{ttft_overhead_pct:.1f}% TTFT, "
            f"decode -{tps_overhead_pct:.1f}% throughput"
        )

    if "naive" in results["modes"] and "two-phase" in results["modes"]:
        nv = results["modes"]["naive"]
        tp = results["modes"]["two-phase"]
        speedup = nv["prefill_ttft_ms_mean"] / tp["prefill_ttft_ms_mean"]
        print(
            f"Two-phase speedup vs naive: "
            f"prefill {speedup:.2f}× ({nv['prefill_ttft_ms_mean']:.0f}ms → "
            f"{tp['prefill_ttft_ms_mean']:.0f}ms)"
        )


if __name__ == "__main__":
    main()
