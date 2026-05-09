"""SGLang built-in profiler for AttnRes inference.

Uses SGLang's `engine.start_profile()` API which routes torch.profiler
into the worker subprocess (where the model actually runs). This is
required because `sgl.Engine` runs the model in a separate process;
a parent-side `torch.profiler` sees nothing.

For each mode (vanilla / naive / two-phase / [shard at TP>1]):
* Boots engine
* Warms up
* Profiles a single prefill+decode pass
* Saves chrome trace + a top-N kernel summary

Output: per-mode chrome trace JSON + a summary CSV.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path


_RUNNER = '''
import json, os, time
import sglang as sgl

PREFILL_LEN = {prefill_len}
DECODE_LEN = {decode_len}
TP_SIZE = {tp_size}
TRACE_DIR = {trace_dir!r}
MODE = {mode!r}

os.makedirs(TRACE_DIR, exist_ok=True)

# Boot engine.
e = sgl.Engine(
    model_path={model_path!r},
    skip_tokenizer_init=True,
    tp_size=TP_SIZE,
    dtype="bfloat16",
    mem_fraction_static=0.5,
    log_level="error",
    attention_backend="flashinfer",
    linear_attn_backend="triton",
)

ids = list(range(1, PREFILL_LEN + 1))

# Warmup (cuda-graph capture, JIT compile).
for _ in range(2):
    e.generate(input_ids=[ids],
               sampling_params={{"max_new_tokens": DECODE_LEN, "temperature": 0}})

# SGLang built-in profiler: starts torch.profiler inside the worker.
e.start_profile(
    output_dir=TRACE_DIR,
    activities=["CPU", "GPU"],
    record_shapes=False,
    with_stack=False,
)

# Single timed pass under profiler.
e.generate(input_ids=[ids],
           sampling_params={{"max_new_tokens": DECODE_LEN, "temperature": 0}})

e.stop_profile()
time.sleep(2)  # let trace flush

print(f"\\nPROFILE_DONE_MARKER mode={{MODE}} trace_dir={{TRACE_DIR}}", flush=True)
e.shutdown()
'''


def _run_one_mode(name: str, env_vars: dict, args, trace_root: Path) -> bool:
    code = _RUNNER.format(
        model_path=args.model,
        prefill_len=args.prefill,
        decode_len=args.decode,
        tp_size=args.tp,
        trace_dir=str(trace_root / name),
        mode=name,
    )
    env = dict(os.environ)
    for k in (
        "SGLANG_ATTN_RES_BYPASS",
        "SGLANG_ATTN_RES_NAIVE_PATH",
        "SGLANG_ATTN_RES_SEQ_SHARD",
    ):
        env.pop(k, None)
    env.update(env_vars)
    print(f"\n{'='*60}\nMODE: {name}   env: {env_vars or '{}'}\n{'='*60}", flush=True)
    proc = subprocess.run(
        ["python3", "-c", code],
        env=env,
        cwd="/sgl-workspace/sglang",
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        print("STDOUT (tail):"); print("\n".join(proc.stdout.splitlines()[-15:]))
        print("STDERR (tail):"); print("\n".join(proc.stderr.splitlines()[-15:]))
        return False
    return any(line.startswith("PROFILE_DONE_MARKER") for line in proc.stdout.splitlines())


def _summarize_trace(trace_dir: Path) -> dict:
    """Read SGLang's torch trace files, return top kernels by cuda time."""
    trace_files = list(trace_dir.glob("*.trace.json.gz")) + list(trace_dir.glob("*.json.gz")) + list(trace_dir.glob("*.json"))
    if not trace_files:
        return {"error": "no trace file found", "files": [str(f) for f in trace_dir.iterdir()] if trace_dir.exists() else []}
    # Pick the largest (rank 0 trace).
    trace_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    trace = trace_files[0]
    opener = gzip.open if str(trace).endswith(".gz") else open
    with opener(trace, "rb") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    # Aggregate: GPU kernels have cat=kernel and dur (in microseconds).
    cuda_time = {}
    cuda_count = {}
    for ev in events:
        cat = ev.get("cat", "").lower()
        if "kernel" in cat or "gpu" in cat:
            name = ev.get("name", "<unknown>")
            dur = ev.get("dur", 0)
            cuda_time[name] = cuda_time.get(name, 0) + dur
            cuda_count[name] = cuda_count.get(name, 0) + 1
    items = sorted(cuda_time.items(), key=lambda x: -x[1])
    total = sum(cuda_time.values())
    top = [
        {"name": n, "cuda_us": int(t), "count": cuda_count[n], "pct": round(100.0 * t / max(total, 1), 2)}
        for n, t in items[:30]
    ]
    return {
        "trace_file": str(trace),
        "total_kernel_us": int(total),
        "top_kernels": top,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--out", default="phase11/profile_results")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    trace_root = out_root / "kineto"
    out_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)

    if args.tp == 1:
        modes = [
            ("vanilla",   {"SGLANG_ATTN_RES_BYPASS": "1"}),
            ("naive",     {"SGLANG_ATTN_RES_NAIVE_PATH": "1"}),
            ("two-phase", {}),
        ]
    else:
        modes = [
            ("vanilla",   {"SGLANG_ATTN_RES_BYPASS": "1"}),
            ("naive",     {"SGLANG_ATTN_RES_NAIVE_PATH": "1"}),
            ("two-phase", {}),
            ("shard",     {"SGLANG_ATTN_RES_SEQ_SHARD": "1"}),
        ]

    results = {"args": vars(args), "modes": {}}
    for name, env in modes:
        ok = _run_one_mode(name, env, args, trace_root)
        if ok:
            summary = _summarize_trace(trace_root / name)
            results["modes"][name] = summary
        else:
            results["modes"][name] = {"failed": True}

    out_json = out_root / f"profile_tp{args.tp}_prefill{args.prefill}.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults → {out_json}\n")

    print(f"{'Mode':<12} {'Total kernel us':>16} {'Top kernel (name | %)':<60}")
    print("-" * 90)
    for name in [m[0] for m in modes]:
        if name not in results["modes"] or results["modes"][name].get("failed"):
            print(f"{name:<12} (failed)")
            continue
        s = results["modes"][name]
        if "error" in s:
            print(f"{name:<12} ERROR: {s['error']}")
            continue
        if not s.get("top_kernels"):
            print(f"{name:<12} no kernels"); continue
        top = s["top_kernels"][0]
        print(f"{name:<12} {s['total_kernel_us']:>16d} {top['name'][:50]:<50} ({top['pct']}%)")


if __name__ == "__main__":
    main()
