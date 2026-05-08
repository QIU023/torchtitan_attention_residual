"""Probe CUDA graph capture compatibility for the AttnRes overlay.

Static-write file to be run AFTER pretrain finishes. Boots SGLang
Engine on the canonical aligned 447M ckpt with cuda graph + piecewise
cuda graph ENABLED (default) and tries a single generation. Logs
clearly whether capture succeeded, captured-with-warnings, or failed.

Possible failure modes we want to detect:
* Dynamic block-list length confuses the capture (Phase 1 / Phase 2
  see different N at first decode steps).
* All-gather / reduce-scatter under seq-shard incompatible with
  capture (likely fine — they're standard NCCL collectives).
* RMSNorm contiguous shim breaks the captured sequence (instance-
  scoped wrap shouldn't, but verify).

Usage::

    python3 phase11/probe_cuda_graph.py --model phase11/hf_aligned_447m \\
        --tp 1
    python3 phase11/probe_cuda_graph.py --model phase11/hf_aligned_447m \\
        --tp 8 --seq-shard 1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


_RUNNER = '''
import sglang as sgl
import time, traceback

t0 = time.perf_counter()
try:
    e = sgl.Engine(
        model_path={model_path!r},
        skip_tokenizer_init=True,
        tp_size={tp_size},
        dtype="bfloat16",
        mem_fraction_static=0.5,
        # CUDA graphs ENABLED — explicit, no disable flags.
        log_level="warning",
        attention_backend="flashinfer",
        linear_attn_backend="triton",
    )
    t1 = time.perf_counter()
    print(f"BOOT_OK in {{t1-t0:.1f}}s (cuda graph capture survived)", flush=True)
except Exception as exc:
    t1 = time.perf_counter()
    print(f"BOOT_FAILED in {{t1-t0:.1f}}s", flush=True)
    traceback.print_exc()
    raise SystemExit(1)

# Single decode generation to exercise the captured graph.
try:
    out = e.generate(
        input_ids=[[1, 2, 3, 4, 5, 6, 7, 8]],
        sampling_params={{"max_new_tokens": 8, "temperature": 0}},
    )
    print(f"GEN_OK ids={{out[0]['output_ids']}}", flush=True)
except Exception:
    print("GEN_FAILED — graph captured but execution crashed", flush=True)
    traceback.print_exc()
    raise SystemExit(2)
finally:
    e.shutdown()

print("PROBE_OK_FULL", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seq-shard", type=int, default=0,
                    help="set SGLANG_ATTN_RES_SEQ_SHARD to this")
    args = ap.parse_args()

    code = _RUNNER.format(
        model_path=args.model, tp_size=args.tp,
    )
    env = dict(os.environ)
    env["SGLANG_ATTN_RES_SEQ_SHARD"] = str(args.seq_shard)
    # Explicitly clear bypass / naive-path env so we test the actual
    # production path with cuda graph.
    env.pop("SGLANG_ATTN_RES_BYPASS", None)
    env.pop("SGLANG_ATTN_RES_NAIVE_PATH", None)

    print(f"=== CUDA graph probe: tp={args.tp} seq_shard={args.seq_shard} ===")
    proc = subprocess.run(
        ["python3", "-c", code],
        env=env,
        cwd="/sgl-workspace/sglang",
        capture_output=True,
        text=True,
        timeout=900,
    )
    print("--- stdout ---")
    print(proc.stdout)
    if proc.returncode != 0:
        print("--- stderr (tail) ---")
        print("\n".join(proc.stderr.splitlines()[-30:]))
    print(f"\nexit code: {proc.returncode}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
