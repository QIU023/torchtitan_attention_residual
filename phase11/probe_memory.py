"""Memory probe: validate the blog's 15GB->1.9GB claim shape at long context.

Boots SGLang Engine on the canonical aligned 447M ckpt at TP=8 with
either replicated (default) or seq-shard mode, runs a single long
prefill (16K tokens), and records peak per-rank GPU memory at three
checkpoints via nvidia-smi (parent torch.cuda is not initialised, the
workers are subprocess pid-namespace siblings).

  * post-Engine-init        - model weights + KV cache pool only
  * post-warmup-prefill     - first prefill pass; block reps materialised
  * post-second-prefill     - steady state

The blog's claim is that under seq-shard, block representations of
shape ``(N, T, d)`` are sharded along ``T`` to ``(N, T/P, d)`` per
rank, so per-rank block-rep memory drops by P. At our 1.4B / d=1024 /
N=4 / 16K context that's:

  block_reps replicated: 4 x 16384 x 1024 x 2 B = 128 MB / rank
  block_reps shard P=8 : 4 x 2048  x 1024 x 2 B =  16 MB / rank

So expect ~110 MB peak-mem delta favouring shard, modulo SGLang's
allocator-cache noise. The blog's headline 15GB->1.9GB number is at
d=7168 N=8 T=128K - same shape, ~58x ours.

Usage:
  python3 phase11/probe_memory.py --shard 0
  python3 phase11/probe_memory.py --shard 1
  diff the two
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _gpu_used_mb() -> list[int]:
    """Return per-GPU memory.used (MB) via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x.strip()) for x in out.strip().splitlines() if x.strip()]


_RUNNER = '''
import json, os, time
import sglang as sgl

PREFILL_LEN = {prefill_len}
TP_SIZE = {tp_size}

def emit(tag):
    print(f"PROBE_TAG {{{{\\"tag\\":\\"{{tag}}\\",\\"t\\":{{time.time()}}}}}}", flush=True)

emit("pre_boot")
t0 = time.perf_counter()
e = sgl.Engine(
    model_path={model_path!r},
    skip_tokenizer_init=True,
    tp_size=TP_SIZE,
    dtype="bfloat16",
    mem_fraction_static=0.6,
    log_level="error",
    attention_backend="flashinfer",
    linear_attn_backend="triton",
)
t1 = time.perf_counter()
emit("post_boot")
time.sleep(2)

ids = list(range(1, PREFILL_LEN + 1))
out = e.generate(input_ids=[ids],
                 sampling_params={{"max_new_tokens": 1, "temperature": 0}})
emit("post_warmup")
time.sleep(2)

out = e.generate(input_ids=[ids],
                 sampling_params={{"max_new_tokens": 1, "temperature": 0}})
emit("post_run")
time.sleep(2)

e.shutdown()
emit("done")
print(f"BOOT_TIME_S {{t1 - t0}}", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/root/torchtitan_attention_residual/phase11/hf_aligned_447m")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--prefill", type=int, default=16384)
    ap.add_argument("--shard", type=int, default=0,
                    help="set SGLANG_ATTN_RES_SEQ_SHARD")
    args = ap.parse_args()

    code = _RUNNER.format(
        model_path=args.model, prefill_len=args.prefill, tp_size=args.tp,
    )
    env = dict(os.environ)
    env["SGLANG_ATTN_RES_SEQ_SHARD"] = str(args.shard)
    env.pop("SGLANG_ATTN_RES_BYPASS", None)
    env.pop("SGLANG_ATTN_RES_NAIVE_PATH", None)

    print(f"=== MEM PROBE: tp={args.tp} prefill={args.prefill} shard={args.shard} ===")
    proc = subprocess.Popen(
        ["python3", "-c", code],
        env=env,
        cwd="/sgl-workspace/sglang",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    samples: list[dict] = []
    boot_time = None
    last_tag_time: dict[str, float] = {}
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            line = line.rstrip()
            if line.startswith("PROBE_TAG "):
                payload = json.loads(line[len("PROBE_TAG "):])
                # Sample nvidia-smi multiple times to catch peak.
                peak = [0] * args.tp
                t_end = time.time() + 1.5
                while time.time() < t_end:
                    cur = _gpu_used_mb()
                    for i, m in enumerate(cur[:args.tp]):
                        if m > peak[i]:
                            peak[i] = m
                    time.sleep(0.1)
                samples.append({
                    "tag": payload["tag"],
                    "per_gpu_MB_peak": peak,
                    "max_MB": max(peak),
                    "min_MB": min(peak),
                })
                last_tag_time[payload["tag"]] = time.time()
                print(f"  [{payload['tag']:>11s}] per-rank max={max(peak)} MB  "
                      f"min={min(peak)} MB  spread={max(peak) - min(peak)} MB")
            elif line.startswith("BOOT_TIME_S "):
                boot_time = float(line.split()[1])
    finally:
        proc.wait(timeout=30)

    # Tag-keyed summary.
    by_tag = {s["tag"]: s for s in samples}
    result = {
        "args": vars(args),
        "boot_time_s": boot_time,
        "samples": samples,
        "summary": {
            tag: {"max_MB": by_tag[tag]["max_MB"]
                  if tag in by_tag else None}
            for tag in ("pre_boot", "post_boot", "post_warmup", "post_run")
        },
    }
    out_dir = Path("/root/torchtitan_attention_residual/phase11/bench_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mem_tp{args.tp}_p{args.prefill}_shard{args.shard}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote -> {out_path}")
    if proc.returncode != 0:
        err = proc.stderr.read()
        print("STDERR (tail):")
        print("\n".join(err.splitlines()[-15:]))


if __name__ == "__main__":
    main()
