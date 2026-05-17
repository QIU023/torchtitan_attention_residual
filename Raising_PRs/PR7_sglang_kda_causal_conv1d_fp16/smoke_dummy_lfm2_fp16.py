"""PR #7 smoke: SGLang Engine boot LFM2-1.2B random-init weights @ fp16.

Purpose: verify the patched ``_causal_conv1d_fwd_kernel`` and
``_causal_conv1d_update_kernel`` in
``python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py``
JIT-compile and run when SGLang is launched with ``--dtype float16``.

Pre-patch behaviour (upstream/main):
    Triton compilation error inside the kernel body:
        triton.compiler.errors.CompilationError: at line ...:
            Mismatched type for col0 between then block (bf16)
            and else block (fp16)

Post-patch (our fork's ``a6c46168a`` isolated as commit ``4dfd8cf27``
on branch ``pr7-kda-causal-conv1d-fp16``):
    Boot succeeds; greedy decode yields tokens (random because of
    ``load_format=dummy``, but the engine doesn't crash).

Resource budget on RTX 4070Ti (12 GB):
    LFM2-1.2B fp16 weights ~2.4 GB + KV cache + workspace ~3 GB.
    Comfortable headroom for the smoke.

Run:
    source ~/.venvs/sglang-dev/bin/activate
    python Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_dummy_lfm2_fp16.py
"""
from __future__ import annotations

import sys
import time

print("[smoke] importing sglang ...", flush=True)
t0 = time.time()
import sglang as sgl

print(f"[smoke] sglang imported in {time.time() - t0:.1f}s, version={sgl.__version__}", flush=True)

print("[smoke] booting Engine: LFM2-1.2B dummy weights fp16 ...", flush=True)
t0 = time.time()
engine = sgl.Engine(
    model_path="LiquidAI/LFM2-1.2B",
    load_format="dummy",
    dtype="float16",
    mem_fraction_static=0.6,
    log_level="warning",
)
print(f"[smoke] engine booted in {time.time() - t0:.1f}s", flush=True)

print("[smoke] running 1-prompt generate to force kernel JIT ...", flush=True)
t0 = time.time()
out = engine.generate(
    prompt=["The quick brown fox"],
    sampling_params={"max_new_tokens": 8, "temperature": 0.0},
)
gen_time = time.time() - t0
print(f"[smoke] generated in {gen_time:.2f}s", flush=True)
print(f"[smoke] output (random text expected from dummy weights): {out[0]['text']!r}", flush=True)

print("[smoke] shutting down ...", flush=True)
engine.shutdown()

print("[smoke] PASS: causal_conv1d_triton fp16 type-join patch works.", flush=True)
sys.exit(0)
