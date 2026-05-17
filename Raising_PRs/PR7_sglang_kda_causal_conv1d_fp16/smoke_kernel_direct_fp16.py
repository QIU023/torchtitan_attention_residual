"""PR #7 direct Triton-kernel smoke (no SGLang Engine needed).

Tests the exact failure mode this PR fixes by invoking
``causal_conv1d_fn`` with the type-mismatch scenario:
    x.dtype          = torch.float16
    conv_states.dtype = torch.bfloat16   (the SGLang default for the MAMBA
                                          conv-state cache, independent of
                                          model dtype)

Pre-patch (upstream/main):
    triton.compiler.errors.CompilationError:
        Mismatched type for col0 between then block (bf16)
        and else block (fp16)

Post-patch (commit 4dfd8cf27 on ``pr7-kda-causal-conv1d-fp16``):
    Kernel compiles + runs; output has the same dtype as x (fp16).

Why this is more rigorous than an Engine boot smoke:
- Removes ALL integration confounders (sgl_kernel SM availability,
  model registry, vLLM compat shims, mem fraction, KV cache mgr).
- Directly exercises the Triton SSA type-join site the patch fixes.
- Runs in < 5 s and needs zero network / no model download.

Run:
    source ~/.venvs/sglang-dev/bin/activate
    python Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_kernel_direct_fp16.py
"""
from __future__ import annotations

import sys
import traceback

import torch

# Patch import path so we can pull just the kernel module without dragging
# in sglang.srt.__init__ (which fails on SM89 due to sgl_kernel missing).
sys.path.insert(
    0,
    "/mnt/f/learning/2026Interview+Resume/AttnResidualTorchTitan/sglang/python",
)

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
)


def make_inputs(
    x_dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: str = "cuda",
):
    """Build a single-sequence varlen batch that hits the type-join site.

    The chunk_offset==0 branch loads from conv_states (state_dtype) while
    the else-branch creates tl.zeros at x_ptr.dtype.element_ty. For the
    bug to fire, those must differ.
    """
    batch, dim, seqlen, kernel_width = 1, 64, 8, 4
    # x layout: (dim, total_tokens) — 2D continuous-batched
    x = torch.randn(dim, seqlen, dtype=x_dtype, device=device)
    weight = torch.randn(dim, kernel_width, dtype=x_dtype, device=device)
    bias = torch.randn(dim, dtype=x_dtype, device=device)
    # conv_states: cache for the (kernel_width-1) prior tokens, may differ in dtype
    conv_states = torch.zeros(
        1, dim, kernel_width - 1, dtype=state_dtype, device=device
    )
    query_start_loc = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    seq_lens_cpu = [seqlen]
    cache_indices = torch.tensor([0], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor([True], dtype=torch.bool, device=device)
    return dict(
        x=x,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )


def run_case(name: str, x_dtype, state_dtype) -> bool:
    print(f"\n=== {name}: x={x_dtype}, conv_states={state_dtype} ===", flush=True)
    try:
        inputs = make_inputs(x_dtype, state_dtype)
        out = causal_conv1d_fn(**inputs)
        torch.cuda.synchronize()
        assert out.dtype == x_dtype, f"output dtype {out.dtype} != x dtype {x_dtype}"
        print(
            f"PASS — out.shape={tuple(out.shape)} out.dtype={out.dtype} "
            f"finite={torch.isfinite(out).all().item()}",
            flush=True,
        )
        return True
    except Exception as e:
        print(f"FAIL — {type(e).__name__}: {e}", flush=True)
        # Compact traceback (last 8 frames is enough for triton compile site)
        tb_lines = traceback.format_exc().splitlines()
        for line in tb_lines[-12:]:
            print("  " + line, flush=True)
        return False


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    print(
        f"device: {torch.cuda.get_device_name(0)} "
        f"(SM {torch.cuda.get_device_capability(0)})",
        flush=True,
    )
    print(f"torch {torch.__version__}", flush=True)

    cases = [
        # Baseline: same dtype → no bug even pre-patch
        ("baseline_bf16_bf16", torch.bfloat16, torch.bfloat16),
        # The actual bug scenario PR #7 fixes
        ("bug_repro_fp16_x_bf16_state", torch.float16, torch.bfloat16),
        # Inverted mismatch (also exercises type-join branch)
        ("inverted_bf16_x_fp16_state", torch.bfloat16, torch.float16),
        # All-fp16 (also new dtype path for KDA)
        ("all_fp16", torch.float16, torch.float16),
    ]

    results = {name: run_case(name, xd, sd) for name, xd, sd in cases}

    print("\n=== SUMMARY ===", flush=True)
    for name, ok in results.items():
        print(f"  {'OK ' if ok else 'BAD'}  {name}", flush=True)

    bug_case_ok = results.get("bug_repro_fp16_x_bf16_state", False)
    print(
        f"\nVerdict: {'PR #7 patch verified' if bug_case_ok else 'FAILURE'} on this device",
        flush=True,
    )
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
