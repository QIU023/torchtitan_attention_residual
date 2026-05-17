"""PR #7 direct smoke for the DECODE path (_causal_conv1d_update_kernel).

Companion to ``smoke_kernel_direct_fp16.py`` which covers prefill
(``_causal_conv1d_fwd_kernel``). Real GRPO/inference traffic hits BOTH
kernels — prefill on the initial prompt then decode step-by-step.

Patch coverage of PR #7 (commit 4dfd8cf27):
- causal_conv1d_triton.py:127 (fwd kernel, ``col_dtype`` constexpr)
- causal_conv1d_triton.py:695 (update kernel, same fix mirrored)

This smoke confirms the update kernel is symmetrically patched.

Run:
    source ~/.venvs/sglang-dev/bin/activate
    python Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_kernel_decode_fp16.py
"""
from __future__ import annotations

import sys
import traceback

import torch

sys.path.insert(
    0,
    "/mnt/f/learning/2026Interview+Resume/AttnResidualTorchTitan/sglang/python",
)

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_update,
)


def make_decode_inputs(
    x_dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: str = "cuda",
):
    """Build a 1-token-per-batch decode call.

    Production: Kimi Linear KDA has short_conv_kernel_size=4 (HF config
    ``short_conv_kernel_size`` field on moonshotai/Kimi-Linear-48B-A3B).
    state_len = width - 1 = 3.
    """
    batch, dim, kernel_width = 2, 64, 4
    state_len = kernel_width - 1
    # decode: (batch, dim) shape — single-token per batch
    x = torch.randn(batch, dim, dtype=x_dtype, device=device)
    weight = torch.randn(dim, kernel_width, dtype=x_dtype, device=device)
    bias = torch.randn(dim, dtype=x_dtype, device=device)
    # conv_state: (num_cache_lines, dim, state_len)
    conv_state = torch.zeros(batch, dim, state_len, dtype=state_dtype, device=device)
    cache_seqlens = torch.tensor([5, 8], dtype=torch.int32, device=device)
    conv_state_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    return dict(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        activation="silu",
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )


def run_case(name: str, x_dtype, state_dtype) -> bool:
    print(f"\n=== {name}: x={x_dtype}, conv_state={state_dtype} ===", flush=True)
    try:
        inputs = make_decode_inputs(x_dtype, state_dtype)
        out = causal_conv1d_update(**inputs)
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
    print(
        "KERNEL_WIDTH=4 (matches Kimi Linear KDA short_conv_kernel_size=4)",
        flush=True,
    )

    cases = [
        ("decode_baseline_bf16_bf16", torch.bfloat16, torch.bfloat16),
        ("decode_bug_repro_fp16_x_bf16_state", torch.float16, torch.bfloat16),
        ("decode_inverted_bf16_x_fp16_state", torch.bfloat16, torch.float16),
        ("decode_all_fp16", torch.float16, torch.float16),
    ]

    results = {name: run_case(name, xd, sd) for name, xd, sd in cases}

    print("\n=== DECODE PATH SUMMARY ===", flush=True)
    for name, ok in results.items():
        print(f"  {'OK ' if ok else 'BAD'}  {name}", flush=True)

    bug_case_ok = results.get("decode_bug_repro_fp16_x_bf16_state", False)
    print(
        f"\nVerdict: decode path "
        f"{'PR #7 patch verified' if bug_case_ok else 'FAILURE'}",
        flush=True,
    )
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
