"""PR #7 byte-identical regression guard for the bf16+bf16 default path.

For the default scenario (model dtype == conv_states dtype == bfloat16), the
``.to(col_dtype)`` cast we add is a no-op (same dtype → Triton's
constexpr-resolved cast compiles away). So kernel output should be
**bit-identical** between upstream/main and our patched version.

This is the most important regression test for non-Kimi-Linear callers
(LFM2 / LFM2-MoE / Qwen3-Next / Qwen3.5 / falcon_h1 / nemotron_h / ...): if
this passes byte-identical, the patch cannot break any default-dtype caller.

Method: run the kernel twice in the same process — output is captured
under one source file content, then we OS-swap the file to the other
version (via subprocess `git checkout`), fresh-import the kernel module,
run again, compare. The sglang submodule must initially be on the patched
branch (`yiqiaoq/kda-causal-conv1d-fp16`); this script restores it at exit.

Run:
    source ~/.venvs/sglang-dev/bin/activate
    python Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_byte_identical_bf16.py
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import torch

SGLANG_REPO = Path("/mnt/f/learning/2026Interview+Resume/AttnResidualTorchTitan/sglang")
sys.path.insert(0, str(SGLANG_REPO / "python"))

KERNEL_REL = "python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py"
PATCHED_REV = "HEAD"          # current branch tip (assumed: yiqiaoq/kda-causal-conv1d-fp16)
UPSTREAM_REV = "upstream/main"


def git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(SGLANG_REPO), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def checkout_file(rev: str) -> None:
    git("checkout", rev, "--", KERNEL_REL)
    # Drop the module from sys.modules so the next import reads the new bytes.
    for name in list(sys.modules):
        if "causal_conv1d_triton" in name or "causal_conv1d" in name:
            del sys.modules[name]


def build_inputs(seed: int, kernel_width: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dim, seqlen = 64, 8
    return dict(
        x=torch.randn(dim, seqlen, dtype=torch.bfloat16, device="cuda", generator=g),
        weight=torch.randn(
            dim, kernel_width, dtype=torch.bfloat16, device="cuda", generator=g
        ),
        bias=torch.randn(dim, dtype=torch.bfloat16, device="cuda", generator=g),
        conv_states=torch.randn(
            1, dim, kernel_width - 1, dtype=torch.bfloat16, device="cuda", generator=g
        ),
        query_start_loc=torch.tensor(
            [0, seqlen], dtype=torch.int32, device="cuda"
        ),
        seq_lens_cpu=[seqlen],
        cache_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        has_initial_state=torch.tensor([True], dtype=torch.bool, device="cuda"),
        activation="silu",
    )


def run_once(label: str, kernel_width: int):
    # Fresh import so the JIT picks up the current file bytes.
    mod = importlib.import_module(
        "sglang.srt.layers.attention.mamba.causal_conv1d_triton"
    )
    inputs = build_inputs(seed=42, kernel_width=kernel_width)
    out = mod.causal_conv1d_fn(**inputs)
    torch.cuda.synchronize()
    print(
        f"  [{label}] KW={kernel_width}: out.shape={tuple(out.shape)} "
        f"out.dtype={out.dtype} finite={torch.isfinite(out).all().item()} "
        f"sum={out.float().sum().item():.6f}",
        flush=True,
    )
    return out.detach().clone()


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    print(
        f"device: {torch.cuda.get_device_name(0)} "
        f"(SM {torch.cuda.get_device_capability(0)})",
        flush=True,
    )
    print(f"sglang repo: {SGLANG_REPO}", flush=True)

    # Capture the current branch tip so we restore correctly at exit.
    saved_rev = git("rev-parse", "HEAD").strip()
    print(f"current HEAD: {saved_rev[:10]} (will restore on exit)", flush=True)
    print(
        f"upstream/main: {git('rev-parse', UPSTREAM_REV).strip()[:10]}", flush=True
    )

    results = {}
    try:
        for kw in (3, 4):  # LFM2 uses 3; Kimi-Linear uses 4
            print(f"\n=== KERNEL_WIDTH={kw} ===", flush=True)

            checkout_file(PATCHED_REV)
            patched_out = run_once("patched", kw)

            checkout_file(UPSTREAM_REV)
            upstream_out = run_once("upstream", kw)

            equal = torch.equal(upstream_out, patched_out)
            max_diff = (
                (upstream_out.float() - patched_out.float()).abs().max().item()
            )
            print(
                f"  torch.equal(upstream, patched) = {equal}; max abs diff = {max_diff}",
                flush=True,
            )
            results[kw] = equal
            if not equal:
                print(
                    "  FAIL: bf16+bf16 path differs between upstream and patched. "
                    "The PR is supposed to be a no-op here.",
                    flush=True,
                )
    finally:
        # Restore the patched kernel for the working tree.
        git("checkout", saved_rev, "--", KERNEL_REL)
        # Drop module cache so subsequent runs see the restored bytes.
        for name in list(sys.modules):
            if "causal_conv1d_triton" in name or "causal_conv1d" in name:
                del sys.modules[name]
        print(f"\nrestored kernel from {saved_rev[:10]}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for kw, ok in results.items():
        print(
            f"  KERNEL_WIDTH={kw}: "
            f"{'BYTE-IDENTICAL' if ok else 'DIFFERS'} on bf16+bf16",
            flush=True,
        )
    all_ok = all(results.values()) and len(results) == 2
    print(
        f"\nVerdict: {'PR #7 is byte-identical for bf16+bf16 default callers' if all_ok else 'POSSIBLE REGRESSION'}",
        flush=True,
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
