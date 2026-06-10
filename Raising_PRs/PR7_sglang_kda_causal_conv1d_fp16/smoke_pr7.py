"""PR #7 smoke harness for ``causal_conv1d_triton.py`` — 3 modes:

  prefill         dtype matrix on ``_causal_conv1d_fwd_kernel``     (needs CUDA)
  decode          dtype matrix on ``_causal_conv1d_update_kernel``  (needs CUDA)
  byte-identical  bf16+bf16 output equality vs upstream/main        (needs CUDA + git)
  all (default)   runs the three modes back-to-back

Both ``prefill`` and ``decode`` iterate the 4-case dtype matrix
(baseline_bf16_bf16 / bug_repro_fp16_x_bf16_state / inverted_bf16_x_fp16_state /
all_fp16) over both production KERNEL_WIDTH values: KW=3 (LFM2 / LFM2-MoE,
``conv_L_cache``) and KW=4 (Kimi-Linear KDA, ``short_conv_kernel_size``).

``byte-identical`` calls ``git checkout`` inside the sglang submodule to swap
``causal_conv1d_triton.py`` between ``upstream/main`` and the current branch
HEAD, runs the kernel under each, and ``torch.equal()``-compares outputs.
The sglang submodule MUST initially be on the patched branch; the script
restores it on exit.

Run:
    source ~/.venvs/sglang-dev/bin/activate
    python smoke_pr7.py                  # all
    python smoke_pr7.py prefill          # just dtype matrix on fwd kernel
    python smoke_pr7.py decode           # just dtype matrix on update kernel
    python smoke_pr7.py byte-identical   # just bf16 regression vs upstream
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import traceback
from pathlib import Path

import torch

SGLANG_REPO = Path("/mnt/f/learning/2026Interview+Resume/AttnResidualTorchTitan/sglang")
KERNEL_REL = "python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py"
sys.path.insert(0, str(SGLANG_REPO / "python"))

# KW=3: LFM2 / LFM2-MoE (conv_L_cache); KW=4: Kimi-Linear KDA (short_conv_kernel_size)
PRODUCTION_KERNEL_WIDTHS = [3, 4]
DTYPE_CASES = [
    ("baseline_bf16_bf16", torch.bfloat16, torch.bfloat16),
    ("bug_repro_fp16_x_bf16_state", torch.float16, torch.bfloat16),
    ("inverted_bf16_x_fp16_state", torch.bfloat16, torch.float16),
    ("all_fp16", torch.float16, torch.float16),
]


def _print_device_info() -> None:
    print(
        f"device: {torch.cuda.get_device_name(0)} "
        f"(SM {torch.cuda.get_device_capability(0)}); torch {torch.__version__}",
        flush=True,
    )


def _fresh_import_kernel():
    for name in list(sys.modules):
        if "causal_conv1d_triton" in name or "causal_conv1d" in name:
            del sys.modules[name]
    return importlib.import_module(
        "sglang.srt.layers.attention.mamba.causal_conv1d_triton"
    )


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(SGLANG_REPO), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


# ---------- prefill (fwd kernel) ----------

def _make_inputs_prefill(x_dtype, state_dtype, kernel_width):
    dim, seqlen = 64, 8
    return dict(
        x=torch.randn(dim, seqlen, dtype=x_dtype, device="cuda"),
        weight=torch.randn(dim, kernel_width, dtype=x_dtype, device="cuda"),
        bias=torch.randn(dim, dtype=x_dtype, device="cuda"),
        conv_states=torch.zeros(
            1, dim, kernel_width - 1, dtype=state_dtype, device="cuda"
        ),
        query_start_loc=torch.tensor([0, seqlen], dtype=torch.int32, device="cuda"),
        seq_lens_cpu=[seqlen],
        cache_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        has_initial_state=torch.tensor([True], dtype=torch.bool, device="cuda"),
        activation="silu",
    )


def _run_prefill_case(name, x_dtype, state_dtype, kernel_width) -> bool:
    print(
        f"\n=== {name} KW={kernel_width}: x={x_dtype}, conv_states={state_dtype} ===",
        flush=True,
    )
    try:
        mod = _fresh_import_kernel()
        out = mod.causal_conv1d_fn(
            **_make_inputs_prefill(x_dtype, state_dtype, kernel_width)
        )
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
        for line in traceback.format_exc().splitlines()[-12:]:
            print("  " + line, flush=True)
        return False


def run_prefill_smoke() -> bool:
    print("\n########## PREFILL (_causal_conv1d_fwd_kernel) ##########", flush=True)
    results = {}
    for kw in PRODUCTION_KERNEL_WIDTHS:
        for name, xd, sd in DTYPE_CASES:
            key = f"KW{kw}_{name}"
            results[key] = _run_prefill_case(key, xd, sd, kw)
    print("\n=== PREFILL SUMMARY ===", flush=True)
    for k, ok in results.items():
        print(f"  {'OK ' if ok else 'BAD'}  {k}", flush=True)
    return all(
        results[f"KW{kw}_bug_repro_fp16_x_bf16_state"]
        for kw in PRODUCTION_KERNEL_WIDTHS
    )


# ---------- decode (update kernel) ----------

def _make_inputs_decode(x_dtype, state_dtype, kernel_width):
    batch, dim = 2, 64
    state_len = kernel_width - 1
    return dict(
        x=torch.randn(batch, dim, dtype=x_dtype, device="cuda"),
        conv_state=torch.zeros(
            batch, dim, state_len, dtype=state_dtype, device="cuda"
        ),
        weight=torch.randn(dim, kernel_width, dtype=x_dtype, device="cuda"),
        bias=torch.randn(dim, dtype=x_dtype, device="cuda"),
        activation="silu",
        cache_seqlens=torch.tensor([5, 8], dtype=torch.int32, device="cuda"),
        conv_state_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
    )


def _run_decode_case(name, x_dtype, state_dtype, kernel_width) -> bool:
    print(
        f"\n=== {name} KW={kernel_width}: x={x_dtype}, conv_state={state_dtype} ===",
        flush=True,
    )
    try:
        mod = _fresh_import_kernel()
        out = mod.causal_conv1d_update(
            **_make_inputs_decode(x_dtype, state_dtype, kernel_width)
        )
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
        for line in traceback.format_exc().splitlines()[-12:]:
            print("  " + line, flush=True)
        return False


def run_decode_smoke() -> bool:
    print("\n########## DECODE (_causal_conv1d_update_kernel) ##########", flush=True)
    results = {}
    for kw in PRODUCTION_KERNEL_WIDTHS:
        for name, xd, sd in DTYPE_CASES:
            key = f"KW{kw}_decode_{name}"
            results[key] = _run_decode_case(key, xd, sd, kw)
    print("\n=== DECODE SUMMARY ===", flush=True)
    for k, ok in results.items():
        print(f"  {'OK ' if ok else 'BAD'}  {k}", flush=True)
    return all(
        results[f"KW{kw}_decode_bug_repro_fp16_x_bf16_state"]
        for kw in PRODUCTION_KERNEL_WIDTHS
    )


# ---------- byte-identical (bf16+bf16 vs upstream/main) ----------

def _build_bytewise_inputs(seed, kernel_width):
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
        query_start_loc=torch.tensor([0, seqlen], dtype=torch.int32, device="cuda"),
        seq_lens_cpu=[seqlen],
        cache_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        has_initial_state=torch.tensor([True], dtype=torch.bool, device="cuda"),
        activation="silu",
    )


def _run_bytewise_under_rev(label, rev, kernel_width):
    _git("checkout", rev, "--", KERNEL_REL)
    mod = _fresh_import_kernel()
    out = mod.causal_conv1d_fn(**_build_bytewise_inputs(seed=42, kernel_width=kernel_width))
    torch.cuda.synchronize()
    print(
        f"  [{label}] KW={kernel_width}: sum={out.float().sum().item():.6f}",
        flush=True,
    )
    return out.detach().clone()


def run_byte_identical_smoke() -> bool:
    print(
        "\n########## BYTE-IDENTICAL (bf16+bf16 vs upstream/main) ##########",
        flush=True,
    )
    saved = _git("rev-parse", "HEAD").strip()
    print(f"current HEAD: {saved[:10]} (will restore on exit)", flush=True)
    print(f"upstream/main: {_git('rev-parse', 'upstream/main').strip()[:10]}", flush=True)
    all_equal = True
    try:
        for kw in PRODUCTION_KERNEL_WIDTHS:
            print(f"\n=== KERNEL_WIDTH={kw} ===", flush=True)
            patched = _run_bytewise_under_rev("patched", saved, kw)
            upstream = _run_bytewise_under_rev("upstream", "upstream/main", kw)
            equal = torch.equal(upstream, patched)
            max_diff = (upstream.float() - patched.float()).abs().max().item()
            print(
                f"  torch.equal(upstream, patched) = {equal}; max abs diff = {max_diff}",
                flush=True,
            )
            if not equal:
                all_equal = False
    finally:
        _git("checkout", saved, "--", KERNEL_REL)
        for name in list(sys.modules):
            if "causal_conv1d_triton" in name or "causal_conv1d" in name:
                del sys.modules[name]
        print(f"\nrestored kernel from {saved[:10]}", flush=True)
    return all_equal


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["prefill", "decode", "byte-identical", "all"],
        help="which smoke to run (default: all)",
    )
    args = parser.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    _print_device_info()

    verdicts = {}
    if args.mode in ("prefill", "all"):
        verdicts["prefill"] = run_prefill_smoke()
    if args.mode in ("decode", "all"):
        verdicts["decode"] = run_decode_smoke()
    if args.mode in ("byte-identical", "all"):
        verdicts["byte-identical"] = run_byte_identical_smoke()

    print("\n########## VERDICTS ##########", flush=True)
    for k, ok in verdicts.items():
        label = {
            "prefill": "PR #7 patch verified across all production KERNEL_WIDTHs",
            "decode": "decode path PR #7 patch verified across all production KERNEL_WIDTHs",
            "byte-identical": "byte-identical to upstream on bf16+bf16",
        }[k]
        print(f"  {'OK ' if ok else 'BAD'}  {k}: {label}", flush=True)

    sys.exit(0 if all(verdicts.values()) else 1)


if __name__ == "__main__":
    main()
