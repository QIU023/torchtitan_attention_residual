"""PR #8 smoke harness for ``_maybe_shrink_config_for_sm120`` — 2 modes:

  real-device   on the actual local GPU: every early-return guard must hand back
                the SAME config object (identity), proving the non-SM-12.0
                byte-identical claim. Runs on any CUDA GPU (4070Ti included).
  mock-sm120    monkeypatch ``is_sm120_supported`` + ``_sm120_shmem_per_block_bytes``
                to fake an RTX 5090 (SM 12.0, 101376-byte cap) and verify the
                shrink arithmetic on the H100 default config + guard cases.
  all (default) both modes back-to-back.

What CANNOT be verified locally without SM 12.0 hardware: the real Triton
OutOfResources trigger and the post-shrink kernel launch. Those are covered by
the fork-side RTX 5090 production run cited in PR8_BODY.md (38.9 tok/s fp8,
coherent 8/8).

Run:
    source ~/.venvs/sglang-dev/bin/activate
    python smoke_pr8.py                # all
    python smoke_pr8.py real-device
    python smoke_pr8.py mock-sm120
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path

import torch

SGLANG_REPO = Path("/mnt/f/learning/2026Interview+Resume/AttnResidualTorchTitan/sglang")
sys.path.insert(0, str(SGLANG_REPO / "python"))


class _Anything:
    """Recursive sink: any attr/call/index returns itself, so deep chains survive."""

    def __getattr__(self, name):
        return self

    def __call__(self, *a, **kw):
        return self

    def __getitem__(self, key):
        return self


_ANYTHING = _Anything()


class _StubModule(types.ModuleType):
    """Any-attribute stub so the import chain survives without sgl_kernel's SM89 wheel."""

    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _ANYTHING


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that satisfies any `sgl_kernel[.sub]` import with a stub."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


# sgl_kernel ships no SM89 binary; the helper under test never calls into it.
sys.meta_path.insert(0, _StubFinder())

from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: E402
    fused_moe_triton_kernels as kmod,
)

# H100 default for fp8 from get_default_config(); the exact config that overflows SM 12.0.
H100_FP8_DEFAULT = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 8,
    "num_stages": 4,
}
RTX5090_SHMEM_CAP = 101376


def _est(cfg) -> int:
    return (
        cfg["BLOCK_SIZE_M"] * cfg["BLOCK_SIZE_K"]
        + cfg["BLOCK_SIZE_K"] * cfg["BLOCK_SIZE_N"]
    ) * cfg["num_stages"]


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}", flush=True)
    return ok


def run_real_device() -> bool:
    print("\n########## REAL-DEVICE (early-return guards) ##########", flush=True)
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)} (SM {cap[0]}.{cap[1]})", flush=True)
    results = []

    cfg = dict(H100_FP8_DEFAULT)
    out = kmod._maybe_shrink_config_for_sm120(cfg, True, False, None)
    if cap[0] >= 12:
        results.append(
            _check(
                "sm120_device_shrinks_h100_default",
                out is not cfg and out["BLOCK_SIZE_M"] == 64,
                f"got {out}",
            )
        )
    else:
        # Non-SM-12.0 must return the *same object* — the byte-identical guarantee.
        results.append(
            _check(
                "non_sm120_returns_same_object",
                out is cfg,
                f"is_sm120_supported()={kmod.is_sm120_supported()}",
            )
        )
        results.append(
            _check("non_sm120_config_unmutated", cfg == H100_FP8_DEFAULT)
        )

    out2 = kmod._maybe_shrink_config_for_sm120(dict(H100_FP8_DEFAULT), False, False, None)
    results.append(
        _check("non_quant_path_returns_same_object_any_device", out2 is not None and out2 == H100_FP8_DEFAULT)
    )
    return all(results)


def run_mock_sm120() -> bool:
    print("\n########## MOCK-SM120 (shrink arithmetic) ##########", flush=True)
    saved_is_sm120 = kmod.is_sm120_supported
    saved_shmem_fn = kmod._sm120_shmem_per_block_bytes
    kmod.is_sm120_supported = lambda: True
    kmod._sm120_shmem_per_block_bytes = lambda: RTX5090_SHMEM_CAP
    results = []
    try:
        # Case 1: H100 fp8 default overflows (est 196608 > 101376) -> full 4-step shrink.
        cfg = dict(H100_FP8_DEFAULT)
        out = kmod._maybe_shrink_config_for_sm120(cfg, True, False, None)
        expected = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }
        results.append(_check("h100_default_shrunk", out == expected, f"got {out}"))
        results.append(
            _check(
                "shrunk_config_fits_cap",
                _est(out) <= RTX5090_SHMEM_CAP,
                f"est {_est(out)} <= {RTX5090_SHMEM_CAP}",
            )
        )
        results.append(_check("input_config_not_mutated", cfg == H100_FP8_DEFAULT))

        # Case 2: int8 path also triggers the shrink.
        out = kmod._maybe_shrink_config_for_sm120(dict(H100_FP8_DEFAULT), False, True, None)
        results.append(_check("int8_path_shrinks_too", out["BLOCK_SIZE_M"] == 64))

        # Case 3: non-quant path early-returns the same object even on SM 12.0.
        cfg = dict(H100_FP8_DEFAULT)
        out = kmod._maybe_shrink_config_for_sm120(cfg, False, False, None)
        results.append(_check("bf16_path_untouched_on_sm120", out is cfg))

        # Case 4: already-fitting config short-circuits (est 49152 <= cap).
        small = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "num_warps": 4,
            "num_stages": 2,
        }
        out = kmod._maybe_shrink_config_for_sm120(small, True, False, None)
        results.append(_check("fitting_config_returns_same_object", out is small))

        # Case 5: block-wise quant pins BLOCK_SIZE_N/K back to block_shape.
        out = kmod._maybe_shrink_config_for_sm120(
            dict(H100_FP8_DEFAULT), True, False, [128, 128]
        )
        results.append(
            _check(
                "blockwise_pins_n_k",
                out["BLOCK_SIZE_N"] == 128 and out["BLOCK_SIZE_K"] == 128,
                f"got N={out['BLOCK_SIZE_N']} K={out['BLOCK_SIZE_K']}",
            )
        )
        results.append(
            _check(
                "blockwise_result_fits_cap",
                _est(out) <= RTX5090_SHMEM_CAP,
                f"est {_est(out)}",
            )
        )

        # Case 6: Hopper-sized cap mock -> early return (guard 3).
        kmod._sm120_shmem_per_block_bytes = lambda: 228 * 1024
        cfg = dict(H100_FP8_DEFAULT)
        out = kmod._maybe_shrink_config_for_sm120(cfg, True, False, None)
        results.append(_check("hopper_cap_on_sm120_returns_same_object", out is cfg))
    finally:
        kmod.is_sm120_supported = saved_is_sm120
        kmod._sm120_shmem_per_block_bytes = saved_shmem_fn

    return all(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["real-device", "mock-sm120", "all"],
        help="which smoke to run (default: all)",
    )
    args = parser.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"

    verdicts = {}
    if args.mode in ("real-device", "all"):
        verdicts["real-device"] = run_real_device()
    if args.mode in ("mock-sm120", "all"):
        verdicts["mock-sm120"] = run_mock_sm120()

    print("\n########## VERDICTS ##########", flush=True)
    for k, ok in verdicts.items():
        print(f"  {'OK ' if ok else 'BAD'}  {k}", flush=True)
    sys.exit(0 if all(verdicts.values()) else 1)


if __name__ == "__main__":
    main()
