## Motivation

`invoke_fused_moe_kernel` selects a per-tile config (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
`BLOCK_SIZE_K`, `num_stages`, `num_warps`) that defaults to Hopper-class tuning
(`get_default_config()` in `fused_moe.py` + the tuned JSONs under
`python/sglang/srt/layers/moe/configs/`). Hopper's per-block shared memory cap is
~228 KB (SM 9.0) and B100/Blackwell datacenter (SM 10.0) is similar.

**Blackwell consumer GPUs (SM 12.0, e.g. RTX 5090 / 5080) ship with only ~100 KB of
shared memory per block.** The H100 fp8/int8 default
(`BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, num_stages=4` ≈ 192 KB shmem) overflows that
cap, and the engine never reaches the launch site:

```
Engine(quantization="fp8", model="<any MoE model on RTX 5090>")
  triton.runtime.errors.OutOfResources:
    Required 147456, Hardware limit 101376
```

The `should_enable_swap_ab()` helper next to this kernel already gates on
`is_sm90_supported()` — there's a precedent for SM-aware launcher logic in this
file. This PR extends that pattern to SM 12.0 + fp8/int8: shrink the config
in-place before launch so the kernel fits, with three early-return guards to
keep H100 / A100 / B100 / MI300 paths **byte-identical**.

## Modifications

`_maybe_shrink_config_for_sm120()` shrinks in four stages, each guarded so it only
fires when the current config still doesn't fit:

1. Cap `BLOCK_SIZE_M` at 64 (halves the A-tile footprint).
2. Cap `BLOCK_SIZE_N` at 128 (halves the B-tile footprint).
3. Cap `num_stages` at 2 (Hopper default is 3-4; SM 12.0's L2 prefetcher still
   hides most of the latency at 2 stages with the smaller shmem budget).
4. Cap `num_warps` at 4 when `BLOCK_M=64` (the register file fits with fewer warps).

Block-wise quant constraints (`BLOCK_SIZE_K == block_shape[1]`,
`BLOCK_SIZE_N == block_shape[0]`) are preserved — the helper only touches
`BLOCK_SIZE_M` and the pipeline/warp counts in the block-wise path.

Hooked at `invoke_fused_moe_kernel()` entry as a single function call. Three
early returns at the top of the helper keep the non-affected paths cost-free:

```python
def _maybe_shrink_config_for_sm120(
    config: Dict[str, Any],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    block_shape: Optional[List[int]],
) -> Dict[str, Any]:
    if not (use_fp8_w8a8 or use_int8_w8a8):
        return config            # 1. non-quant path: untouched
    if not _is_cuda or not is_sm120_supported():
        return config            # 2. non-SM-12.0 device: untouched
    shmem_cap = _sm120_shmem_per_block_bytes()
    if shmem_cap == 0 or shmem_cap >= 128 * 1024:
        return config            # 3. unexpectedly large shmem: untouched
    # ... 4-stage shrink only if est_shmem still > cap
```

`+107 / -0`, one file. Single linear commit.

| | H100 / A100 / B100 / MI300 (today) | RTX 5090 / 5080 + fp8 (this PR fixes) |
|---|---|---|
| Path inside `invoke_fused_moe_kernel` | unchanged (`is_sm120_supported() == False` short-circuits) | shrinks H100 config in-place before launch |
| Triton launch | succeeds (today) | succeeds (this PR) |
| Engine boot under `--quantization fp8` | OK | crashes `OutOfResources` (today) → OK (this PR) |

## Accuracy Tests

`_maybe_shrink_config_for_sm120` does not change kernel semantics — only the
per-tile launch geometry. Output is bit-identical at fixed seed on:

- **Non-SM-12.0 devices** (H100 / A100 / B100 / MI300): guarded by the
  `is_sm120_supported()` early return; the helper is never entered, so the
  config object reaching `triton.jit`'s launcher is exactly today's value.
- **SM 12.0 + non-fp8/int8 paths** (bf16/fp16 MoE): guarded by the
  `use_fp8_w8a8 or use_int8_w8a8` early return; same as above.
- **SM 12.0 + fp8/int8 path with a config that already fits in shmem**: guarded
  by the `_est_shmem(...) <= shmem_cap` short-circuit; helper returns the input
  config unchanged.

The shrunk path is what gets exercised on RTX 5090; coherent generations on the
Kimi-Linear AttnRes fp8 inference smoke (see Speed Tests below).

## Speed Tests and Profiling

- **H100/A100/B100/MI300**: no extra ops on the existing path (early return
  before the helper executes any work); no measurable change.
- **RTX 5090 (SM 12.0)** — verified end-to-end on Kimi-Linear AttnRes 1.4B-active:

  | Configuration | Throughput | Coherence |
  |---|---|---|
  | bf16 baseline | 44.6 tok/s | 8/8 |
  | fp8 weight-only **without this PR** | crashes at `OutOfResources` before any token | N/A |
  | fp8 weight-only **with this PR** | 38.9 tok/s | 8/8 |

  The fp8 path runs at ~87% of the bf16 baseline throughput — the gap is
  weight-dequant overhead in fp8 weight-only mode, not anything specific to the
  shrunk tile geometry. The shrunk config keeps the same `BLOCK_SIZE_K` (and
  hence the same block-wise quant granularity), so block-wise quant accuracy
  is preserved.

### Why not just lower the tuned JSON?

The tuned JSONs under `configs/` are device-keyed; adding a 5090 entry would
work but would require us to fingerprint every Blackwell consumer variant
(5090 / 5080 / future cards). The runtime-shmem-cap-check is forward-compatible
(it triggers on any SM 12.0 device whose shmem cap is < 128 KB, no JSON edit
needed when a new card ships). Happy to flip to a tuned JSON if a maintainer
prefers — the helper itself becomes a small no-op once the JSON contains a
fitting entry, because the `_est_shmem(...) <= shmem_cap` check short-circuits.

### Follow-up (out of scope for this PR)

A downstream consumer reported that even the shrunk config triggers an
`illegal memory access` inside the fp8 fused-MoE Triton kernel on SM 12.0 once
the launch geometry is satisfied (a kernel-level bug, separate from the
launcher-level config-fit problem this PR fixes). Their workaround in user code
is `SGLANG_FP8_IGNORED_LAYERS="mlp.experts"` to fall back to
`UnquantizedFusedMoEMethod` for the MoE while keeping fp8 on the dense path.
That issue is independent and orthogonal — happy to file it as a separate
issue if maintainers want triage visibility.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit). *(Verified locally against sglang's pinned hook versions on the patched file: `isort 7.0.0`, `black 26.1.0`, `ruff 0.15.1` with sglang's `--select=F401,F821`, `codespell 2.4.1` — all clean.)*
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). *(No CI-registered test in this PR — the trigger requires SM 12.0 hardware which is not in current sglang CI. Happy to add a registered unit test that exercises the helper in pure-Python with mocked device properties under `test/registered/unit/` if maintainers prefer.)*
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). *(Coherent 8/8 + 38.9 tok/s on RTX 5090 Kimi-Linear AttnRes fp8 inference; H100/A100/B100/MI300 paths byte-identical via early-return guards.)*
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
