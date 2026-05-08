# torchtitan env-compat patches (vast.ai SGLang environment)

**Patch file:** `phase11/torchtitan_vast_ai_env_compat.patch` (203 lines, 5 files)

## Why this exists (not committed to torchtitan submodule)

The vast.ai box this project rents to run SGLang AttnRes inference + the
Phase 11 fabric/bench sweep has the following stack pinned by sgl_kernel
binary compatibility:

| component | version | reason |
| --- | --- | --- |
| GPU | 8× RTX 5090 (SM 12.0, Blackwell) | rented hardware |
| CUDA | 12.9 | sgl_kernel cu129 wheel |
| Python | 3.12 | sgl_kernel py312 wheel |
| **torch** | **2.9.1+cu129 stable** | sgl_kernel ABI |
| triton | 3.6 | matching torch 2.9 |

torchtitan `main` (the upstream branch we track) targets **torch nightly
~2.10**, so it imports several private/nightly-only APIs that simply
don't exist on stable 2.9. The patches in
`torchtitan_vast_ai_env_compat.patch` add `hasattr` guards and
`try/except ImportError` wrappers so the module tree imports cleanly on
the locked-down environment without changing any logic on torch
nightly.

These patches deliberately stay **local to the vast.ai box** —
committing them upstream would pin torchtitan to stable APIs and lose
the nightly-only optimisations (skip_fwd_side_effects_in_bwd,
varlen_attn, FA3 selection, etc.). On a torch-nightly box the patches
are no-ops at runtime (every guard's "unavailable" branch is dead).

## What each patch does

| File | Symbol | Reason |
| --- | --- | --- |
| `torchtitan/distributed/context_parallel.py` | `_context_parallel_shard`, `_ContextParallel`, `_HeadTailLoadBalancer`, `_PTRRLoadBalancer`, `_enable_context_parallel_dispatcher` | Private CP APIs in `torch.distributed.tensor.experimental._attention` only exist on nightly. Import is guarded; CP just unavailable on stable. |
| `torchtitan/distributed/parallel_dims.py` | `DeviceMesh._unflatten` | Nightly-only `_unflatten` method. Falls back to manual mesh-axis listing on stable. |
| `torchtitan/distributed/utils.py` | `init_process_group(_ranks=...)` | Nightly added a `_ranks` kwarg to `init_process_group`. Stable doesn't accept it; we drop it on the call. |
| `torchtitan/experiments/kimi_linear/parallelize.py` | `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint` | Nightly-only dynamo flag. Wrapped in `hasattr` so stable silently skips. |
| `torchtitan/models/common/attention.py` | `activate_flash_attention_impl`, `current_flash_attention_impl`, `varlen_attn`, `wrap_inductor_compiled_regions` | FA3-selection helpers + varlen kernel + inductor-compiled-region flag are all torch ≥2.10 / nightly-only. Stubs raise `NotImplementedError` if reached at runtime; on SM 12.0 (no FA3) the call sites are dead anyway. |

## Apply / refresh on a fresh box

```bash
# From repo root, with torchtitan submodule clean
cd torchtitan && git apply ../phase11/torchtitan_vast_ai_env_compat.patch
```

To regenerate this patch file after editing torchtitan locally:
```bash
cd torchtitan && git diff HEAD > ../phase11/torchtitan_vast_ai_env_compat.patch
```

## When this patch becomes obsolete

When the vast.ai box's torch is bumped to nightly ≥2.10 with cu129
support, the patch is a no-op (every guard's stable branch is dead) and
the diff can be deleted. Until then it lives here outside the
torchtitan submodule pointer to keep the upstream submodule clean.
