# MoonEP: the weight half moves to MoonEP's own kernels (2026-09-17)

Written for whoever takes `k3_moonep_seam` next on a GPU box. It records what the
branch now does, the MoonEP facts it rests on, and what is still unverified.

Decision behind it (user, 2026-09-17): the PR is a draft, so go straight to the
fused path instead of compacting the copy backend first. The audit that raised
the question is in `MOONEP_TEST_PLAN_2026-09-16.md`, section "Diff audit against
the #4577 rules".

## What changed

Before (branch head `9e71d1073`): `MoonEPTableBackendNVLink` moved slot weights
and slot gradients with plain copies over `create_nvl_single_owner_tensor`
mappings, the compute tables were local `[E + B, in, out]` (a physical row for
every expert on every rank), and each MoE layer cost three EP-group barriers and
two `plan.experts_to_copy.tolist()` host syncs per step.

After: the same class calls MoonEP's own `launch_prefetch` and
`launch_grad_reduce`. Each rank holds `[E / R + B, in, out]` bf16 rows per
projection instead of `[E + B]`, the gradient table is one fp32 `[E, in, out]`
per projection of which the kernel writes only this rank's span, and the cost per
MoE layer per step is one barrier before the prefetch, one before the reduce, and
no host sync.

## The MoonEP facts this rests on (read at `2bd860b`)

- `prefetch.py:357` `launch_prefetch(remote_expert, prefetch_buffers, experts_to_copy, num_sms)` takes the source rows and the slot rows as **two separate tensors**, each only required to be contiguous and rank 3, with `experts_to_copy` the **1-D `[B]`** row of the plan for this rank (`api.py:716` passes `experts_to_copy[ctx['rank']]`). It derives `E` from `remote_expert.shape[0]`, so a per-owner chunk works as long as the ids handed in are local to that chunk.
- `prefetch.py:401` asserts `H % 128 == 0 and Hp % 128 == 0`: the kernel tiles 128 by 128. K3's latent 3584 and expert hidden 3072 pass, and so do the debug flavor's 512 and 384.
- `grad_reduce.py:437` `launch_grad_reduce(remote_expert_grads, remote_reduce_buffers, experts_to_copy, rank, num_sms, meta_buf, meta_stride, barrier_off, grid_sync_bar)` wants fp32 `[E, H, Hp]` grads (global expert ids; it updates only `rank * E/R : (rank+1) * E/R`), fp32 `[R, B, H, Hp]` reduce buffers, and the `[R, B]` plan. It fences the ranks **inside the kernel** after its remote reads and clears the slots it consumed; the caller only has to fence the writes that precede it.
- `buffer.py:217` `create_nvl_dist_tensor(chunk_shape, dtype, local_rank, world_size, group)` returns the concatenation of every rank's chunk, each chunk physically on its owner, and demands `chunk_shape` already padded with `pad_dim0_for_alignment` (`buffer.py:95`). `buffer.py:338` `create_nvl_single_owner_tensor` puts one tensor on one owner, mapped RW everywhere, and a leading-dim slice of it stays contiguous.
- The public wrappers are not usable here. `api.py` `prefetch_weight` asserts `w.shape[0] == E + B` and slices `[:E]` / `[E:]`; `_launch_full_grad_reduces` (`api.py:207`) asserts the same for the grads. The README states the layout as a hard requirement ("one contiguous VMM range `[E+B, H, H']`, identically laid out on every rank", rows `[0, E)` being "all ranks' local experts ... each chunk physically *is* the home rank's parameter memory") because "the group GEMM addresses experts purely by row index", and **no allocator for that range ships with the package**: MoonEP's own `tests/test_e2e.py:72` builds it as a local `torch.empty(E + B, H, Hp)`. Nothing in the package addresses expert parameters that are FSDP-sharded DTensors, which is what titan hands us.
- MoonEP's own `tests/test_grad_reduce.py:404` calls `launch_grad_reduce` directly with a plain local grad tensor, a `create_nvl_dist_tensor` reduce buffer viewed as `[R, B, H, Hp]`, and the meta handles taken from `buffer._require_ctx()`. That is the precedent this branch follows.

## The design that follows

One allocation does three jobs. Per projection, every rank allocates a
`create_nvl_single_owner_tensor` of `[P + B, in, out]` bf16 on each owner
(`_MappedRows`), where `P = E / R`. Its own tensor is the compute table; rows
`[:P]` are what peers read; rows `[P:]` are handed to `launch_prefetch` as the
prefetch buffers. So the prefetch kernel writes straight into the rows the
grouped GEMM will read, with no staging copy.

Prefetch runs once per owner: the ids for that owner are rebased to its chunk
(`ids - owner * P`, everything else `-1`), so the kernel's global-id addressing
is satisfied by a chunk-local tensor and the `[E]`-wide contiguous range is never
needed. Unused slots are not written, which is the kernel's documented
behaviour.

Gradients keep MoonEP's layout: one fp32 `[E, in, out]` table per projection,
plus `create_nvl_dist_tensor([B, in, out], fp32).view(R, B, in, out)` as the
reduce buffers. The backward writes this rank's own row grads into its span of
the table and its slot grads into its chunk of the reduce buffers, then
`launch_grad_reduce` adds every rank's slot grads for this rank's experts into
that span and clears them. Parameter grads are read back from the span.

The GEMM sees a compacted row axis. MoonEP's `cu_seqlens` covers `E + B` rows;
`_offsets` takes this rank's expert rows and its slot rows and concatenates them
(`cat(cu[lo:hi], cu[E:])`). That is valid because a token reaches either its
expert's home rank or a rank holding a slot copy, so rows of experts homed
elsewhere are empty here. Two `torch._assert_async` tripwires pin exactly that
(`cu[lo-1] == 0` and `cu[E-1] == cu[hi-1]`), device side, so they cost no host
sync. `matrix_scripts/moonep_row_occupancy_probe.py` is the same check as a
one-step probe with readable output.

`B` is now derived on MoonEP's VMM granularity (`padded_slot_count` in the
dispatcher, `E / R` rounded up so a reduce-buffer chunk is a legal dist-tensor
chunk) and the `Buffer` is built with that value, because the plan's slot width
and the buffers have to agree. A `num_prefetch_slots` set in a config is taken as
it stands, which is what the CPU tests use.

## What was not taken, and why

- Our own VMM reservation (reserve a range, map the R chunks and the slot pages
  back to back) would give the wrappers what they want, but it is a distributed
  allocator inside a model folder, and being faithful to the README's version of
  it means the expert parameter storage itself lives in the symmetric range,
  which reaches FSDP's allocation, DTensor sharding and the optimizer. That is a
  core-level change, not a model-folder one.
- `Buffer.prefetch_weight` / `Buffer.reduce_grad`: blocked by the `E + B`
  contiguity assert above.
- The copy transport was not kept as a second selectable backend. One layout,
  one transport: `9e71d1073` carries the copy version if a measurement needs it.

The one private access is `buffer._require_ctx()` for the four barrier handles,
wrapped in `MoonEPTokenDispatcher.grad_reduce_handles()` so it has one named
place. Upstream ask for MoonEP, in this order: expose those handles (or accept
the weight rows and the slot rows as separate tensors in `prefetch_weight` and
`reduce_grad`, which its own kernels already do), and fix the grad-reduce kernel
on cutlass-dsl 4.6.

## Size and style

| file | total | code | comment | docstring |
| --- | ---: | ---: | ---: | ---: |
| `kimi_k3/moon_ep_experts.py` | 339 | 249 | 18 | 25 |
| `kimi_k3/moon_ep_dispatcher.py` | 288 | 192 | 16 | 40 |

Added lines against the branch base `a3a819c67`: 688 production (experts 339,
dispatcher 288, `moe.py` 20, `__init__.py` 25/-8, core seam 16/-9) and 638 tests
and double, so 1326 in total against 1256 for the copy version: the fused path
costs about 70 production lines more. Comment plus docstring is 18.3% of the
production lines, and all of it is one line per public class or method plus one
line per config field, with no `Args:` blocks, no module-level design argument
and no measured numbers in code. No line exceeds 88 columns.

## Evidence on the Windows box

- The CPU unit passes: two ranks as threads through the in-process double, the
  compacted rows, forward and backward against a dense fp32 reference with a
  duplicated expert (output, input gradients, and expert gradients including the
  rows another rank computed in its slots). Run through
  `scratchpad/run_moonep_cpu.py`, which injects a bare `kimi_k3` package because
  the package `__init__` needs attn-gym's CuTeDSL kernels, absent here.
- The double is not vacuous. Three mutations, each applied in process only:
  offsets drop the slot rows, `reduce_grad` does nothing, the own-row refresh
  skips one projection. All three were caught.
- `py_compile` clean on every changed file; the 88-column scan clean.
- Not run here: `pytest` on the two CPU test files (their first cases import
  `model_registry`, which needs CuTeDSL), ufmt, pyrefly, and anything that
  touches a GPU. **No MoonEP kernel in this rewrite has executed yet.**

## For the GPU box, in order

1. The H200 tables currently in `MOONEP_H200_2026-09-17.md` were measured on
   `9e71d1073`, the copy transport. They no longer describe the branch head.
   Re-run the cells that go in the body.
2. `pytest tests/unit_tests/cpu/test_kimi_k3_moon_ep_dispatcher.py tests/unit_tests/cpu/test_ep_token_dispatcher_capacity.py -q`
   and refresh the pass count in the PR body's test plan.
3. `moonep_row_occupancy_probe.py` for one step: it is the readable form of the
   two tripwires, and it also logs slot occupancy, which cell 3 wants anyway.
4. Resolved on the box the same day, so this paragraph's earlier warning is
   retired: the cutlass-dsl 4.6.0 failure was not an MLIR location problem but
   `cute.make_fragment` renamed to `cute.make_rmem_tensor` in 4.6, surfacing
   through the DSL tracer as `AttributeError` on the K3 path's first backward.
   One line in MoonEP's `grad_reduce.py:295`
   (`matrix_scripts/moonep_onbox/h200/moonep_grad_reduce_cutlass46.patch`) makes
   all six MoonEP suites pass on 4.6.0, the version Attention Gym's KDA needs.
   The rename is the third upstream ask for MoonEP.
5. Cell 9 has run on real MoonEP (dp2 x ep2, one step, `3c458bdf1`): 8 of 8
   dispatches reported only this rank's home rows and its planned slots
   receiving tokens, no violation, 32 of 48 rows useful at R = 2. The empty-row
   assumption the compacted offsets rest on is therefore measured, not assumed.
   The probe's `tokens=... of S*K=...` line counts padded VM-group rows rather
   than tokens, which is a wording bug in the probe, now fixed.
