# PR #16 — common MoE: routing-map scatter drops the router's shard placement (breaks TP+EP)

**Target repo**: `pytorch/torchtitan`
**Target file**: `torchtitan/models/common/moe.py` (`MoE.forward`, the one-hot routing-map construction)
**Fork reference**: torchtitan `attention_residual_dev`, commit `129e29de`
**Upstream audit (2026-07-25)**: `upstream/main` @ `fd277658` still has the unmodified `scatter_` and the stale comment claiming DTensor runs it as a local op. **Not obsoleted; still applies.**
**Re-audit (2026-08-03)**: `upstream/main` @ `681fd4b50` — bare `scatter_` still at `moe.py:465`; patch verified to apply cleanly in a fresh worktree. **Can be filed in parallel with PR19** (`moe_sharding.py` in_grad_placements): different files, no overlap, both apply to `681fd4b50` in either order.
**Effort**: ~half a day (patch + a TP+EP integration cell + unit test).
**Risk**: low — the non-DTensor path is untouched, and the DTensor branch only makes the placement explicit.

---

## Suggested PR title

> [MoE] fix: preserve the router's shard placement when building the routing map (unblocks TP+EP)

## Suggested PR body

### Summary

`MoE.forward` builds the one-hot routing map with

```python
routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
    -1, topk_expert_ids_BLK, True
)
```

and the comment above it asserts that "scatter_ writes along the (replicated)
expert dim, so DTensor runs it as a local op with no redistribution". That is
not what DTensor does. When the router outputs are DTensors sharded on the
token dim (TP/SP with EP enabled), there is no sharding strategy for
`aten.scatter_` that preserves `Shard(dim=1)`:

- the **in-place** form errors out, and
- the out-of-place form would redistribute the operand to `Replicate`, which
  silently breaks the downstream `Partial(sum)` token-count contract that
  `RoutedExperts` declares — `num_local_tokens_per_expert_E` would then be a
  full-sequence count on every rank instead of a per-shard partial.

Net effect: TP+EP does not run.

### Fix

Do the scatter on the local shard and rewrap with the router's own placement,
so the sequence shard is preserved by construction:

```python
if isinstance(scores_BLE, DTensor):
    local_map = torch.zeros_like(
        scores_BLE.to_local(), dtype=torch.bool
    ).scatter_(-1, topk_expert_ids_BLK.to_local(), True)
    routing_map_BLE = DTensor.from_local(
        local_map, scores_BLE.device_mesh, scores_BLE.placements
    )
else:
    routing_map_BLE = torch.zeros_like(
        scores_BLE, dtype=torch.bool
    ).scatter_(-1, topk_expert_ids_BLK, True)
```

The plain-tensor path is byte-identical to today's code. The comment is
corrected too — it currently documents behaviour DTensor does not have.

### Evidence

A unit test in upstream's own style pins the contract: it fails on the current
construction and passes on the fixed one (see "Test plan"). The fix also
unblocks TP+EP and the full 4D `FSDP x TP x EP x PP` mesh on our Kimi-K3-like
MoE model (8 GPUs), where TP+EP was previously unreachable. See "Reproducer"
for what we could and could not run, and why.

### Reproducer

Any MoE model with `--parallelism.tensor_parallel_degree 2
--parallelism.expert_parallel_degree 2` (EP is carved out of the data-parallel
axes, so e.g. `dp_shard 2 * tp 2 = 4` ranks).

**Honest caveat on our own evidence.** We could not run this cell on an
upstream model on the hardware available (8x RTX 5060 Ti, sm_120, torch 2.12.0
stable). Both `deepseek_v3_debugmodel` and `qwen3_moe_debug` fail *before*
reaching the MoE, for two reasons unrelated to this PR:

1. `TypeError: create_block_mask() got an unexpected keyword argument
   'separate_full_blocks'` -- `models/common/decoder.py` passes a kwarg that
   only exists in torch nightly.
2. With that shimmed out locally, the compiled flex-attention kernel then
   fails with `OutOfMemoryError: out of resource:
   triton_tem_fused_flex_attention_0 Required: 139776 Hardware limit: 101376`
   -- consumer sm_120 cards do not have the shared memory the kernel config
   wants.

So the evidence we can stand behind is: the unit test in this kit (which
isolates exactly the placement contract and fails on the pre-fix
construction), plus the fix unblocking TP+EP and the full 4D
FSDP x TP x EP x PP mesh on our Kimi-K3-like MoE model on 8 GPUs. A maintainer
with datacenter cards (and/or nightly) should run the upstream-model cell
before merge -- it should reproduce as an error in the routing-map scatter.

### Test plan

- Unit test (included: `test_moe_routing_map_placement.py`, `DTensorTestBase`
  + `@with_comms`, world_size 2, the pattern of
  `tests/unit_tests/test_fsdp_moe_sharding.py`): build `Shard(1)` `scores` /
  `topk` DTensors, run the routing-map construction, and assert (a) the result
  keeps the input's placements, (b) its gathered value equals the plain-tensor
  result, (c) each rank holds LOCAL token counts -- the `Partial(sum)`
  contract. Verified to fail when pointed at the pre-fix construction.
- GPU integration: any MoE model with `--parallelism.tensor_parallel_degree 2
  --parallelism.expert_parallel_degree 2` (EP is carved out of the
  data-parallel axes, so e.g. `dp_shard 2 * tp 2 = 4` ranks).

---

## Notes for the filer

- This is a **core** `models/common` change, not an experiment change — it
  should be filed on its own and not bundled with any Kimi-K3 work. Per the
  maintainer history in CLAUDE.md, core-adjacent changes are best proposed as
  narrow, independently-justified fixes.
- The fork commit message already frames it as a core fix; reuse it, but lead
  with the upstream-model reproducer rather than the K3 model.
