# PR #16 — common MoE: routing-map scatter drops the router's shard placement (breaks TP+EP)

**Target**: `pytorch/torchtitan`, `torchtitan/models/common/moe.py` (`MoE.forward`)
**Fork reference**: `attention_residual_dev` @ `129e29de`
**Upstream audit**: 2026-07-25 @ `fd277658` and 2026-08-03 @ `681fd4b50` — bare `scatter_` still at `moe.py:465`, patch applies cleanly. **Parallel-safe with PR19** (different files, no overlap).
**Risk**: low — the plain-tensor path is byte-identical; the DTensor branch only makes the placement explicit.

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

The comment above it says DTensor runs this as a local op. It does not: when
the router outputs are sharded on the token dim (TP with EP), no
`aten.scatter_` strategy preserves `Shard(dim=1)` — the in-place form errors
out, and the out-of-place form would redistribute to `Replicate`, silently
breaking the `Partial(sum)` token-count contract `RoutedExperts` declares.
Net effect: TP+EP does not run.

### Fix

Scatter on the local shard and rewrap with the router's own placement:

```python
if isinstance(scores_BLE, DTensor):
    local_map = torch.zeros_like(
        scores_BLE.to_local(), dtype=torch.bool
    ).scatter_(-1, topk_expert_ids_BLK.to_local(), True)
    routing_map_BLE = DTensor.from_local(
        local_map, scores_BLE.device_mesh, scores_BLE.placements
    )
else:
    ...  # unchanged plain-tensor path
```

The stale comment is corrected as well.

### Reproducer

Unmodified `deepseek_v3_debugmodel` on 4 ranks:

```bash
NGPU=4 ./run_train.sh --model.name deepseek_v3 --model.flavor debugmodel \
  --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2
```

Errors in the routing-map construction at step 1; with this patch it trains.
The fix also unblocks the full FSDP x TP x EP x PP mesh on our Kimi-K3-family
MoE model (8 GPUs).

### Test plan

Unit test included (`test_moe_routing_map_placement.py`, `DTensorTestBase` +
`@with_comms`, world_size 2, same pattern as `test_fsdp_moe_sharding.py`):
asserts the routing map keeps the router's `Shard(1)` placement, gathers to
the plain-tensor result, and each rank holds LOCAL token counts. Fails on the
pre-fix construction; passes after. Plus the integration cell above.

---

## Notes for the filer

- **Pre-filing gate (mandatory)**: the DSv3 tp2+ep2 cell above has not been
  executed yet on our box — run it once before filing (deepseek_v3_debugmodel
  is known to run there at tp2/tp4 since the 07-31 sessions, and this bug
  triggers in the MoE forward, before anything model-specific). Paste the
  actual pre-fix error text and the post-fix step-1 loss into the PR body. If
  the cell behaves differently, stop and re-scope.
- File on its own; do not bundle with any Kimi-K3 work. Can be opened the same
  day as PR19 — cross-link once numbers exist.
