# PR #16 — common MoE: routing-map scatter drops the router's shard placement (breaks EP under the declarative SPMD backends)

**Target**: `pytorch/torchtitan`, `torchtitan/models/common/moe.py` (`MoE.forward`)
**Fork reference**: `attention_residual_dev` @ `129e29de`
**Upstream audit**: 2026-08-03 @ `681fd4b50` — bare `scatter_` still at `moe.py:465`, patch applies cleanly.
**Trigger condition (settled 2026-08-04)**: `--parallelism.spmd_backend full_dtensor|spmd_types` + EP. Under these backends `_router_gate_config` declares the router output `DTensor(Shard(1))` whenever `enable_ep` (its own docstring: "EP on: ... output DTensor(Shard(1))"), and that placement reaches the bare `scatter_`. The **default** backend routes TP/EP through the classic path where the router output never arrives sharded — verified live: default-backend DSv3 tp2+ep2 trains fine on pristine upstream (loss 8.01456 / 6.19790). Our K3 model wires the declarative path unconditionally, which is why it crashed there first.
**Parallel-safe with PR19** (different files, no overlap). **Risk**: low — plain-tensor path byte-identical.

---

## Suggested PR title

> [MoE] fix: preserve the router's shard placement when building the routing map (EP under full_dtensor/spmd_types)

## Suggested PR body

### Summary

`MoE.forward` builds the one-hot routing map with

```python
routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
    -1, topk_expert_ids_BLK, True
)
```

and the comment above it says DTensor runs this as a local op. Under the
declarative SPMD backends that is not what happens: with EP enabled,
`_router_gate_config` declares the router output `DTensor(Shard(1))` (its
docstring: "EP on: ... output DTensor(Shard(1))"), and no `aten.scatter_`
strategy preserves `Shard(dim=1)` — the in-place form errors out, and the
out-of-place form would redistribute to `Replicate`, silently breaking the
`Partial(sum)` token-count contract `RoutedExperts` declares downstream. So
the sharding config promises a placement the forward cannot carry through.
The default backend is unaffected (router outputs never arrive sharded there).

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

Unmodified `deepseek_v3_debugmodel` on 4 ranks, declarative backend:

```bash
NGPU=4 ./run_train.sh --model.name deepseek_v3 --model.flavor debugmodel \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.expert_parallel_degree 2 \
  --parallelism.spmd_backend full_dtensor
```

Errors in the routing-map construction at step 1; with this patch it trains.
The same cell on `--parallelism.spmd_backend default` trains with or without
the patch (router outputs are not sharded on that path), which bounds the
blast radius. The fix also unblocks EP and the full FSDP x TP x EP x PP mesh
on our Kimi-K3-family MoE model (8 GPUs), which uses the declarative MoE
sharding path unconditionally.

### Test plan

Unit test included (`test_moe_routing_map_placement.py`, `DTensorTestBase` +
`@with_comms`, world_size 2, same pattern as `test_fsdp_moe_sharding.py`):
builds `Shard(1)` router outputs directly, asserts the routing map keeps the
placement, gathers to the plain-tensor result, and that each rank holds LOCAL
token counts. Fails on the pre-fix construction; passes after. Plus the
integration cell above.

---

## Notes for the filer

- **Pre-filing gate (mandatory, updated 2026-08-04)**: the default-backend
  half is DONE (trains clean on pristine upstream, numbers above). Still to
  run once: the same cell with `--parallelism.spmd_backend full_dtensor`
  (and `spmd_types` if the first errors somewhere unrelated — both route
  `model.parallelize(parallel_dims)`). Expected: error inside the routing-map
  construction at step 1; then apply the patch and record step-1 loss. Paste
  both into the body. If it errors before reaching the MoE for backend
  reasons unrelated to this fix, say so in the body and lean on the unit
  test and the K3-observed crash; if it does not error at all, STOP and
  re-scope.
- The declarative backends are upstream's active direction (deepseek_v3 has
  full sharding-config wiring; `transformers_modeling_backend` routes MoE
  through `set_moe_sharding_config`) — this is a core fix for that path, not
  a K3 accommodation. Keep that framing.
- File on its own; can open the same day as PR19 — cross-link once numbers
  exist.
