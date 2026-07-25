# Source commits (fork torchtitan `attention_residual_dev`)

| commit | scope |
|---|---|
| `129e29de` | `moe: preserve router shard placement in routing-map scatter (fixes TP+EP)` -- 1 file (`torchtitan/models/common/moe.py`), +17/-8. Core common-MoE change; nothing kimi-specific. |

Files here:

- `moe_routing_scatter_PR16.patch` -- the fix, as a git patch against
  `torchtitan/models/common/moe.py`. Applies cleanly to upstream main
  @ `fd277658` (the surrounding code is unchanged there).
- `test_moe_routing_map_placement.py` -- CPU/GPU unit test in upstream's own
  style (`DTensorTestBase` + `@with_comms`, world_size 2, the pattern used by
  `tests/unit_tests/test_fsdp_moe_sharding.py`). Asserts the routing map keeps
  the router's `Shard(1)` placement, that its gathered value equals the
  plain-tensor result, and that each rank holds LOCAL token counts (the
  `Partial(sum)` contract). Verified to FAIL when the helper is pointed at the
  pre-fix construction. Landed on the fork alongside the fix.

Upstream state verified 2026-07-25 against `pytorch/torchtitan` main
@ `fd277658`: `MoE.forward` still builds the routing map with a bare
`zeros_like(...).scatter_(...)`, and the comment above it still claims DTensor
runs it as a local op.
