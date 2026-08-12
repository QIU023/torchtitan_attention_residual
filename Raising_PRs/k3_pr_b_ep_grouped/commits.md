# PR-B backing work and extraction recipe

**Base**: PR-4025's head (`pr4025_latest`). See PR-A's `commits.md` for why the four
`k3_pr_*` branches are bookmarks rather than content, and why the split is by function
rather than by cherry-pick.

## Extraction list

From `parallelize.py`:

| symbol | lines | note |
|---|---|---|
| `apply_ep_kimi_k3` | 33 | the EP wiring |
| `_model_has_moe` | 4 | predicate |
| the `ep_degree > 1` explicit-prefetch block in `apply_fsdp` | ~15 | upstream's own comment says EP's device-to-host syncs defeat FSDP's implicit prefetch |

From `model.py`: the `MoE.Config` / `RoutedExperts.Config` construction that selects
`make_token_dispatcher_config(comm_backend="standard")`, and the `moe_enable_ep` branch.

The grouped-GEMM state-dict conversion (per-expert on disk, stacked in memory) is the part
the PR body flags as belonging behind a hook in 4025's `state_dict_adapter.py`. File the
ask before extracting, or this PR carries a mapping that will move.

## Must NOT come along

* `packed_mxfp4.py` -- MXFP4 weight import is its own PR (PR20).
* TP-side expert sharding (PR-A).

## Verification

Cells: `ep2_fsdp2`, `ep8_fsdp8`, `ep2_fsdp2_tp2_cp2`, `ep2_fsdp2_pp2_cp2`. Note
`ep8_fsdp8` is the one that exercises 8-way expert placement; the report's 896 experts are
config-level beyond that.

Independent of the matrix, `dep_ep_grad_probe.py` and `replicate_axis_check.py` in
`matrix_scripts/` are the per-parameter gradient checks for the EP axis.

## Status

Ready to extract. Second-order dependency on the 4025 state_dict_adapter hook.
