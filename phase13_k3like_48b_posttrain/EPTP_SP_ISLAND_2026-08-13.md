# EP x TP: declaring the MoE as an SP island

Fixes the regression the upstream merge introduced in all three matrices
(`MERGE_GATE_EP_TP_2026-08-12.md`). Four attempts; the first three are recorded because
each one located the problem one level deeper, and two of them were worse than the bug.

## The mesh fact that explains everything

`torchtitan/distributed/parallel_dims.py:283`:

    efsdp = fsdp * self.tp // self.ep
    sparse_mesh = ["pp", "dp_replicate", "efsdp", "ep"]

**There is no tp axis in the sparse mesh.** Inside the MoE region the ranks that TP would
have used are folded into `efsdp` and used as additional FSDP shards for the experts. So
`tp` stops being a tensor-parallel axis there and becomes a **token axis** -- which is
exactly what `dense_sequence_parallel_placement()` says with
`partition_spec=(DP, (CP, TP), None)`, and why `#3970` could drop `tp_mesh` from
`wire_meshes` without anyone noticing.

Upstream models never meet the contradiction because TP implies SP for their whole
decoder (`enable_sp=parallelism.enable_sequence_parallel`, and they enable it with TP).
K3 does not: the decoder layer hands the FFN a tp-**Replicate** activation, by design --
plain-ish boundaries are what let PP's P2P, AttnRes's `torch.stack` and fla's triton
kernels work.

Before the merge this was invisible because `out_src` and `out_dst` for the routed
experts were the SAME variable, so the redistribution was an identity no-op and nothing
was ever validated. `#3996` split them into two expressions keyed on different flags, and
the EP-on / SP-off combination then asks DTensor for `S(1) -> P(sum)`, which it forbids.

## The fix: declare what is actually true

SP **inside** the MoE, Replicate **at its boundary**. Nothing about either side changes;
only the declaration becomes honest, and DTensor inserts the scatter on entry and the
all-gather on exit.

```python
set_moe_sharding_config(moe_cfg, enable_ep=..., enable_sp=ep and tp, ...)
if ep and tp:
    replicated = dense_activation_placement(tp=spmd.R)
    wanted = moe_cfg.sharding_config.in_dst_shardings["x_BLD"]
    moe_cfg.sharding_config = dataclasses.replace(
        moe_cfg.sharding_config,
        in_src_shardings={"x_BLD": replicated, "router_input_BLD": replicated},
        in_dst_shardings={"x_BLD": wanted, "router_input_BLD": wanted},
        out_dst_shardings=replicated,
    )
```

`in_src` describes what ARRIVES and `in_dst` what the module WANTS -- that distinction is
the whole fix, and it is expressible in the declarative vocabulary without touching a core
file. The override is on the config we already own.

### Two entry points, not one

K3's latent MoE (report Eq. 11) calls

    self._moe(self.latent.to_latent(x), router_input_BLD=x)

because the router reads the PRE-latent activation while the experts consume `W_down x`.
Upstream's config knows only about `x_BLD`, so `router_input_BLD` reached the router's gate
un-redistributed: Replicate arriving at a gate whose own declaration asked for SP. This is
an architectural feature of K3 that upstream's MoE has no notion of, and any future
declaration on this module has to name both.

## The three attempts that failed, and what each taught

| attempt | outcome | what it located |
|---|---|---|
| key `desired_experts_output_layout` on `enable_ep` (core) | error moved up one level to the MoE wrapper's own `out_src` check | three sites are keyed on `enable_sp` where the sources are keyed on `enable_ep` |
| key all three consistently (core) | **NCCL collective timeout -- a deadlock, worse than the error** | the disagreement is physical placement, not a label; declaring SP without distributing tokens that way hangs |
| declare only `x_BLD`'s boundary | error moved into `Linear.input ... expects Shard(dim=1)` | the router is fed by a second, undeclared entry point |

The deadlock is the most useful of the three: it ruled out the entire class of
"relabel it" fixes in one run. Both core patches were reverted; no core file is modified.

## Verification

`ep2_fsdp2_tp2_pp2` on the merged tree with the fix:

    12.05902 12.00313 11.75453 11.34475

byte-identical to the frozen pre-merge baseline for the same cell
(`baseline_pre_merge/full_13.txt`). Full three-arm verification is the gate for folding
the merge into `attention_residual_dev`, judged by
`matrix_scripts/compare_to_dev_baseline.py`.

## What this does NOT fix

* The 1e-5 drift on six cells, which comes from the merge itself (a reduction moved onto
  the device) and cannot be undone from our side.
* The text arm's fla SM120 failures, which are a kernel/hardware ceiling.
* The broader divergence: `kimi_k3/parallelize.py` is still the only file in
  `torchtitan/models/` calling `parallelize_module`, and `use_local_output` still appears
  20 times against once in all of upstream. EP x TP was the first time that bill came due;
  this fix pays that instalment without retiring the debt. See
  `DECLARATIVE_MIGRATION_2026-08-13.md` -- and note its size estimate is the part to
  re-check, since the migration unit turned out to be a whole residual stream (AttnRes
  injects two more sources into it), not a module.
