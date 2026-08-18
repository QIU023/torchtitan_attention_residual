# PR #21 — Kimi K3: expert parallelism and the grouped-GEMM expert layout

**Target**: `pytorch/torchtitan`, on top of #4025
**Scope**: `moe.py` (the SiTU-GLU routed experts, 53 lines now that it overrides a hook
instead of duplicating the base `forward`) and `quantile_balance.py` (the router-side load
balancing of report sec 2.3.3). `apply_ep_kimi_k3` and `verify_ep_applied` live in
`parallelize.py`, which lands once with PR22 as shared infrastructure since TP, EP and CP all
apply from it.
**Depends on**: the state_dict layout hook #4025 review asks for (per-expert
<-> grouped-GEMM). This PR is the other side of that hook.
**Risk**: medium — it changes how expert weights are stored, so checkpoint
compatibility is the thing to review. The model math is unchanged.

## The layout gap

The release stores routed experts per expert, as individual `nn.Linear`s
(`modeling_kimi_linear.py`):

    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)   # gate
    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)   # down
    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)   # up

A grouped-GEMM MoE needs them stacked on an expert dimension --
`w1_EFD`, `w2_EDF`, `w3_EFD` -- so it can run all experts in one kernel. Without
a conversion, either the released checkpoint cannot be loaded or the grouped
kernel cannot be used.

## What this adds

**`hf_key_map.py`** — bidirectional mapping across the whole released key set,
including the stacked-tensor indexing that makes the two layouts
interconvertible:

    language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight
      -> layers.1.ffn._moe.routed_experts.inner_experts.w1_EFD[3]

    language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight
      -> layers.1.ffn.latent.down.weight

The `[3]` is the point: one released key maps to a SLICE of one of our tensors,
so the reverse direction needs `expert_idx` and cannot be a pure string rewrite.
The map also resolves `g_proj` by layer type -- KDA layers keep `g_proj` while
MLA layers use `attn_gate_proj` -- which is why it takes `kda_layers`.

**`moe.py`** — `KimiSiTUGroupedExperts`, identical parameters to torchtitan's
`GroupedExperts` (`w1_EFD` / `w2_EDF` / `w3_EFD`); only the activation differs,
being K3's SiTU-GLU rather than SiLU.

**EP wiring** in `parallelize.py` — routed experts sharded on the ep mesh, with
the dense-vs-sparse split that keeps FSDP's mesh from overlapping EP's on the
same ranks.

## Evidence

- All 497,220 released keys map. `g_proj` resolution by layer type is what makes
  that complete rather than nearly complete.
- EP verified across the parallelism matrix: `ep2 x fsdp2`,
  `ep2 x fsdp2 x tp2 x pp2`, `ep2 x fsdp2 x tp2 x cp2`,
  `ep2 x fsdp2 x pp2 x cp2` all train, and per-parameter gradient checks show EP
  contributing nothing -- `ep2_fsdp2_pp2` equals `fsdp2_pp2` to five decimals,
  so enabling EP changes no digit.
- Re-checked after the expert class stopped duplicating the base `forward`. It used to
  copy the whole method -- DTensor unwrapping, offset cumsum, SPMD type mutation, three
  grouped-mm calls -- to change one line, and now overrides a `gate_up_combine` hook
  instead (92 -> 53 lines). That matters for EP specifically: the MXFP8 converter installs
  its quantized GEMM by overriding `_grouped_mm`, so a copied `forward` calling
  `torch._grouped_mm` directly would silently opt every routed expert out of it. With the
  hook the seam cannot drift. The full three-arm matrix passes on the hook version, and
  `ep2 x fsdp2` and `ep2 x fsdp2 x tp2 x cp2` converge normally over 10 steps
  (12.0509 -> 9.9162 and 12.0594 -> 11.8446).

## Related, already found and fixed upstream-facing

While verifying EP this fork found `moe_sharding.py` dropping the computed
`in_grad_placements` when EP is off, which silently under-reduces TP gradients
below the experts. That is filed separately (PR19) and reproduces on unmodified
`deepseek_v3`; it is not part of this PR.

## PASTE (the body that goes upstream)

---

The release stores routed experts per expert, as individual `nn.Linear`s, while a
grouped-GEMM MoE needs them stacked on an expert dimension (`w1_EFD`, `w2_EDF`,
`w3_EFD`) so all experts run in one kernel. Without a conversion either the released
checkpoint will not load or the grouped kernel cannot be used, so this adds the mapping
and the EP wiring on top of it.

`hf_key_map.py` covers the whole released key set in both directions. The part that is
not a string rewrite is the stacked-tensor indexing: one released key such as
`...experts.3.w1.weight` maps to a slice, `...inner_experts.w1_EFD[3]`, so the reverse
direction needs an `expert_idx`. The map also resolves `g_proj` by layer type, since KDA
layers keep `g_proj` while MLA layers use `attn_gate_proj`, which is why it takes
`kda_layers`. All 497,220 released keys map, and the by-layer-type resolution is what
makes that complete rather than nearly complete.

The expert class overrides one hook. It carries the same parameters as `GroupedExperts`
and differs only in the GLU variant, K3's SiTU-GLU rather than SiLU. It used to copy the
whole base `forward` -- DTensor unwrapping, offset cumsum, SPMD type mutation and all
three grouped-mm calls -- to change that one line, and is 53 lines instead of 92 now that
`GroupedExperts` exposes `gate_up_combine`. That hook is a separate PR and matters here
specifically because the MXFP8 converter installs its quantized GEMM by overriding
`_grouped_mm`: a copied forward calling `torch._grouped_mm` directly would silently opt
every routed expert out of it.

EP wiring shards the routed experts on the ep mesh, with the dense-vs-sparse split that
keeps FSDP's mesh from overlapping EP's on the same ranks.

Verified across ep2 x fsdp2, ep2 x fsdp2 x tp2 x pp2, ep2 x fsdp2 x tp2 x cp2 and
ep2 x fsdp2 x pp2 x cp2. Per-parameter gradient checks show EP contributing nothing:
ep2_fsdp2_pp2 equals fsdp2_pp2 to five decimals, so enabling EP changes no digit. On the
hook version specifically, ep2 x fsdp2 and ep2 x fsdp2 x tp2 x cp2 converge normally over
10 steps, 12.0509 -> 9.9162 and 12.0594 -> 11.8446.

Checkpoint compatibility is the thing to review here; the model math does not change.

## Note: ep8_fsdp8 is NOT taken from the gate

The gate's mm_full arm runs the same flavor and would seem to hand this cell over for
free, and its numbers are 12.01898 -> 9.90946 over 10 steps, next to ep2_fsdp2's
12.0509 -> 9.9162. Close enough to look like the third row of the same table.

It is not one. The gate sets `KIMI_VIT_PREFETCH=1` on that arm and `run_cells.sh`, which
produced the other two rows, does not. Prefetch is documented numerically inert, but that
was established on four pp8xvp4 pairs, not on an ep8 layout, so putting the three side by
side would be asserting something nobody checked.

This is the failure `run_cells.sh`'s own header records -- the same tree gave 12.04691
under one set of defaults and 12.07827 under the gate's, and the gap was briefly read as
a defect in the adapter. Re-run this cell through `run_cells.sh` so all three rows share
their knobs, rather than annotating the difference away.
