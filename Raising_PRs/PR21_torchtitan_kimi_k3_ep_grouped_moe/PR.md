# PR #21 — Kimi K3: expert parallelism and the grouped-GEMM expert layout

**Target**: `pytorch/torchtitan`, on top of #4025
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
