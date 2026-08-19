# PR #21 — Kimi K3: expert parallelism and the grouped-GEMM expert layout

**Target**: `pytorch/torchtitan`, `main`. Draft.
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
  `ep2 x fsdp2`, `ep2 x fsdp2 x tp2 x cp2` and `ep8 x fsdp8` converge normally over 10
  steps: 12.0509 -> 9.9162, 12.0594 -> 10.2364, and 12.0190 -> 9.9095.

## Related, already found and fixed upstream-facing

While verifying EP this fork found `moe_sharding.py` dropping the computed
`in_grad_placements` when EP is off, which silently under-reduces TP gradients
below the experts. That is filed separately (PR19) and reproduces on unmodified
`deepseek_v3`; it is not part of this PR.

## PASTE (the body that goes upstream)

---

Draft. This branch is our Kimi K3 implementation carrying the expert parallel plan; the other two axes raise in the parallelize entry, so what is under review here is EP and the model it needs.

PR-4025 is a separate implementation of the same model, further along on the model itself and with all four parallelism axes raising `NotImplementedError`. When it lands this rebases onto it and the diff narrows to the EP plan alone. It is a draft now so the axis work is visible while that happens -- and so the three axis PRs can be read as a set, since today each carries the model and they overlap heavily for that reason.

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

Per-parameter gradient checks show EP contributing nothing: with the same seed
checkpoint, enabling EP changes no digit to five decimals.

Ten steps on three model arms -- text, multimodal, multimodal plus LoRA -- first and last
loss:

    ep2 x fsdp2   text 7.70385 -> 4.87498   mm 12.05090 -> 9.91618   mm+lora 12.03633 -> 11.89115
    ep8 x fsdp8   text 7.71120 -> 4.87902   mm 12.01898 -> 9.90946   mm+lora 12.05817 -> 11.86901

Those are the ep-only cells; ep8 is the full 8-rank case. This branch carries no TP or CP
plan, so combinations with them belong to the sibling PRs rather than here.

Checkpoint compatibility is the thing to review here; the model math does not change.

## Where the ep8_fsdp8 row comes from, and one number that was wrong

`11.8446` appeared in an earlier draft of this evidence as the endpoint of
`ep2 x fsdp2 x tp2 x cp2`. It is that run's step 3. The step 10 value is `10.2364`. Caught
by running the same cell in the gate and getting a different last column against an
identical first one.

The `ep8 x fsdp8` row is taken from the gate's mm_full arm rather than from a separate
run. That needed checking, because the gate sets `KIMI_VIT_PREFETCH=1` on that arm and
`run_cells.sh`, which produced the other two rows, does not -- and `run_cells.sh`'s own
header records a case where exactly that kind of default mismatch made one tree read
12.04691 and 12.07827 and got taken for an adapter defect.

Checked instead of assumed: the two cells the two runs have in common are identical at
every one of 10 steps to all 7 digits.

    ep2_fsdp2          12.05090 11.97917 11.79904 11.50565 11.03584 10.55429 10.23278 10.05370 10.00390 9.91618
    ep2_fsdp2_tp2_cp2  12.05941 11.98916 11.84457 11.58146 11.23144 10.80611 10.44410 10.30850 10.29510 10.23644

Prefetch is documented numerically inert, but on four pp8xvp4 pairs; this extends that to
these EP layouts, which is what licenses reading the third row off the gate.
