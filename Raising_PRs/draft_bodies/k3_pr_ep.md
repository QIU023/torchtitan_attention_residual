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
