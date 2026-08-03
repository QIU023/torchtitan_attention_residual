# Non-standard / invented parts — review sheet

Honest accounting of the ~30% of this fork's diff that is NOT standard
framework composition. These are the pieces a maintainer should
scrutinize; each lists what it works around, upstream precedent, and
risk.

The ~70% that IS standard composition: TP/EP via
`set_moe_sharding_config` + `Module.parallelize` (verbatim deepseek_v3
pattern), FLOPs accounting, debugmodel CI, loss wiring, flavor
registration, the torch-2.12 compat shims (3 core files, do NOT belong
in the K3 PR).

## 1. Alpha graft gate (`attn_res_model.py`, ~40 lines)

- What: per-read scalar alpha (zero-init) gating each AttnRes read
  `h = plain + alpha*(mix - plain)`, plain = sequentially-threaded
  residual stream. alpha=0 => bit-exact identity with the plain
  backbone.
- Works around: grafting AttnRes onto a NON-AttnRes-pretrained base.
  The paper's ungated zero-init read is a uniform source-AVERAGE
  (measured 0.126 max|dlogit| off identity) -- fine from-scratch,
  wrong for a graft that must start == the checkpoint.
- Precedent: gated/near-identity init is standard (ReZero, LayerScale,
  gated MLA in K3). Application to AttnRes graft is ours.
- 7.27 impact: NONE directly -- K3 is AttnRes-NATIVE, never grafts.
  May CONFIRM the instinct; cannot invalidate (orthogonal).
- Risk: LOW as experiment code (opt-in flavor). As a general mechanism:
  maintainers may fold into a graft utility or reject as model-specific.
- Verified: debug bit-identity test (both directions) + 48B real-weight
  anchor (max|dlogit|=0.0, top-1 100%).

## 2. Module-level LoRA (`lora.py`, ~180 lines pre-NF4)

- What: `KimiLoRALinear` swap + base-freeze with the AttnRes-graft
  full-param exception + `trainable_state_dict`.
- Works around: upstream `LoRAConverter` is Config-tree based; the Kimi
  model is plain modules (same reason Float8 needed
  `KimiK3Float8Spec`).
- Precedent: semantics mirror LoRAConverter (kaiming A, zero B, scaling
  alpha/rank); torchtune module-swap LoRA is the close precedent.
- 7.27 impact: NONE -- titan-integration concern. Clean long-term fix:
  refactor the model onto the Config-tree so the upstream converter
  applies.
- Risk: MEDIUM. Wrapped FQNs (`q_proj`->`q_proj.base`) need TP-plan +
  FQN-consumer extension before LoRA+TP>1 (documented, not done). KDA
  excluded (fla reads `.weight` directly).
- Verified: P0 trio tests; 48B real-weight LoRA SFT end-to-end via veRL.

## 3. NF4 GroupedExperts subclass hack (`lora.py`, ~60 lines)

- What: pack 3-D `[E,A,B]` expert params to 2-D NF4 under a `*_nf4`
  param; a dynamic subclass adds dequant properties so `grouped_mm`
  reads logical-shape bf16; FSDP shards the packed bytes.
- Works around: torchao NF4 supports <=2-D only; GroupedExperts hold
  the 48B bulk (~42/48 GB).
- Precedent: torchtune NF4+FSDP2 proven for 2-D linears. The 3-D repack
  + dynamic-subclass dequant property is OURS, NO upstream precedent --
  highest-risk item.
- 7.27 impact: PARTIALLY SUPERSEDED. NF4 is not K3's format (MXFP4 is).
  The plumbing survives any format, but NF4 should be reframed as a
  customer-facing QLoRA option with a separate torchao `mx` (MXFP4)
  path for K3 fidelity.
- Risk: HIGH for upstream. `__class__` reassignment + `_parameters`
  renaming could break under future FSDP/torchao changes. Defensible
  ONLY as experiment code. Before upstreaming: replace with a proper
  3-D-capable tensor subclass or a torchao-mx GroupedExperts path.
- Verified: FSDP2 x NF4 2-GPU composition at debug scale.

## Bottom line

- Land #1, #2 in `experiments/kimi_k3/` (looser bar; off by default).
- Do NOT propose #3 upstream as-is -- keep as experiment-only QLoRA
  convenience; pursue torchao-mx MXFP4 for the K3-faithful story.
