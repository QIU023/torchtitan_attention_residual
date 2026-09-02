# PR title: [Draft] MXFP4-weight / MXFP8-activation fake-quant QAT for grouped experts

Branch `k3_qat` (`99defeab8`: the original commit, a merge of upstream/main `1dcb14a0c`, and the base-class alignment). 11 CPU tests pass on the merged tree. DEFERRED behind QB and LoRA -- see `phase13_k3like_48b_posttrain/QAT_UPSTREAM_AUDIT_2026-09-02.md`; if it goes up, it goes up as this draft, with the relation section, or it will sit like PR-3265. Results are a placeholder. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Fake-quant QAT for MoE grouped experts. Before this change the grouped-expert forward consumes the bf16 parameters directly; after it, with `MXFP4QATConverter` in the model's converters, the forward consumes `dequant(quant(w))` -- MXFP4 weights, MXFP8 activations, OCP microscaling block 32 -- through a straight-through estimator, while the bf16 masters keep training underneath. Quantization is emulated over torchao's MX primitives, so QAT runs on any GPU; FP4 hardware only speeds deployment. Kimi K3's released quantization scope (report sec 4.1.4) is the routed experts, which is what the converter targets, but nothing in it is model-specific.

### Design

- `MXFP4QATConverter` is a `QuantizationConverter` like the MXFP8 and NVFP4 converters and is exported with them. It matches on `GroupedExperts.Config` -- a type check rather than a name list -- and a model with no grouped experts raises rather than silently training unquantized.
  - K3's shipped `quantization_config` targets `Linear` with an ignore list that removes attention, shared experts, dense FFN, `lm_head` and the vision tower; in this module tree that set is exactly the grouped experts, so the isinstance check is the official scope.
- The fake-quantized weights shadow `_parameters` through the instance `__dict__` for the duration of forward only.
  - FSDP2's `reset_sharded_param` does `getattr` outside forward and needs the DTensor parameter back; renaming the masters would break the state-dict contract and the expert TP/EP layout, which key off these names.
- Under expert TP the block scales are per-block max-abs and `w2`'s blocked dim is the sharded one, so a shard that stops being block-divisible falls back to bf16 with a once-per-shape warning instead of a silent skip.
- Out-of-range values fall back elementwise to the high-precision value rather than emitting non-finites; emulated MX targets the OCP spec but is not verified bit-identical to vendor kernels -- MX-deployable, not bit-parity.
- This is the complement of QLoRA's really-packed frozen bases; the two do not compose on the same weights.
- CPU tests: STE identity gradient, unblockable passthrough, converter swap, forward-window restore, gradient flow to the masters.

### Relation to in-flight work

Three groups are moving this ground; this converter is the narrowest of them and is written to be folded rather than to compete.

- PR-3265 (torchao `QATConfig` post-build converter, seven schemes) and PR-3889 (NVFP4 quantization-aware distillation, self-contained fake-quant on experts and dense linears) are the other two fake-quant designs. If PR-3889 lands first, the MXFP4 element dtype and the experts-only scope here are what it lacks, and `_fake_quant_mx` (block-32 non-divisible fallback, non-finite fallback) is the contributable piece.
- PR-4368 / PR-4344 redefine the expert-weight lifecycle for real MXFP8 compute (FSDP owns qdata and scale after all-gather). This converter does not override the grouped-mm seam, so PR-4368 does not touch it; PR-4344 makes the forward-window shadowing a second mechanism on the same tensors, and the right form after it lands is an MXFP4 fake-quant policy in that lifecycle.

### Results

<placeholder: dp1 / dp2 rows on `kimi_k3_debugmodel_mx_qat`, steps 1 / 3 / 10, one seed, warmed compile cache, with the unquantized flavor as the control>

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_mx_qat \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.data_parallel_shard_degree 2
```

### Changed files

    torchtitan/components/quantization/
      mx_qat.py                     +211/-0  the converter and the STE fake-quant (new)
      __init__.py                   +2/-0  export
    torchtitan/models/kimi_k3/
      config_registry.py            +15/-0  the mx_qat flavor
      tests/test_mx_qat.py          +74/-0  (new)

### CI/CD Coverage

The QAT tests are CPU and run in the default suite; no GPU cell is added.

--- PASTE END ---
