# PR title: [Draft] [Kimi K3] LoRA adapter export and QLoRA (NF4 and packed MXFP4)

Branch `k3_lora_extras` (`2344c1f9e`, base `30eb5e502`). Merges clean onto upstream/main `1dcb14a0c`; the 18 CPU tests pass on the merged tree. Paste between the markers into the PR body.

Upstream's only LoRA precedent is llama3's `float8_emulate_lora` flavor: one `LoRAConverter.Config(rank=8, alpha=16.0, target_modules=[...])` line, a CI cell in the features suite, and `components/lora.py` at 235 lines with no export and no quantization. Everything below the flavor is new core surface with no upstream counterpart; expect the reviewers to ask for the split noted after the markers.

--- PASTE BEGIN ---

### Summary

Extends core LoRA with the export path and QLoRA. Three pieces: `merge_lora_state_dict` / `trainable_state_dict` (a trained adapter is otherwise unexportable -- the raw state dict carries keys nothing downstream recognizes), 4-bit frozen bases (NF4 and packed MXFP4, the latter FSDP-shardable), and the packed bases under TP. Model-agnostic in `components/lora.py`; the Kimi K3 flavors exercise all of it, following the llama3 `float8_emulate_lora` flavor's shape. Unrelated to the QAT PR: that one fake-quantizes trainable masters through an STE; here MXFP4 is the real packed storage of frozen bases.

### Design

- The merge happens IN the modules -- the merged weight goes in as a fresh Parameter object, `state_dict()` is taken, and the original re-binds -- because a base linear's serialization is not necessarily `<fqn>.weight`: the fused attention linear exports split wq/wk/wv keys through a state-dict hook, and composing key names by hand misses every such hook. The object swap also keeps the returned dict from aliasing storage the restore reverts, and never writes THROUGH a quantized tensor (copy_ into an NF4 base would re-quantize the merged value). Wrapper segments (AC/FSDP/compile) differ between `named_modules()` and `state_dict()`; an unknown wrapper raises rather than guessing.
- `quantize_base='mxfp4'` swaps the base for split storage AT BUILD: qdata uint8 `[out, in/2]` plus e8m0-as-uint8 scale `[out, in/32]` (MXTensor itself cannot be a param -- non-contiguous logical view; the scale stores as uint8 because FSDP2's all-gather has no e8m0 copy kernel). Building packed means FSDP2 shards packed bytes natively -- the pack-then-shard order. From-scratch init draws each rank's rows locally and quantizes them, exact because MX block-32 is row-blockwise and commutes with Shard(0). Meta builds register the layout only; `scripts/quantize_lora_dcp.py` repacks an unquantized-flavor checkpoint into this layout (key map derived from the packed flavor built on meta), so no rank ever materializes the full bf16 model.
- `quantize_base='nf4'` (torchao) packs post-init and is library-scope: FSDP2's lazy_init cannot take a post-hoc packed param -- both a plain NF4 param (no `_local_tensor`) and NF4 inside the DTensor shell (invalid storage) were tried and refused; the error says so.
- `quantize_experts='mxfp4'` packs the grouped experts (the MoE parameter bulk) the same way, behind dequant properties so the grouped-GEMM forward is unchanged.
- Under TP: a TP-invariant base (rank-sized compressions) gets replicated adapters -- the third case next to colwise/rowwise. Packed colwise/rowwise bases run a packed-TP forward: local dequant + local matmul; colwise x and lora_a carry Partial grad placements (a bare to_local silently skips the tp reduction); rowwise reduces base+adapters in one collective and emits Partial with bias/tp, the declared contract. Expert weights TP-sharded on INNER dims refuse: expert TP splits the intermediate dim and the 2-D packed flatten cannot express that. Nothing here depends on an unmerged PR: adapter and packed-pair sharding derive from each linear's `sharding_config`, which core never sets today, so on current main the TP paths are constructed inert and only the CPU tests exercise them; they activate when the tensor-parallel PR lands.
- One core fix ships here: a fully frozen model part gets no optimizer. An adapter run's vision-tower stage has no LoRA targets (the MLLM convention keeps the tower frozen), and raising there made every frozen pipeline stage a hard error.

### Results

Training loss on the branch merged with current main, one seed, warmed compile cache. The adapters train (loss moves) while the bases stay frozen; the packed-MXFP4 flavor starts from a different step-1 value because its bases are quantized at build.

| cell | flavor | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | lora | 12.45603 | 11.93088 | 10.42394 |
| dp2 | lora | 12.48369 | 11.89999 | 10.51706 |
| dp1 | qlora_mxfp4 | 12.48328 | 12.00891 | 10.42474 |
| dp2 | qlora_mxfp4 | 12.50176 | 12.00203 | 10.45663 |

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_lora \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.data_parallel_shard_degree 2
# the packed-MXFP4 flavor: --config kimi_k3_debugmodel_qlora_mxfp4
```

<placeholder: peak memory, qlora_mxfp4 vs lora at dp2>

### Changed files

    torchtitan/components/
      lora.py                        +730/-2  export, NF4 and packed-MXFP4 bases, packed experts, packed TP
      optimizer/optimizer.py         +13/-0  a fully frozen model part gets no optimizer
    scripts/
      quantize_lora_dcp.py           +158/-0  repack an unquantized checkpoint into the packed layout (new)
    torchtitan/models/kimi_k3/
      config_registry.py             +87/-0  lora, qlora_mxfp4, qlora_mxfp4_linear flavors
    tests/unit_tests/cpu/
      test_lora.py                   +230/-0  merge key sets, hook keys, wrapper traversal, NF4 and MXFP4 round trips
    torchtitan/models/kimi_k3/tests/
      test_qlora_experts.py          +66/-0  packed experts build and merge (new)

### CI/CD Coverage

16 lora tests and 2 experts tests are CPU and run in the default suite. No GPU cell is added on this branch.

--- PASTE END ---

Notes for us, not for the body:

- Likely split if asked: (1) the K3 lora flavor plus `trainable_state_dict` / `merge_lora_state_dict`, which mirrors the llama3 flavor and closes a real gap; (2) QLoRA (NF4, packed MXFP4) with the repack script, a design the maintainers have to accept on its own; (3) the packed-TP forward, which belongs with the TP PR.
- A CI cell should sit on llama3 in the features suite (the core pieces are model-agnostic), not on K3, whose lane is B200-only after the CI reorg.
- The eleven-cell table across every parallelism family was measured on the integration tree where TP and EP are live; on this branch they are inert, so that table stays in the logbook (`LORA_QLORA_QAT_EVIDENCE`) rather than in the body.
