### Summary

Extends core LoRA with the export path and QLoRA. Three pieces: `merge_lora_state_dict` / `trainable_state_dict` (a trained adapter is otherwise unexportable -- the raw state dict carries keys nothing downstream recognizes), 4-bit frozen bases (NF4 and packed MXFP4, the latter FSDP-shardable), and the packed bases under TP. Model-agnostic in `components/lora.py`; the Kimi K3 flavors exercise all of it.

### Design

- The merge happens IN the modules -- the merged weight goes in as a fresh Parameter object, `state_dict()` is taken, and the original re-binds -- because a base linear's serialization is not necessarily `<fqn>.weight`: the fused attention linear exports split wq/wk/wv keys through a state-dict hook, and composing key names by hand misses every such hook. The object swap also keeps the returned dict from aliasing storage the restore reverts, and never writes THROUGH a quantized tensor (copy_ into an NF4 base would re-quantize the merged value). Wrapper segments (AC/FSDP/compile) differ between `named_modules()` and `state_dict()`; an unknown wrapper raises rather than guessing.
- `quantize_base='mxfp4'` swaps the base for split storage AT BUILD: qdata uint8 `[out, in/2]` plus e8m0-as-uint8 scale `[out, in/32]` (MXTensor itself cannot be a param -- non-contiguous logical view; the scale stores as uint8 because FSDP2's all-gather has no e8m0 copy kernel). Building packed means FSDP2 shards packed bytes natively -- the pack-then-shard order. From-scratch init draws each rank's rows locally and quantizes them, exact because MX block-32 is row-blockwise and commutes with Shard(0). Meta builds register the layout only; `scripts/quantize_lora_dcp.py` repacks an unquantized-flavor checkpoint into this layout (key map derived from the packed flavor built on meta), so no rank ever materializes the full bf16 model.
- `quantize_base='nf4'` (torchao) packs post-init and is library-scope: FSDP2's lazy_init cannot take a post-hoc packed param -- both a plain NF4 param (no `_local_tensor`) and NF4 inside the DTensor shell (invalid storage) were tried and refused; the error says so.
- `quantize_experts='mxfp4'` packs the grouped experts (the MoE parameter bulk) the same way, behind dequant properties so the grouped-GEMM forward is unchanged.
- Under TP: a TP-invariant base (rank-sized compressions) gets replicated adapters -- the third case next to colwise/rowwise. Packed colwise/rowwise bases run a packed-TP forward: local dequant + local matmul; colwise x and lora_a carry Partial grad placements (a bare to_local silently skips the tp reduction); rowwise reduces base+adapters in one collective and emits Partial with bias/tp, the declared contract. Expert weights TP-sharded on INNER dims refuse: expert TP splits the intermediate dim and the 2-D packed flatten cannot express that.

### Evidence

Integration-tree matrices (multimodal debug flavor, one seed per table, warm cache, steps 1/3/10):

LoRA, every parallelism family the tree carries. PP required one core fix
that ships here: a fully frozen model part gets no optimizer (an adapter
run's vision-tower stage has no LoRA targets -- the MLLM convention keeps
the tower frozen -- and raising there made every frozen stage a hard
error). With it, pp2 x vp2 prints dp1's step-1 loss to every digit:

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.45474 | 11.94849 | 10.71402 |
| dp2 | 2 | 12.46697 | 11.82895 | 10.53144 |
| dp4 | 4 | 12.49144 | 12.08583 | 10.70239 |
| tp2 | 2 | 12.45324 | 11.93854 | 10.72911 |
| tp4 | 4 | 12.46047 | 11.96414 | 10.76673 |
| fsdp2 x tp2 | 4 | 12.46155 | 11.89166 | 10.47442 |
| cp2 | 2 | 12.44350 | 11.94324 | 10.70531 |
| fsdp2 x cp2 | 4 | 12.45038 | 11.84340 | 10.46684 |
| dp2 x ep2 | 2 | 12.46478 | 11.84854 | 10.47317 |
| dp4 x ep4 | 4 | 12.49350 | 12.07487 | 10.62631 |
| pp2 x vp2 | 2 | 12.45474 | 11.92275 | 10.68007 |

Branch worktree (the branch bases on upstream main, where the K3 TP/EP gates
are still on: its cells are data-parallel, and the TP/EP interaction code is
present but inert until those PRs land -- adapter sharding derives from
sharding_configs, which are None on upstream today):

| cell | flavor | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | lora | 12.61333 | 12.13142 | 10.95973 |
| dp2 | lora | 12.52952 | 12.21486 | 10.67500 |
| dp2 | qlora_mxfp4 | 12.56775 | 12.20790 | 10.98469 |

QLoRA (packed MXFP4, bases + experts): dp1 12.45288 / 11.96454 / 10.68903, dp2 12.45138 / 11.87566 / 10.37547, dp4 12.47301 / 12.00136 / 10.55821, dp2 x ep2 12.45138 / 11.87404 / 10.42552, dp4 x ep4 12.47301 / 11.99923 / 10.50310 -- the ep cells print the same step-1 loss as their dp twins, expert parallel is transparent at the forward; the full-flavor TP cells stop at the designed experts guard (expert TP shards the intermediate dim, which the 2-D packed flatten cannot express). QLoRA linears-only x tp2 (matrix standard, own seed): 12.54449 / 11.97425 / 10.69844. Checkpoint loop: an unquantized-LoRA seed checkpoint repacked by the script (217 weights: 148 linears + 69 expert tables) loads into the packed flavor under dp2/FSDP and trains.

Peak memory, full QLoRA vs LoRA at dp2: 3.20 vs 3.52 GiB -- the debug model is activation-dominated; the parameter-side ~4x cut shows at scale.

CPU: 16 lora tests + 2 experts tests (merge key-set exactness, serialization-hook keys, wrapper traversal, aliasing restore, NF4 pack/forward/merge, MXFP4 build-pack and merge round-trip).

### Changed files

    torchtitan/components/lora.py                 +~600
    torchtitan/components/optimizer/optimizer.py    +13  frozen parts skip
    scripts/quantize_lora_dcp.py                  +150
    torchtitan/models/kimi_k3/config_registry.py  +100  lora/qlora flavors
    tests/unit_tests/cpu/test_lora.py             +250
    torchtitan/models/kimi_k3/tests/test_qlora_experts.py +70
