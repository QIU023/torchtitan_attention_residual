### Summary

Adds tensor parallelism to the Kimi K3 model, declared through the sharding-config system and structured as qwen3_5's: `sharding.py` declares, `model.parallelize()` applies. Head-parallel on the MLA layers; KDA runs TP-invariant. Sequence parallel is deliberately not offered -- the stream stays Replicate on the TP axis and only head/feature axes shard.

### Design

- MLA is head-parallel: `wq_b`, `wkv_b` and the output gate split colwise on the head axis, `wo` rowwise; the two compressions (`wq_a`, `wkv_a`) stay whole because they are rank-sized, not head-sized. The FlexAttention body runs under the shared inner-attention `local_map` (`set_gqa_inner_attention_local_map`), the same helper qwen3_5's full-attention layers use, because FlexAttention indexes plain mask tensors.
- KDA is declared invariant, and has to be: its kernels are fla triton and never see a DTensor. Every projection and conv weight is declared Replicate on the TP axis (a weight-only config: declaring activation boundaries would lift the input while `Linear.forward` unwraps its own weight, meeting as "aten.mm got mixed"). The kernel call site unwraps each operand and re-wraps the result to the layout it received (`dtensor_ops.py`); each rank computes the recurrence redundantly.
- Every norm and residual projection on the block stream is declared replicated. Left undeclared they meet DTensor activations as plain tensors: the block-residual aggregation multiplies a norm weight and a projection weight together, and one declared, one not is a mixed mul.
- The latent MoE pair (`routed_down`, `routed_up`) stays whole -- it compresses to a rank, not heads or experts.
- The multimodal splice: under TP the embedding output is a DTensor while the replicated (undeclared, inert) vision tower returns a plain tensor, and the splice's `copy_` refuses the mix. The tower output is Replicate-consistent across the mesh -- replicated weights, same pixels on every rank -- so it is lifted with `DTensor.from_local` before the scatter: a wrap, not a transfer.
- One core change, `torchtitan/distributed/utils.py`: grad-norm computation groups parameters by mesh. With inert replicated modules under TP the model holds grads on two meshes ((fsdp, tp) and (fsdp,)), and `get_total_norm`'s foreach stack refuses to mix them. Disjoint groups combine exactly ((sum of norm^p)^(1/p); max for inf) -- the same algebra the existing EP path uses for its two groups -- and the clip applies the one total_norm group by group. With one mesh it is the single `get_total_norm` call it always was.

### K3 TP runs

To reproduce, from the torchtitan checkout root on this branch, 8 GPUs. Every cell loads the same seed checkpoint; run each cell twice and read the second run (a cold compile cache moves step 1). The runner we used, with the seed-load assertion and a disk gate, is https://github.com/QIU023/torchtitan_attention_residual/blob/611385d4e123d4d0527c6d08b06f8d701bb63e21/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh.

```sh
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 512 --training.max-context-length 512 --checkpoint.enable"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint --dump-folder seed
cell() { d=$1; n=$2; shift 2; rm -rf $d; mkdir -p $d; cp -r seed/checkpoint $d/; torchrun --nproc_per_node=$n $COMMON --training.steps 10 --metrics.log_freq 1 --checkpoint.interval 100000 "$@" --dump-folder $d; }
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
cell dp1 1 $D 1
cell tp2 2 $D 1 $T 2;  cell tp4 4 $D 1 $T 4;  cell tp8 8 $D 1 $T 8
cell dp2 2 $D 2;  cell fsdp2_tp2 4 $D 2 $T 2
```

`kimi_k3_debugmodel` (multimodal), seed 42, `--debug.deterministic`, one seed checkpoint loaded by every cell; the dp2 row is in because changing the data-parallel degree alone moves the loss more than the head split does (step 2: tp2/tp4/tp8 at 1.20e-1/1.76e-1/9.1e-2 from dp1, dp2 at 3.02e-1 from dp1; the mesh cell at 1.06e-1 from dp2).:

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.60370 | 7.40097 | 3.38761 |
| tp2 | 2 | 12.61424 | 7.14961 | 3.30040 |
| tp4 | 4 | 12.61019 | 7.13944 | 3.32639 |
| tp8 | 8 | 12.60325 | 7.14432 | 3.37905 |
| dp2 | 2 | 12.57909 | 7.55822 | 3.42322 |
| fsdp2 x tp2 | 4 | 12.59098 | 7.62463 | 3.42794 |

Not in this PR: sequence parallel, TP inside the vision tower (the tower is replicated under TP), and composition with the CP/PP/EP branches -- each composes on the integration tree and the boundary work rides with the later PRs. Without `tensor_parallel_degree > 1` none of this executes.

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py            +169  the TP declarations: MLA head-parallel, KDA
                                   invariant, the replicated stream modules
      dtensor_ops.py          +41  to_local_if_dtensor, the kernel-boundary unwrap
      kda.py                +16/-7 kernel call site unwraps and re-wraps
      model.py                 +19 the parallelize call and the splice lift
      parallelize.py          +3/-1 tensor parallel off the unsupported list
    torchtitan/distributed/utils.py  +42/-4  grad-norm grouped by parameter mesh
    tests/integration_tests/models.py        the multimodal cell runs fsdp2 x tp2
    torchtitan_recipes/tests/models.py       its configuration

### CI/CD Coverage

The existing multimodal model test extends from fsdp2 to fsdp2 x tp2 (4 GPUs), so the declarations, the kernel-boundary unwraps and the mesh-grouped grad norm all execute in CI.
