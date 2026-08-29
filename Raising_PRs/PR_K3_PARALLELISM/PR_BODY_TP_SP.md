### Summary

Enables tensor parallelism for Kimi K3, with SequenceParallel on the same mesh. Before this change the model declares no TP sharding and `tensor_parallel_degree > 1` is rejected; after it the config tree carries the declarations, TP comes off the unsupported list, and `parallelism.enable_sequence_parallel` (core default: on) decides whether the token stream between modules is Replicate or the TP-axis Shard(0).

### Design

- Declarations live in `kimi_k3/sharding.py` following `qwen3_5/sharding.py`: declarations only, applied through the Module protocol; nothing touches a mesh or a device.
  - MLA shards on the head axis: `wq_b`/`wkv_b`/`gate` colwise, `wo` rowwise; the two compressions stay replicated -- they are rank-sized, not head-sized.
  - KDA is TP-invariant and declared so: its kernels never see a DTensor, and the declaration keeps every parameter on one mesh so `clip_grad_norm_` can stack them.
  - Norms and the attention-residual projections sit on the block stream, which TP does not split.
- Under SP the norms compute on the sequence shard, the MLA and KDA module boundaries gather it on the way in, and the rowwise outputs reduce-scatter back: the llama3 template.
  - KDA's delta recurrence consumes the whole sequence and slices back to the shard on the way out, so SP's activation saving does not extend into KDA bodies -- that is the recurrence, not a choice.
  - The multimodal splice is a token-indexed seam and redistributes to the full stream before scattering the tower's features: *vision_positions* index the global token axis, and slicing the mask per tp rank instead would hand the replicated tower disjoint partial gradients that nothing reduces over the tp axis.
- Grad-norm computation groups parameters by mesh so a mixed TP/replicated parameter set clips once, globally.

### Results

TBD: TP-only and TP x DP tables exist for the pre-SP head; the SP-on/SP-off matrix on this head is queued.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2
```

Training loss on `kimi_k3_debugmodel`, one seed, warmed compile cache:

| config | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | TBD | TBD | TBD |
| tp2 (SP on) | TBD | TBD | TBD |
| tp2 (SP off) | TBD | TBD | TBD |
| fsdp2 x tp2 | TBD | TBD | TBD |

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py           +205   the TP/SP declarations (new file, following
                                   qwen3_5/sharding.py)
      model.py             +41/-1  enable_sp consumption under the tp>1 gate;
                                   the splice redistributes to the full stream
      kda.py               +16/-7  kernel-boundary declarations for the KDA path
      dtensor_ops.py          +41  DTensor helpers the declarations use (new)
      parallelize.py        +3/-1  tensor parallel off the unsupported list
    torchtitan/distributed/
      utils.py             +42/-4  grad-norm computation grouped by parameter mesh
    tests/integration_tests/
      models.py             +4/-4  the multimodal model test runs fsdp2 x tp2
    torchtitan_recipes/tests/
      models.py             +2/-1  ditto for the recipe test list

### CI/CD Coverage

The multimodal debug-flavor integration test runs fsdp2 x tp2, which with the core default exercises the SP-on path; TBD whether the SP-off arm needs its own cell.
