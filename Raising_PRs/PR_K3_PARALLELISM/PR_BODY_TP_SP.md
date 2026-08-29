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

Measured on this head (`ea021264`), 8x consumer GPUs, one seed checkpoint loaded by every cell; run each cell twice and read the second run (a cold compile cache moves step 1).

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2
```

Training loss on `kimi_k3_debugmodel`, one seed, warmed compile cache:

| config | world | step 1 | step 3 | step 10 | peak mem | tps |
|---|---|---|---|---|---|---|
| dp1 | 1 | 12.58962 | 8.12642 | 3.95057 | 11.94GiB | 540 |
| tp2 (SP on) | 2 | 12.58262 | 8.20421 | 3.97382 | 7.09GiB | 147 |
| tp2 (SP off) | 2 | 12.61339 | 8.64604 | 3.88164 | 7.11GiB | 167 |
| fsdp2 x tp2 | 4 | 12.58591 | 7.79146 | 3.51720 | 4.25GiB | 104 |
| tp4 | 4 | 12.59771 | 8.56692 | 3.98220 | 4.27GiB | 74 |
| fsdp2 x tp4 | 8 | 12.61042 | 8.04123 | 3.57420 | 2.72GiB | 61 |

At this debug scale SP shows no memory win and a small speed cost (7.09 vs 7.11 GiB, 147 vs 167 tps): the norm/activation share it saves is negligible at seq 512 while the boundary redistributes are real. The table is a correctness and composition claim, not a benefit claim; the benefit case is long-sequence.

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
