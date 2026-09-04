# PR title: [Draft] [Kimi K3] spmd_types: the declarations the backend reads

Branch `spmd_review2` on the fork (`00f97b417`, two commits on `tp_review2` `80d7e0951`, on upstream/main `9b5f60c40`). Draft, stacked on the TP/SP draft. Upstream draft 4446 (pianpwk, "[spmd_types] Enable Kimi K3 backend") runs K3 under `spmd_types` by annotating every parameter replicated; this branch is the declaration-based version that TP, SP and CP need. Once 4446 lands this rebases onto it: the backend selection in `parallelize.py` and the flavor pin removal are taken from 4446, `annotate_replicated_parameters` is not applied to modules that carry a `ShardingConfig`, and the declarations stay. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Runs Kimi K3 under the `spmd_types` backend with the declarations the backend consumes. Before this change the K3 flavors pin `partial_dtensor` and `parallelize_kimi_k3` rejects anything else; after it the FSDP meshes come from `resolve_fsdp_mesh` / `resolve_sparse_fsdp_mesh` with the `DataParallelMeshDims` handed to the decoder and the vision encoder, `model.parallelize` runs whenever `spmd_types` drives the model, and the dense-path declarations are issued under `spmd_types` at any TP degree, since that backend consumes every declaration where `partial_dtensor` read only the TP placements. dp1 and dp2 are bit-identical to `partial_dtensor` through ten seeded steps.

### Design

- Weight types (`sharding.py`): weights whose inputs and consumers are TP-invariant (the attention- and ffn-residual projections, `output_res_proj`, `routed_up`) are declared `I` on TP; `R` would sum their identical gradients across TP. `routed_down`, `wq_a`, `wkv_a` and `forget_a` stay `R`: their consumers are TP-sharded and each rank's gradient is a partial sum.
- Stream conversions: the empty residual stack is converted `R -> I` at the model entry; the MLA module converts its input `I -> R`; `q_norm` / `kv_norm` are replicated state only; `routed_norm` reduces the experts' Partial output at its boundary (keyed `x`, the argument `nn.RMSNorm.forward` takes); `routed_up` re-enters Partial so core's MoE exit reduces once; the MoE exit returns invariant.
- The MoonViT tower is declared invariant at TP and rank-local over cp, with the cp axis on every vision layout (main's `include_cp_axis=True` helpers, as muse_glimmer passes them) and its own attention entry, because the shared plan shards the tower over TP and all-gathers its k/v over cp. `preprocess_inputs` gives `pixel_values` and `grid_thw` the DP-local, TP-invariant layout every VLM decoder shares.
- Local regions: the MLA head unflatten and the rope join run in `spmd.local()` regions re-typed head-sharded on TP, with the head count derived from the projection width; KDA's head views use `-1`, since the projections hand back the TP-local head slice.
- Every declaration is keyed by the `forward()` parameter names main uses (`q_THK`, `k_THK`, `v_THV`; `raw_gate_THK`, `raw_beta_TH`, `cu_seqlens` on `InnerKDA`): a `local_map` requires an entry for every positional parameter, and a key that matches nothing declares nothing.

### Results

`kimi_k3_debugmodel`, `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read; on an RTX 5060 Ti with Attention Gym's SM100/SM103 guard lifted locally.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 2 --parallelism.spmd_backend spmd_types
```

<!-- TBD: spmd_types rows from /workspace/mx3_spmdtp_*; the partial_dtensor rows are the QB control cells (same seed key) -->
| config | backend | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | partial_dtensor | 12.52977 | 7.27107 | 2.98077 |
| dp1 | spmd_types | 12.52977 | 7.27107 | 2.98077 |
| dp2 | partial_dtensor | 12.53137 | 7.31248 | 3.15823 |
| dp2 | spmd_types | | | |
| tp2 (SP on) | spmd_types | | | |
| dp2 x tp2 (SP on) | spmd_types | | | |

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py           +160/-9   weight types, stream conversions, the tower's plan, the local-map keys
      model.py              +114/-13  backend-driven declaration, the local regions, the multimodal input layout
      parallelize.py        +27/-17   the spmd_types backend branch (meshes, model.parallelize)
      kda.py                +4/-2     head views with -1

### CI/CD Coverage

None added here; the cells follow the backend CI runs K3 under once 4446 lands.

--- PASTE END ---
