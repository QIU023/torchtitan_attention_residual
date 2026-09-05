# PR title: [Kimi K3] Tensor parallelism with sequence parallel, on the spmd_types declarations

Branch `tpsp_review3` on the fork (`1fe86490f`, two commits on `spmd_decl_review1` `010c96e45`, the declarations PR, on upstream/main `390e2985b` with 4446 merged). Stacked: it is filed after the declarations PR and rebases when that one lands. The matrix below runs on it; the previous head `ebd83e941` (byte-identical to `tpsp_spmd_review1` `2e2230cbb`) differs only by the base's tp > 1 guard on the entry conversion. CPU: 13 tests pass on the branch's files, pyrefly count equal to main's. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Enables tensor parallelism for Kimi K3, with sequence parallel on the same mesh, on the declarations of the previous PR. Before this change `parallelize_kimi_k3` rejects `tensor_parallel_degree > 1` and the declarations are issued at tp = 1; after it TP comes off the unsupported list, `model.parallelize` also runs under it, and `parallelism.enable_sequence_parallel` (core default: on) decides whether the token stream between modules is Replicate or the TP-axis Shard(0). Both attention kinds are head-parallel: MLA on its head projections, KDA on its per-head state with Attention Gym's kernel running on the local heads behind a `local_map` on `inner_kda` (both declared in the base PR's `sharding.py`). The multimodal splice runs under sequence parallel, the latent MoE norm runs on the sequence shard the experts return, and `clip_grad_norm_` groups parameters by mesh so a model with undeclared modules under TP clips correctly.

### Design

- What the declarations give TP (base PR): MLA's `wq_b`, `wkv_b` and `gate` colwise, `wo` rowwise, the compressions `wq_a` / `wkv_a` replicated (rank-sized, not head-sized); KDA's `q_proj`, `k_proj`, `v_proj`, `forget_b`, `beta`, `output_gate` colwise, `output_proj` rowwise, the per-head state sharded with the heads, `inner_kda` behind a `local_map` with inputs declared feature- or head-sharded on TP and token-sharded on DP/CP; the block stream's residual projections replicated with no activation boundary; the dense FFN on core's `set_dense_ffn_sharding`; the latent MoE's `routed_down` / `routed_up` replicated with `routed_norm` on the sequence shard.
- Sequence parallel (`model.py`): norms compute on the sequence shard; the attention module boundaries gather the shard on the way in (the attention core needs the full sequence) and `wo` / `output_proj` reduce-scatter back to Shard(0), the GQA pattern; `enable_sp` comes from the parallelism config.
- The multimodal splice under sequence parallel: *vision_positions* index the global token axis and a placeholder run can cross the shard boundary, so `parallelize_kimi_k3` hands the model its tp group and `_splice_under_sequence_parallel` gathers the stream (`spmd.redistribute` `S(0) -> R`, reduce-scatter backward), splices on the whole sequence with the tokens, which sequence parallel leaves whole, and hands back the shard (`R -> S(0)`, all-gather backward); a one-rank gloo test checks it against the whole-sequence splice and the gradient routing. Found by the tp4 cell, whose 64-token shards cut the debug image.
- `clip_grad_norm_` (`distributed/utils.py`): parameters are grouped by mesh before the norm. A model with undeclared, hence replicated, modules under TP holds gradients on two meshes ((fsdp, tp) and (fsdp,)), and `get_total_norm`'s foreach stack refuses to mix them; disjoint groups combine exactly ((sum of norm^p)^(1/p), max for inf), the same algebra the EP path already uses, and the clip applies one scale group by group. With one mesh it is the single call it always was.

### Results

`kimi_k3_debugmodel` (multimodal), `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read; on an RTX 5060 Ti with Attention Gym's SM100/SM103 guard lifted locally. The first row names `partial_dtensor` (since 4446 the default backend is `spmd_types`), every other row runs under `spmd_types`; the last three run the flavor the way 4446's B200 cell does (type checking on, activation checkpointing off).

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2
# sequence parallel off: add --parallelism.no-enable-sequence-parallel; expert parallel: --parallelism.expert_parallel_degree 2
```

Running locally, the rows follow:

| cell | world | backend | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | 1 | partial_dtensor | | | |
| dp1 | 1 | spmd_types | | | |
| dp2 | 2 | spmd_types | | | |
| dp2 x ep2 | 2 | spmd_types | | | |
| tp2 (SP on) | 2 | spmd_types | | | |
| tp2 (SP off) | 2 | spmd_types | | | |
| tp4 (SP on) | 4 | spmd_types | | | |
| dp2 x tp2 (SP on) | 4 | spmd_types | | | |
| dp2 x ep2 x tp2 (SP on) | 4 | spmd_types | | | |
| dp1 | 1 | spmd_types, type checking, AC off | | | |
| tp2 (SP on) | 2 | spmd_types, type checking, AC off | | | |
| dp2 x ep2 x tp2 (SP on) | 4 | spmd_types, type checking, AC off | | | |

On the previous head (`8e7d4998d`, before 4446) dp1, dp2 and dp2 x ep2 under `spmd_types` were bit-identical to `partial_dtensor` through step 10, and the tensor-parallel cells sat within 2.4e-2 of dp1 at step 1: the head-sharded matmuls and the boundary collectives round differently, and this model's step-1 loss is sensitive to it. The table is a correctness and composition claim; the SP benefit case is long-sequence, and at this scale it shows neither a memory win nor a speed cost worth reporting.

### Changed files

    torchtitan/models/kimi_k3/
      model.py              +59/-4   enable_sp from the parallelism config; the splice under sequence parallel; the latent MoE norm on the sequence shard
      parallelize.py        +14/-2   tensor parallel off the unsupported list; model.parallelize under TP; the tp group for the splice
    torchtitan/distributed/
      utils.py              +42/-4   clip_grad_norm_ grouped by parameter mesh
    tests/unit_tests/cpu/
      test_kimi_k3_sp_splice.py  +76/-0  the sequence-parallel splice on a one-rank group (new)

### CI/CD Coverage

One CPU test (the sequence-parallel splice through the real collectives on a one-rank group). A tp2 cell next to 4446's `kimi_k3_debugmodel_mm_fsdp2` on B200 is the natural addition; the type-checked rows above are that cell run locally.

--- PASTE END ---
