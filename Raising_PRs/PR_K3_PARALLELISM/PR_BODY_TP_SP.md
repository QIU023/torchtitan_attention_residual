# PR title: [Draft] [Kimi K3] Tensor parallelism: head-parallel MLA and KDA, sequence parallel on the same mesh

Branch `tp_review2` on the fork (`80d7e0951`, four commits on upstream/main `9b5f60c40`, the first commit after the expert-parallel merge). Draft on purpose: it is the base of the spmd_types declaration PR (`PR_BODY_SPMD.md`) and both rebase once upstream draft 4446 (K3 under spmd_types) lands. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Enables tensor parallelism for Kimi K3, with sequence parallel on the same mesh. Before this change the model declares no TP sharding and `tensor_parallel_degree > 1` is rejected; after it the config tree carries the declarations in `kimi_k3/sharding.py` (declarations only, applied through the Module protocol, the qwen3_5 shape), TP comes off the unsupported list, and `parallelism.enable_sequence_parallel` (core default: on) decides whether the token stream between modules is Replicate or the TP-axis Shard(0). Both attention kinds are head-parallel: MLA on its head projections, KDA on its per-head state with Attention Gym's kernel running on the local heads behind a `local_map` on `inner_kda`.

### Design

- MLA (`_set_mla_sharding`): `wq_b`, `wkv_b` and `gate` colwise, `wo` rowwise; the two compressions `wq_a` and `wkv_a` stay replicated because they are rank-sized, not head-sized, and `q_norm` / `kv_norm` are replicated state. The inner attention takes core's identity `local_map`. Under SP the module boundary gathers the sequence shard on the way in (the attention core needs the full sequence) and `wo` reduce-scatters back to Shard(0), the GQA pattern.
- KDA (`_set_kda_sharding`): the delta rule is independent per head, so `q_proj`, `k_proj`, `v_proj`, `forget_b`, `beta` and `output_gate` are colwise, `output_proj` rowwise, the per-head state (`A_log`, `dt_bias`, the depthwise conv weights) shards with the heads, and `inner_kda` runs behind a `local_map` whose inputs are declared feature- or head-sharded on TP and token-sharded on DP/CP. `forget_a`, the one low-rank compression, stays whole. Under SP the module boundary gathers once for every projection and `output_proj` reduce-scatters back.
- The block stream: norms compute on the sequence shard under SP; the attention- and ffn-residual projections and `output_res_proj` are replicated weights with no activation boundary (`_tp_replicate_config`, the replicated member of the colwise/rowwise family core does not have: declaring their boundaries would lift the input to a DTensor while `Linear.forward` unwraps its own weight). The dense FFN takes core's `set_dense_ffn_sharding`; the latent MoE keeps the EP declarations and adds `routed_down` / `routed_up` replicated with `routed_norm` on the sequence shard the experts hand back.
- The multimodal splice under SP: *vision_positions* index the global token axis and a placeholder run can cross the shard boundary, so the splice gathers the stream to Replicate, scatters the tower's features, and re-shards; under `partial_dtensor` that is a DTensor redistribute, and the tower's plain output is wrapped as a replicated DTensor first since `copy_` refuses the mix. Under `spmd_types` the stream is a plain shard, so the declaration PR stacked on this one carries the same gather in `_splice_under_sequence_parallel` (found by the tp4 cell, whose 64-token shards cut the debug image).
- `clip_grad_norm_` (`distributed/utils.py`): parameters are grouped by mesh before the norm. A model with undeclared, hence replicated, modules under TP holds gradients on two meshes ((fsdp, tp) and (fsdp,)), and `get_total_norm`'s foreach stack refuses to mix them; disjoint groups combine exactly ((sum of norm^p)^(1/p), max for inf), the same algebra the EP path already uses, and the clip applies one scale group by group. With one mesh it is the single call it always was.

### Results

`kimi_k3_debugmodel`, `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read. Measured on the spmd_types declaration branch stacked on this one (`spmd_review2`), under the `spmd_types` backend; the dp1 row is bit-identical to the same flavor under `partial_dtensor` through step 10. On an RTX 5060 Ti with Attention Gym's SM100/SM103 guard lifted locally.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 --parallelism.spmd_backend spmd_types
```

<!-- TBD: fill from /workspace/mx3_spmdtp_* -->
| config | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | | | |
| dp2 | 2 | | | |
| tp2 (SP on) | 2 | | | |
| tp2 (SP off) | 2 | | | |
| tp4 (SP on) | 4 | | | |
| dp2 x tp2 (SP on) | 4 | | | |

Step 1 under TP sits about 1e-2 from dp1 in either direction (the earlier measurement on the pre-rebase tree was 12.55057 for tp2 and 12.54164 for tp2 with SP against 12.52977): the head-sharded matmuls and the boundary collectives round differently, and this model's step-1 loss is sensitive to it. The table is a correctness and composition claim; the SP benefit case is long-sequence, and at this scale it shows neither a memory win nor a speed cost worth reporting.

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py           +162/-0  the TP/SP declarations: MLA, KDA, the block stream, the latent MoE seams
      model.py              +43/-3   enable_sp from the parallelism config; the splice under SP
      kda.py                +2/-1    cu_seqlens keyword-only on InnerKDA.forward
      parallelize.py        +1/-2    tensor parallel off the unsupported list
    torchtitan/distributed/
      utils.py              +42/-4   clip_grad_norm_ grouped by parameter mesh

### CI/CD Coverage

No cell yet: the multimodal fsdp2 cell on B200 runs the model without TP. A tp2 cell on the text debug flavor is the natural addition once the backend question (4446) settles which backend CI runs K3 under.

--- PASTE END ---
