# PR title: [Kimi K3] Tensor parallelism with sequence parallel, on the spmd_types declarations

PR branch `k3_tp_sp` on the fork (`c0ab32d07`, same commit as `tpsp_review3`: three commits on the declarations PR `k3_spmd_decl` `dbc60701d`, on upstream/main `390e2985b` with 4446 merged). Stacked: filed as a draft on top of the declarations PR and rebases when that one lands. CPU: 19 tests pass on the branch's files, pyrefly count equal to main's. The TP types were checked at tp2 with float32 per-parameter gradient dumps (see Results). Paste between the markers; the first line of the GitHub body says which PR it stacks on.

--- PASTE BEGIN ---

### Summary

Enables tensor parallelism for Kimi K3, with sequence parallel on the same mesh, on the declarations of the previous PR. Before this change `parallelize_kimi_k3` rejects `tensor_parallel_degree > 1` and the declarations are issued at tp = 1; after it TP comes off the unsupported list, `model.parallelize` also runs under it, and `parallelism.enable_sequence_parallel` (core default: on) decides whether the token stream between modules is Replicate or the TP-axis Shard(0). Both attention kinds are head-parallel: MLA on its head projections, KDA on its per-head state with Attention Gym's kernel running on the local heads behind a `local_map` on `inner_kda` (both declared in the base PR's `sharding.py`). The multimodal splice runs under sequence parallel, the latent MoE norm runs on the sequence shard the experts return, and `clip_grad_norm_` groups parameters by mesh so a model with undeclared modules under TP clips correctly.

### Design

- What the declarations give TP (base PR): MLA's `wq_b`, `wkv_b` and `gate` colwise, `wo` rowwise, the compressions `wq_a` / `wkv_a` replicated (rank-sized, not head-sized); KDA's `q_proj`, `k_proj`, `v_proj`, `forget_b`, `beta`, `output_gate` colwise, `output_proj` rowwise, the per-head state sharded with the heads, `inner_kda` behind a `local_map` with inputs declared feature- or head-sharded on TP and token-sharded on DP/CP; the stream weights (the residual projections, `output_res_proj`, `routed_up`) on `norm_config`'s rule, invariant without sequence parallel and replicated under it; the dense FFN on core's `set_dense_ffn_sharding`; the latent MoE's `routed_down` / `routed_up` replicated with `routed_norm` on the sequence shard.
- Sequence parallel (`model.py`): norms compute on the sequence shard; the attention module boundaries gather the shard on the way in (the attention core needs the full sequence) and `wo` / `output_proj` reduce-scatter back to Shard(0), the GQA pattern; `enable_sp` comes from the parallelism config.
- The multimodal splice under sequence parallel: *vision_positions* index the global token axis and a placeholder run can cross the shard boundary, so `parallelize_kimi_k3` hands the model its tp group and `_splice_under_sequence_parallel` gathers the stream (`spmd.redistribute` `S(0) -> R`, reduce-scatter backward) and splices on the whole sequence with the tokens, which sequence parallel leaves whole; `forward` cuts the shard back out (`R -> S(0)`, all-gather backward) past the DP-local vision region, where the checker records the stream's DP shard; a one-rank gloo test checks it against the whole-sequence splice and the gradient routing. Found by the tp4 cell, whose 64-token shards cut the debug image.
- Under type checking at tp > 1 the shared vision encoder mutated its position tables from `R` on every axis while the tower declares them invariant on tp, and `local_head_split` asserted heads sharded on tp; both keep the declared TP type now (`kimi_k2_7/vision_encoder.py`: only the dp axis changes; `common/attention.py`: an invariant input stays invariant through the view). After the DP-local vision region the stream is re-asserted on the sequence-parallel layout when SP is on.
- `clip_grad_norm_` (`distributed/utils.py`): parameters are grouped by mesh before the norm. A model with undeclared, hence replicated, modules under TP holds gradients on two meshes ((fsdp, tp) and (fsdp,)), and `get_total_norm`'s foreach stack refuses to mix them; disjoint groups combine exactly ((sum of norm^p)^(1/p), max for inf), the same algebra the EP path already uses, and the clip applies one scale group by group. With one mesh it is the single call it always was.

### Results

`kimi_k3_debugmodel` (multimodal), `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read; on an RTX 5060 Ti with Attention Gym's SM100/SM103 guard lifted locally. The first row names `partial_dtensor` (since 4446 the default backend is `spmd_types`), every other row runs under `spmd_types`; the last three run the flavor the way 4446's B200 cell does (type checking on, activation checkpointing off): dp1 and tp2 are bitwise their checked-off rows, the dp2 x ep2 x tp2 row moves from step 3 on with activation checkpointing off, as the declarations PR's dp2 rows do.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2
# sequence parallel off: add --parallelism.no-enable-sequence-parallel; expert parallel: --parallelism.expert_parallel_degree 2
```

The rows:

| cell | world | backend | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | 1 | partial_dtensor | 12.52977 | 7.27107 | 2.98077 |
| dp1 | 1 | spmd_types | 12.52977 | 7.27107 | 2.98077 |
| dp2 | 2 | spmd_types | 12.53137 | 7.31248 | 3.15823 |
| dp2 x ep2 | 2 | spmd_types | 12.53146 | 7.20212 | 3.10296 |
| tp2 (SP on) | 2 | spmd_types | 12.54164 | 7.39052 | 3.07310 |
| tp2 (SP off) | 2 | spmd_types | 12.55332 | 7.27805 | 3.05026 |
| tp4 (SP on) | 4 | spmd_types | 12.52816 | 7.12313 | 2.91429 |
| dp2 x tp2 (SP on) | 4 | spmd_types | 12.53383 | 7.25942 | 3.23869 |
| dp2 x ep2 x tp2 (SP on) | 4 | spmd_types | 12.53826 | 7.41038 | 3.29505 |
| dp1 | 1 | spmd_types, type checking, AC off | 12.52977 | 7.27107 | 2.98077 |
| tp2 (SP on) | 2 | spmd_types, type checking, AC off | 12.54164 | 7.39052 | 3.07310 |
| dp2 x ep2 x tp2 (SP on) | 4 | spmd_types, type checking, AC off | 12.53826 | 7.34445 | 3.37616 |

The correctness claim is the gradient, not the step-10 loss: at tp2, with and without sequence parallel, float32 end-to-end step-1 gradient dumps (a per-expert float32 loop for the experts) against dp1 show every replicated parameter bitwise identical across the two TP ranks and within noise of dp1 (744 and 742 of 750 parameters; the rest are `A_log` entries whose gradients are 1e-4-scale noise in bf16). The step-1 loss sits 0.1 to 0.2 percent from dp1 (the head-sharded matmuls and the boundary collectives round differently, and the MoE router is sensitive to it), and the later steps move by the few percent any re-partition of this flavor shows. The SP benefit case is long-sequence; at this scale it shows neither a memory win nor a speed cost worth reporting.

### Changed files

    torchtitan/models/kimi_k3/
      model.py              +70/-7   enable_sp from the parallelism config; the splice under sequence parallel; the stream's layout after the vision region; the latent MoE norm on the sequence shard
      parallelize.py        +14/-2   tensor parallel off the unsupported list; model.parallelize under TP; the tp group for the splice
    torchtitan/distributed/
      utils.py              +42/-4   clip_grad_norm_ grouped by parameter mesh
    torchtitan/models/common/
      attention.py          +7/-2    local_head_split keeps an invariant TP type
    torchtitan/models/kimi_k2_7/
      vision_encoder.py     +3/-6    the position tables keep their declared TP type under type checking
    tests/unit_tests/cpu/
      test_kimi_k3_sp_splice.py  +76/-0  the sequence-parallel splice on a one-rank group (new)

### CI/CD Coverage

One CPU test (the sequence-parallel splice through the real collectives on a one-rank group). A tp2 cell next to 4446's `kimi_k3_debugmodel_mm_fsdp2` on B200 is the natural addition; the type-checked tp2 row above is that cell run locally. Type checking passes at tp2 with and without sequence parallel and at dp2 x ep2 x tp2 (the last three rows).

--- PASTE END ---
