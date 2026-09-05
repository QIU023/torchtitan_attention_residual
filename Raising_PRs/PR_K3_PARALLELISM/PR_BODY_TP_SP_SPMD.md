# PR title: [Draft] [Kimi K3] Tensor and sequence parallelism under spmd_types: the declarations the backend reads

Branch `tpsp_spmd_review1` on the fork (`2e2230cbb`, six commits on upstream/main `390e2985b`, which carries the merged 4446 "[spmd_types] Enable Kimi K3 backend"; rebased on 2026-09-05 with two conflicts in `parallelize.py`, both resolved in 4446's favour: its `annotate_replicated_parameters` seeding and its FSDP mesh resolution stay, the branch's own mesh block went, and `model.parallelize` now also runs when tensor parallel is on). One PR, as decided on 09-04: the declarations are what TP, SP and CP consume, so they ship with the parallelism that first needs them. 4446 runs K3 under `spmd_types` by annotating every parameter replicated and declaring only the MoE; this branch adds the declarations for the dense path, the tower and the TP/SP seams on top of it. CPU: 16 tests pass, pyrefly count equal to main's. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Enables tensor parallelism with sequence parallel on the same mesh for Kimi K3 under the `spmd_types` backend that 4446 turned on. Before this change the model declares sharding for its MoE only, every other parameter is annotated replicated (`annotate_replicated_parameters`), and `parallelize_kimi_k3` rejects `tensor_parallel_degree > 1`; after it the config tree carries the declarations for the dense path, the tower and the TP/SP seams in `kimi_k3/sharding.py` (declarations only, applied through the Module protocol, the qwen3_5 shape), `model.parallelize` also runs when tensor parallel is on, TP comes off the unsupported list, and `parallelism.enable_sequence_parallel` (core default: on) decides whether the token stream between modules is Replicate or the TP-axis Shard(0). Both attention kinds are head-parallel: MLA on its head projections, KDA on its per-head state with Attention Gym's kernel running on the local heads behind a `local_map` on `inner_kda`. dp1 and dp2 under `spmd_types` are bit-identical to `partial_dtensor` through ten seeded steps.

### Design

- MLA (`_set_mla_sharding`): `wq_b`, `wkv_b` and `gate` colwise, `wo` rowwise; the two compressions `wq_a` and `wkv_a` stay replicated because they are rank-sized, not head-sized, and `q_norm` / `kv_norm` are replicated state. The inner attention takes core's identity `local_map`. Under SP the module boundary gathers the sequence shard on the way in (the attention core needs the full sequence) and `wo` reduce-scatters back to Shard(0), the GQA pattern.
- KDA (`_set_kda_sharding`): the delta rule is independent per head, so `q_proj`, `k_proj`, `v_proj`, `forget_b`, `beta` and `output_gate` are colwise, `output_proj` rowwise, the per-head state (`A_log`, `dt_bias`, the depthwise conv weights) shards with the heads, and `inner_kda` runs behind a `local_map` whose inputs are declared feature- or head-sharded on TP and token-sharded on DP/CP. `forget_a`, the one low-rank compression, stays whole. Under SP the module boundary gathers once for every projection and `output_proj` reduce-scatters back.
- The block stream: norms compute on the sequence shard under SP; the attention- and ffn-residual projections and `output_res_proj` are replicated weights with no activation boundary (`_tp_replicate_config`, the replicated member of the colwise/rowwise family core does not have: declaring their boundaries would lift the input to a DTensor while `Linear.forward` unwraps its own weight). The dense FFN takes core's `set_dense_ffn_sharding`; the latent MoE keeps the EP declarations and adds `routed_down` / `routed_up` replicated with `routed_norm` on the sequence shard the experts hand back.
- Weight types the backend reads: weights whose inputs and consumers are TP-invariant (the attention- and ffn-residual projections, `output_res_proj`, `routed_up`) are declared `I` on TP; `R` would sum their identical gradients across TP. `routed_down`, `wq_a`, `wkv_a` and `forget_a` stay `R`: their consumers are TP-sharded and each rank's gradient is a partial sum.
- Stream conversions: the empty residual stack is converted `R -> I` at the model entry; the MLA module converts its input `I -> R`; `routed_norm` reduces the experts' Partial output at its boundary (keyed `x`, the argument `nn.RMSNorm.forward` takes); `routed_up` re-enters Partial so core's MoE exit reduces once; the MoE exit returns invariant.
- The MoonViT tower is declared invariant at TP and rank-local over cp, with the cp axis on every vision layout (main's `include_cp_axis=True` helpers, as muse_glimmer passes them) and its own attention entry, because the shared plan shards the tower over TP and all-gathers its k/v over cp. `preprocess_inputs` gives `pixel_values` and `grid_thw` the DP-local, TP-invariant layout every VLM decoder shares.
- Local regions: the MLA head unflatten and the rope join run in `spmd.local()` regions re-typed head-sharded on TP, with the head count derived from the projection width; KDA's head views use `-1`, since the projections hand back the TP-local head slice.
- The multimodal splice under sequence parallel: *vision_positions* index the global token axis and a placeholder run can cross the shard boundary, so `parallelize_kimi_k3` hands the model its tp group and `_splice_under_sequence_parallel` gathers the stream (`spmd.redistribute` `S(0) -> R`, reduce-scatter backward), splices on the whole sequence with the tokens, which sequence parallel leaves whole, and hands back the shard (`R -> S(0)`, all-gather backward); a one-rank gloo test checks it against the whole-sequence splice and the gradient routing. Found by the tp4 cell, whose 64-token shards cut the debug image.
- Every declaration is keyed by the `forward()` parameter names main uses (`q_THK`, `k_THK`, `v_THV`; `raw_gate_THK`, `raw_beta_TH`, `cu_seqlens` on `InnerKDA`): a `local_map` requires an entry for every positional parameter, and a key that matches nothing declares nothing.
- `clip_grad_norm_` (`distributed/utils.py`): parameters are grouped by mesh before the norm. A model with undeclared, hence replicated, modules under TP holds gradients on two meshes ((fsdp, tp) and (fsdp,)), and `get_total_norm`'s foreach stack refuses to mix them; disjoint groups combine exactly ((sum of norm^p)^(1/p), max for inf), the same algebra the EP path already uses, and the clip applies one scale group by group. With one mesh it is the single call it always was.

### Results

`kimi_k3_debugmodel`, `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read; on an RTX 5060 Ti with Attention Gym's SM100/SM103 guard lifted locally. The first row names `partial_dtensor` (since 4446 the default backend is `spmd_types`), every other row runs under `spmd_types`.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 2 --parallelism.spmd_backend spmd_types
# sequence parallel off: add --parallelism.no-enable-sequence-parallel; expert parallel: --parallelism.expert_parallel_degree 2
```

Running locally on the rebased head (the rows below are from `8e7d4998d`, before 4446; every cell ran twice on the same seed checkpoint and the second run is read). On that head dp1, dp2 and dp2 x ep2 under `spmd_types` were bit-identical to `partial_dtensor` through step 10, the tensor-parallel cells sat within 2.4e-2 of dp1 at step 1, and every TP mesh composed with data and expert parallel.

| cell | world | backend | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | 1 | partial_dtensor | 12.52977 | 7.27107 | 2.98077 |
| dp1 | 1 | spmd_types | 12.52977 | 7.27107 | 2.98077 |
| dp2 | 2 | spmd_types | 12.53137 | 7.31248 | 3.15823 |
| dp2 x ep2 | 2 | spmd_types | 12.53146 | 7.20212 | 3.10296 |
| tp2 (SP on) | 2 | spmd_types | 12.54164 | 7.35554 | 3.16327 |
| tp2 (SP off) | 2 | spmd_types | 12.55332 | 7.38015 | 3.00522 |
| tp4 (SP on) | 4 | spmd_types | 12.52816 | 7.03474 | 3.09639 |
| dp2 x tp2 (SP on) | 4 | spmd_types | 12.53383 | 7.26831 | 3.16722 |
| dp2 x ep2 x tp2 (SP on) | 4 | spmd_types | 12.53826 | 7.30627 | 3.10813 |

Step 1 under TP sits about 1e-2 from dp1 in either direction: the head-sharded matmuls and the boundary collectives round differently, and this model's step-1 loss is sensitive to it. The table is a correctness and composition claim; the SP benefit case is long-sequence, and at this scale it shows neither a memory win nor a speed cost worth reporting.

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py           ++313/-0   the TP/SP declarations (MLA, KDA, the block stream, the latent MoE seams), the weight types, the stream conversions, the tower's plan, the local-map keys
      model.py              ++210/-14  enable_sp from the parallelism config; the local regions; the multimodal input layout; the splice under sequence parallel
      parallelize.py        ++14/-2   model.parallelize under tensor parallel; the tp group for the splice; tensor parallel off the unsupported list
      kda.py                ++6/-3     cu_seqlens keyword-only on InnerKDA.forward; head views with -1
    torchtitan/distributed/
      utils.py              ++42/-4    clip_grad_norm_ grouped by parameter mesh
    tests/unit_tests/cpu/
      test_kimi_k3_sp_splice.py  ++76/-0  the sequence-parallel splice on a one-rank group (new)

### CI/CD Coverage

One CPU test (the sequence-parallel splice through the real collectives on a one-rank group). No GPU cell yet: the multimodal fsdp2 cell on B200 runs the model without TP; a tp2 cell on the debug flavor is the natural addition once the backend CI runs K3 under is settled.

--- PASTE END ---

Notes for us, not for the body: `PR_BODY_TP_SP.md` and `PR_BODY_SPMD.md` are the two halves this replaces; the CP PR (`cp_pr_candidate`) carries its own copy of the tp=1 declarations, so whichever lands second rebases over the other's `sharding.py`.
