# PR title: [Kimi K3] The declarations spmd_types reads: the dense path, the tower over cp, the MoE seams, the multimodal inputs

PR branch `k3_spmd_decl` on the fork (`dbc60701d`, same commit as `spmd_decl_review1`; one commit on upstream/main `390e2985b`, main with 4446). The declarations commit the CP PR carries as `8c8d9436f`, lifted onto main after 4446. The TP/SP PR (`tpsp_review3`, three commits) and the CP PR stack on it. CPU: 18 tests pass (the three varlen failures are main's), pyrefly count equal to main's. Paste between the markers.

--- PASTE BEGIN ---

### Summary

- Declares, for the `spmd_types` backend 4446 turned on, what 4446 left to `annotate_replicated_parameters`: the dense path's weight types and stream conversions, the MoonViT tower over cp, the latent MoE's seams, the multimodal inputs.
- Before: only the MoE modules declare layouts, and the multimodal debug flavor stops at step 1 under the now-default backend with `ValueError: spmd_types backend requires an SPMD layout for every tensor input, but these have no entry in input_sharding: ['grid_thw', 'pixel_values']` (the path of 4446's `kimi_k3_debugmodel_mm_fsdp2` cell).
- After: `kimi_k3/sharding.py` carries the declarations (declarations only, applied through the Module protocol, the qwen3_5 shape), issued at tp = 1 where `spmd_types` reads every one of them; `preprocess_inputs` gives `pixel_values` and `grid_thw` the DP-local, TP-invariant layout every VLM decoder shares.
- Tensor parallelism stays on the unsupported list here; the TP/SP PR stacked on this one turns it on.

### Design

- Stream weights follow `norm_config`'s rule. Without SP the stream is invariant on TP and every module converts it `I -> R` on entry, so the residual projections, `output_res_proj` and `routed_up` see one gradient on every rank and are `I`; under SP (the TP/SP PR) they are `R` and FSDP sums the per-rank partials.
- `routed_down`, `wq_a`, `wkv_a` and `forget_a` are `R` either way: their consumers are TP-sharded, each rank's gradient is a partial sum.
- MLA and KDA convert the stream `I -> R` on entry (colwise projections, so the backward all-reduces the input gradient) and their rowwise exits hand it back invariant. The empty residual stack is cut from the stream, so it carries the stream's layout.
- `routed_norm` reduces the experts' Partial output at its boundary (keyed `x`, the argument `nn.RMSNorm.forward` takes); `routed_up` re-enters Partial so core's MoE exit reduces once.
- The MoonViT tower is invariant at TP and rank-local over cp (main's `include_cp_axis=True` helpers and its own attention entry, since the shared plan shards the tower over TP and all-gathers k/v over cp). Its exit follows the stream: `I` without SP, `R` under SP, where the exit's backward all-reduce sums the shards' feature gradients.
- Local regions: the MLA head unflatten and the rope join run in `spmd.local()` regions re-typed head-sharded on TP; KDA's head splits go through core's `local_head_split`, a plain view cannot split a feature-sharded dim.
- The vision prep runs in main's `multimodal_context()` and the stream is re-asserted on the decoder's layout after it, as qwen3_5 and kimi_k2_7 do.
- Every declaration is keyed by main's `forward()` parameter names (`q_THK`, `k_THK`, `v_THV`; `raw_gate_THK`, `raw_beta_TH`, `cu_seqlens` on `InnerKDA`): a `local_map` needs an entry per positional parameter, and a key matching nothing declares nothing.
- `parallelize.py` is untouched: 4446's replicated annotation seeds the parameters this PR does not declare, and its FSDP mesh resolution is the one every model shares.

### Results

`kimi_k3_debugmodel` (multimodal), `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256, on an RTX 5060 Ti with Attention Gym at upstream/main `b19162e` and its SM100/SM103 guard in `kda.py` lifted locally (KDA on Attention Gym's portable kernels). Each row is one cell under both backends: the same degree reads the same samples (the loader shards documents by dp rank, `components/data/sources.py`, so the dp1 and dp2 rows do not, and their step-1 losses differ with identical weights); the six cells share one compile cache, each warmed once and then run for 10 steps.

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
  --parallelism.spmd_backend spmd_types
```

| cell | world | partial_dtensor (step 1 / 3 / 10) | spmd_types (step 1 / 3 / 10) |
|---|---|---|---|
| dp1 | 1 | 12.52977 / 7.27107 / 2.98077 | 12.52977 / 7.27107 / 2.98077 |
| dp2 | 2 | 12.53137 / 7.31248 / 3.15823 | 12.53137 / 7.31248 / 3.15823 |
| dp2 x ep2 | 2 | 12.53146 / 7.20212 / 3.10296 | 12.53146 / 7.20212 / 3.10296 |

Step-1 gradients of the same cell under the two backends, every parameter (rank 0, own dtype, before clipping): dp2 750/750 bitwise, dp2 x ep2 750/750 bitwise, zero sign flips over 1.12e9 elements each.

Noise floor of this flavor (bf16 end to end, lr 8e-4, 2-step warm-up, so Adam's first update is lr times sign(g)): the same dp1 cell on another compile cache reads 12.52977 / 7.36833 / 2.91045 with bitwise step-1 gradients; EP on with the same samples (the dp2 and dp2 x ep2 rows) flips the sign of 1.4% of the gradient elements at step 1. Any two runs that round differently separate by a few percent by step 10; the pairs above do not.

4446's CI cell (`kimi_k3_debugmodel_mm_fsdp2`: `_use_spmd_types(typechecking=True)`, which turns activation checkpointing off since the checker rejects selective AC with FlexAttention), the cell this PR unbreaks, on its own compile cache; its dp1 is bitwise the AC-on dp1 of that cache:

| cell | world | spmd_types, type checking on, AC off | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | 1 | AC on, the cache's reference | 12.52977 | 7.36833 | 2.91045 |
| dp1 | 1 | | 12.52977 | 7.36833 | 2.91045 |
| dp2 | 2 | | 12.53137 | 7.22561 | 3.20438 |
| dp2 x ep2 | 2 | | 12.53146 | 7.15088 | 3.14271 |

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py           +332/-0   the declarations: weight types, stream conversions, module boundaries, the tower's plan, the local-map keys; the TP/SP seams, issued at tp = 1
      model.py              +164/-26  the declared modules, the local regions, the multimodal input layout and region, the stream's stack
      kda.py                +13/-7    cu_seqlens keyword-only on InnerKDA.forward; head splits through core's local_head_split

### CI/CD Coverage

- 4446's B200 cell `kimi_k3_debugmodel_mm_fsdp2` (this flavor under `spmd_types` with type checking, dp2) is this path; no new cell. As merged it raises at step 1 on the flavor's inputs (reproduced on `390e2985b` with the cell's own recipe); the dp2 row with type checking above is that cell.
- The TP entries are inert here (tp = 1); the TP/SP PR exercises them: at tp2, with and without SP, float32 per-parameter gradient dumps against dp1 show every replicated parameter bitwise identical across the TP ranks and within noise of dp1.

--- PASTE END ---

Notes for us, not for the body: this is the base of `tpsp_review3` (TP/SP, three commits on top) and the commit the CP PR carries; when it lands, `k3_cp_text` rebases over it and drops its own copy.
