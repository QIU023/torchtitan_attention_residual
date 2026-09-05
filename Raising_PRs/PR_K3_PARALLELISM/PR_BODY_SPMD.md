# PR title: [Kimi K3] The declarations spmd_types reads: the dense path, the tower over cp, the MoE seams, the multimodal inputs

Branch `spmd_decl_review1` on the fork (`dbc60701d`, one commit on upstream/main `390e2985b`, which carries the merged 4446 "[spmd_types] Enable Kimi K3 backend"). It is the declarations commit the CP PR (`k3_cp_text`) already carries as `8c8d9436f`, lifted onto main after 4446: the backend branch that commit added to `parallelize.py` is now 4446's, so nothing of it remains here. The TP/SP PR (`tpsp_review3`) and the CP PR stack on this one. CPU: 18 tests pass (the three varlen failures are main's, a missing fla import), pyrefly count equal to main's; the TP declarations were validated at tp2 on the TP/SP branch with float32 per-parameter gradient dumps against dp1 (every replicated parameter bitwise identical across the TP ranks). Paste between the markers.

--- PASTE BEGIN ---

### Summary

Declares, for the `spmd_types` backend 4446 turned on, everything in Kimi K3 that 4446 left to `annotate_replicated_parameters`: the dense path's weight types and stream conversions, the MoonViT tower over the cp axis, the latent MoE's seams, and the multimodal inputs. Before this change the model declares layouts for its MoE modules only, every other parameter is annotated replicated, and the multimodal debug flavor stops at its first step under the now-default backend: `ValueError: spmd_types backend requires an SPMD layout for every tensor input, but these have no entry in input_sharding: ['grid_thw', 'pixel_values']` (`decoder.py` `preprocess_inputs` to `spmd_types.py` `annotate_input_spmd_types`; the path of 4446's own `kimi_k3_debugmodel_mm_fsdp2` cell). After it the config tree carries the declarations in `kimi_k3/sharding.py` (declarations only, applied through the Module protocol, the qwen3_5 shape), issued at tp = 1, where `spmd_types` consumes every one of them, and `preprocess_inputs` gives `pixel_values` and `grid_thw` the DP-local, TP-invariant layout every VLM decoder shares. Tensor parallelism stays on the unsupported list here; the TP/SP PR turns it on, on these declarations. dp1 and dp2 are bit-identical to `partial_dtensor` through ten seeded steps.

### Design

- Weight types follow main's rule for the token stream between the TP modules. Without sequence parallel the stream is invariant on TP and every module converts it `I -> R` on entry, so the weights that read and feed it (the attention- and ffn-residual projections, `output_res_proj`, `routed_up`) see one gradient on every rank and are `I`, where `R` would sum the copies; under sequence parallel the stream is the sequence shard, their gradients are per-rank partials, and they are `R`, summed by FSDP (`norm_config`'s rule). `routed_down`, `wq_a`, `wkv_a` and `forget_a` are `R` either way: their consumers are TP-sharded and each rank's gradient is a partial sum.
- Stream conversions: the MLA and KDA modules convert the invariant stream `I -> R` on entry (their projections are colwise, so the backward all-reduces the input gradient) and their rowwise exits hand it back invariant; the empty residual stack is cut from the stream, so it carries the stream's layout; `q_norm` / `kv_norm` are replicated state only; `routed_norm` reduces the experts' Partial output at its boundary (keyed `x`, the argument `nn.RMSNorm.forward` takes); `routed_up` re-enters Partial so core's MoE exit reduces once; the MoE exit returns invariant.
- The MoonViT tower is declared invariant at TP and rank-local over cp, with the cp axis on every vision layout (main's `include_cp_axis=True` helpers, as muse_glimmer passes them) and its own attention entry, because the shared plan shards the tower over TP and all-gathers its k/v over cp. Its exit follows the stream the features splice into: invariant without sequence parallel, replicated under it, where the `I -> R` exit's backward all-reduce sums the shards' feature gradients into the one gradient the tower's invariant weights expect.
- Local regions: the MLA head unflatten and the rope join run in `spmd.local()` regions re-typed head-sharded on TP, with the head count derived from the projection width; KDA's head splits go through core's `local_head_split`, since a plain view cannot split a feature-sharded dim.
- The vision prep runs in main's `multimodal_context()` (dp local, as qwen3_5 and kimi_k2_7 do) and the stream is re-asserted on the decoder's layout after it; without the region the checker rejects the DP-local pixel tensors at dp > 1.
- Every declaration is keyed by the `forward()` parameter names main uses (`q_THK`, `k_THK`, `v_THV`; `raw_gate_THK`, `raw_beta_TH`, `cu_seqlens` on `InnerKDA`): a `local_map` requires an entry for every positional parameter, and a key that matches nothing declares nothing.
- Nothing changes in `parallelize.py`: 4446's replicated annotation seeds the parameters this PR does not declare, and its FSDP mesh resolution is the one every model shares.

### Results

`kimi_k3_debugmodel` (multimodal), `--debug.seed 42 --debug.deterministic`, one seed checkpoint, 8192 tokens per step in micro-batches of 256; every cell runs twice and the second run is read; on an RTX 5060 Ti with Attention Gym's SM100/SM103 guard lifted locally. The last three rows run the flavor the way 4446's B200 cell does (`spmd_types` with type checking, activation checkpointing off).

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 --metrics.log_freq 1 \
  --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2
```

Running locally, the rows follow:

| cell | world | backend | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | 1 | partial_dtensor | 12.52977 | 7.27107 | 2.98077 |
| dp1 | 1 | spmd_types | 12.52977 | 7.27107 | 2.98077 |
| dp2 | 2 | spmd_types | 12.53137 | 7.31248 | 3.15823 |
| dp2 x ep2 | 2 | spmd_types | 12.53146 | 7.20212 | 3.10296 |
| dp1 | 1 | spmd_types, type checking, AC off | | | |
| dp2 | 2 | spmd_types, type checking, AC off | | | |
| dp2 x ep2 | 2 | spmd_types, type checking, AC off | | | |

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py           +332/-0   the declarations: weight types, stream conversions, the module boundaries, the tower's plan, the local-map keys; the TP/SP seams, issued at tp = 1
      model.py              +164/-26  the declared modules, the local regions, the multimodal input layout and region, the stream's stack
      kda.py                +13/-7    cu_seqlens keyword-only on InnerKDA.forward; head splits through core's local_head_split

### CI/CD Coverage

4446's B200 cell `kimi_k3_debugmodel_mm_fsdp2` (the multimodal flavor under `spmd_types` with type checking, dp2) is this path; no new cell. As merged, that cell raises at the first step on this flavor's inputs (`ValueError: spmd_types backend requires an SPMD layout for every tensor input, but these have no entry in input_sharding: ['grid_thw', 'pixel_values']`, from `annotate_input_spmd_types` under `Decoder.preprocess_inputs`; reproduced on `390e2985b` with the cell's own recipe): the `preprocess_inputs` override here declares the two layouts, and the dp2 row with type checking above is that cell.

--- PASTE END ---

Notes for us, not for the body: this is the base of `tpsp_review3` (TP/SP, three commits on top) and the commit the CP PR carries; when it lands, `k3_cp_text` rebases over it and drops its own copy.
