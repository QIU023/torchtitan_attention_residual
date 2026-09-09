# PR title: [Kimi K3] Tensor parallelism with sequence parallel, stacked on the CP PR

For PR 4499, re-pointed at fork branch `tp_sp_on_4500` (3 commits on `acisseJZhong:kda_cp` = `2884d82a9`). Paste between the markers. Numbers come from `phase13_k3like_48b_posttrain/TP_SP_ON_4500_2026-09-09.md`.

--- PASTE BEGIN ---

## PR stack

- #4500 (CP for Kimi K3; this PR's base)
- #4450, #4449, #4322

Stacked on #4500: the CP-side declarations that #4492 carried are #4500's now, so #4492 is closed and this PR holds the TP/SP delta only (the three commits after `2884d82a9`).

## Summary

Enable tensor parallelism, with and without sequence parallel, for Kimi K3's hybrid KDA/MLA decoder and its multimodal input path, on both SPMD backends.

- KDA and MLA are head-parallel: the projections that produce or consume the head axis are colwise / rowwise, the per-head KDA state (`A_log`, `dt_bias`, the depthwise convolutions) shards with the heads, and the kernel runs on the local heads behind the `local_map` that #4500 installs for CP, re-declared head-sharded on tp with #4500's cp gradient placements unchanged. The two rank-sized compressions (`wq_a` / `wkv_a`, `forget_a`) stay whole.
- Sequence parallel carries the tp-axis `Shard(0)` of the token stream between modules: norms on the shard, the attention boundaries gather, the rowwise outputs reduce-scatter (the llama3 template); the MoE internals take and return the shard (`set_moe_sharding_config(enable_sp=True)`).
- The multimodal splice under SP indexes global token positions, so it gathers the shard for the scatter and re-shards after; under spmd_types the splice learns the tp group from `parallelize` (`_sp_group`).
- `clip_grad_norm_` groups parameters by mesh: undeclared modules under TP hold gradients on the fsdp-only mesh.

## Implementation

`sharding.py` keeps #4500's `set_kimi_k3_sharding_config` (CP local maps, the tower over CP, the MoE) and gains `enable_sp` and `declare_vision_encoder`; `set_tensor_parallel_sharding_config` adds the head-parallel and stream declarations. `update_from_config` issues #4500's declarations under spmd_types or tp > 1 (the MoE declarations serve TP on partial_dtensor too, the tower's only under spmd_types) and the TP declarations at tp > 1. The head splits read the local head count from the projection width (`local_head_split` in KDA, the MLA reshapes), so the model code has no tp-degree arithmetic.

## Limitations

- TP x CP is not exercised: the CP path's vision-bank gather returns before the SP splice, and the two are not combined.
- EP x TP waits for #4500's rebase past the K3 EP merge (`9b5f60c40`); #4500's base lists EP as unsupported.
- The tower under spmd_types at tp > 1 takes #4500's TP-sharded projector declarations; under partial_dtensor it stays undeclared and runs whole on every rank.

## Tests

```text
pytest tests/unit_tests/cpu/test_kimi_k3_sp_splice.py tests/unit_tests/test_kda_attention.py tests/unit_tests/gpu/test_kimi_k3.py
```

Result: TESTS_PLACEHOLDER

## Results

Same protocol as #4500: `seed=42`, deterministic, 256 tokens per step, 100 steps, one seed checkpoint; tp=1 on the parent commit is the reference, percentages relative to it. The debug config trains in bf16 end to end.

TABLE_PLACEHOLDER

Float32 masters and float32 compute (`--training.mixed_precision_param float32`, a float32 loop for the experts), 9-layer alias of the flavor, dp1 against tp2: the embedding output, the KDA gates and per-head parameters are bitwise, the colwise projections differ by 3e-7 (the fp32 matmul's tiling), the KDA recurrence turns that into 6.5e-5 at its output (the kernel itself is bitwise for 16 heads in one call against two calls of 8; a 3e-7 perturbation of its inputs moves its output by 1.8e-4), and the hidden state stays at 3e-6 to 3e-5 through the nine layers on both backends. Step-1 gradients in the same regime, 24 layers: GRADIENT_PLACEHOLDER

--- PASTE END ---
