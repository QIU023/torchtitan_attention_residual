# PR title: [Kimi K3] Tensor parallelism with sequence parallel, stacked on the CP PR

For PR 4499, re-pointed at fork branch `tp_sp_on_4500` = `3fdd8c40d` (3 commits on `acisseJZhong:kda_cp` = `2884d82a9`; `git diff 2884d82a9` = 8 files, +676/-38). Paste between the markers. Numbers come from `phase13_k3like_48b_posttrain/TP_SP_ON_4500_2026-09-09.md`.

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

Result: `test_kimi_k3_sp_splice.py` 1 passed, `test_kda_attention.py` 5 passed 1 skipped, `test_kimi_k3.py` 6 passed 1 skipped (3 GPUs); pre-commit clean on the touched files except pyrefly hits that are #4500's / main's (`cp_kda.py`, the `ContextParallelRouting` import, `common/attention.py:832`).

## Results

Same protocol as #4500: `seed=42`, deterministic, 256 tokens per step, 100 steps, one seed checkpoint; tp=1 on the parent commit is the reference, percentages relative to it. The debug config trains in bf16 end to end.

| cell | step 1 loss (diff) | step 10 loss (diff) | step 100 loss (diff) | mean abs loss diff, steps 1-100 | step 1 grad norm (diff) | step 10 (diff) | step 100 (diff) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp=1, parent | `12.611600` | `4.951900` | `2.861500` | - | `25.75` | `9.25` | `5.7812` |
| tp=1, parent, spmd_types | bitwise | bitwise | bitwise | 0 | bitwise | bitwise | bitwise |
| tp=1, this branch, partial_dtensor | bitwise | bitwise | bitwise | 0 | bitwise | bitwise | bitwise |
| tp=1, this branch, spmd_types | bitwise | bitwise | bitwise | 0 | bitwise | bitwise | bitwise |
| tp=2 SP on, partial_dtensor | `12.622550` (`0.087%`) | `3.893010` (`21.4%`) | `2.907680` (`1.61%`) | `9.1%` | `25.0` (`2.9%`) | `6.8438` (`26%`) | `5.625` (`2.7%`) |
| tp=2 SP on, spmd_types | `12.622550` (`0.087%`) | `4.051770` (`18.2%`) | `2.921690` (`2.1%`) | `6.8%` | `25.0` (`2.9%`) | `7.7812` (`15.9%`) | `5.5938` (`3.2%`) |
| tp=2 SP off, partial_dtensor | `12.618100` (`0.052%`) | `4.238600` (`14.4%`) | `2.926060` (`2.26%`) | `4.8%` | `26.125` (`1.5%`) | `6.875` (`25.7%`) | `5.5` (`4.9%`) |

The four tp=1 rows are #4500's control: nothing changes at tp=1 on either backend, for 100 steps. The two SP-on rows read the same step-1 loss and grad norm on both backends. The step-10 spread is what bf16 masters do with a 0.05 to 0.1 percent step-1 difference on this flavor (the same cell twice is bitwise), and float32 masters do not remove it (9-layer alias: step 1 within 0.05 percent, mean over 100 steps 7 to 12 percent); correctness is the float32-compute comparison below. The spmd_types SP-off cell is withheld from this table until a 1e-3 gap in its step-1 gradients is located (its forward matches dp1 like the others).

Float32 masters and float32 compute (`--training.mixed_precision_param float32`, a float32 loop for the experts), 9-layer alias of the flavor, dp1 against tp2: the embedding output, the KDA gates and per-head parameters are bitwise, the colwise projections differ by 3e-7 (the fp32 matmul's tiling), the KDA recurrence turns that into 6.5e-5 at its output (the kernel itself is bitwise for 16 heads in one call against two calls of 8; a 3e-7 perturbation of its inputs moves its output by 1.8e-4), and the hidden state stays at 3e-6 to 3e-5 through the nine layers on both backends. Step-1 gradients in the same regime, 24 layers, 750 parameters, relative difference against dp1 with the sharded ones gathered: tp=2 SP on and SP off on partial_dtensor and SP on on spmd_types give median 1.1e-4 to 1.2e-4, p90 2.0e-3, max 1.6e-2 to 1.8e-2, with the KDA per-head parameters (`dt_bias`, `A_log`) as the tail; dp1 itself with the KDA projections perturbed by `1 + 3e-7 * N(0,1)` (the colwise matmul's roundoff) gives median 1.6e-4, p90 2.0e-3, max 1.5e-2 with the same class-by-class profile, so the TP gradients sit inside the floor the recurrence sets for that roundoff. Every TP cell holds the same gradient on both ranks for all 750 parameters.

--- PASTE END ---
