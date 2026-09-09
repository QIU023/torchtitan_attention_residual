# PR title: [Kimi K3] Tensor parallelism with sequence parallel, stacked on the multimodal spmd fix

For PR 4499 (keep it; close 4492 with the comment in PP_RUNTIME_ENGAGEMENT §7), re-pointed at fork branch `tp_sp_on_4527` = `ce49ab922` (5 commits on `shuhuayu:k3` = `d1e3979c7`, PR 4527, on main `53326e559`). The same delta on 4500's head is `tp_sp_on_4500` = `3fdd8c40d`, kept for when CP lands. Numbers below: this branch on the 5060 Ti box, 10 steps (`mx3_tp4527_*`, 2026-09-09); the A100 run of `matrix_scripts/tp_a100/` gives the 100-step and tp4 rows. Paste between the markers. Numbers come from `phase13_k3like_48b_posttrain/TP_SP_ON_4500_2026-09-09.md`.

--- PASTE BEGIN ---

## PR stack

- #4527 (the multimodal spmd annotations and the KDA local map; this PR's base)

Stacked on #4527, which carries the K3 spmd declarations this PR builds on; the CP-side declarations #4492 made are #4500's now, so #4492 is closed and this PR holds the TP/SP delta only (the five commits after `d1e3979c7`; the same delta rebased onto #4500 is on the fork as `tp_sp_on_4500`). One commit accepts any CUDA capability of 8.0 or newer for the KDA kernels, which Attention Gym's default Triton path requires (drop it if the SM100 gate is deliberate); one makes `RouterGateLinear` run on the local shards when its weight or input is a DTensor, since `aten.mm.dtype` has no DTensor sharding strategy and the gate is replicated on tp.

## Summary

Enable tensor parallelism, with and without sequence parallel, for Kimi K3's hybrid KDA/MLA decoder and its multimodal input path, on both SPMD backends.

- KDA and MLA are head-parallel: the projections that produce or consume the head axis are colwise / rowwise, the per-head KDA state (`A_log`, `dt_bias`, the depthwise convolutions) shards with the heads, and the kernel runs on the local heads behind the `local_map` that #4527 installs, re-declared head-sharded on tp. The two rank-sized compressions (`wq_a` / `wkv_a`, `forget_a`) stay whole.
- Sequence parallel carries the tp-axis `Shard(0)` of the token stream between modules: norms on the shard, the attention boundaries gather, the rowwise outputs reduce-scatter (the llama3 template); the MoE internals take and return the shard (`set_moe_sharding_config(enable_sp=True)`).
- The multimodal splice under SP indexes global token positions, so it gathers the shard for the scatter and re-shards after; under spmd_types the splice learns the tp group from `parallelize` (`_sp_group`).
- `clip_grad_norm_` groups parameters by mesh, on the dense and the EP path: undeclared modules under TP hold gradients on the fsdp-only mesh while the declared ones sit on (fsdp, tp).

## Implementation

`sharding.py` keeps #4527's `set_kimi_k3_sharding_config` (the KDA local map, the vision buffer, the MoE with `enable_ep`) and passes it `enable_sp`; `set_tensor_parallel_sharding_config` adds the head-parallel and stream declarations, and under spmd_types declares the tower invariant on tp (it runs whole on every rank, as under partial_dtensor). `update_from_config` issues the TP declarations at tp > 1. The head splits read the local head count from the projection width (`local_head_split` in KDA, the MLA reshapes), so the model code has no tp-degree arithmetic.

## Limitations

- TP x CP is not exercised (CP is #4500's; the two are not combined).
- EP x TP waits for #4500's rebase past the K3 EP merge (`9b5f60c40`); #4500's base lists EP as unsupported.
- The tower runs whole on every rank under both backends (invariant on tp).

## Tests

```text
pytest tests/unit_tests/cpu/test_kimi_k3_sp_splice.py tests/unit_tests/test_kda_attention.py tests/unit_tests/gpu/test_kimi_k3.py
```

Result: `test_kimi_k3_sp_splice.py` 1 passed (this box); `gpu/test_kda_attention.py` + `gpu/test_kimi_k3.py` 2 passed, 2 skipped (4 GPUs). pre-commit clean on the touched files except one pyrefly hit that is main's (`common/attention.py:805`).

## Results

Same protocol as #4500: `seed=42`, deterministic, 256 tokens per step, one seed checkpoint, tp=1 on the parent commit as the reference, percentages relative to it; 10 steps here (the 100-step run follows on A100). The debug config trains in bf16 end to end.

| Step | tp=1 parent loss | tp=1 parent, spmd_types | tp=1 this branch | tp=1 this branch, spmd_types | tp=2 SP on (diff) | tp=2 SP on, spmd_types (diff) | tp=2 SP off (diff) | tp=2 SP off, spmd_types (diff) | tp=4 SP on (diff) | tp=4 SP off (diff) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `12.600330` | bitwise | bitwise | bitwise | `12.603320` (`0.024%`) | `12.603320` (`0.024%`) | `12.633610` (`0.26%`) | `12.607430` (`0.056%`) | `12.641470` (`0.33%`) | `12.636180` (`0.29%`) |
| 3 | `9.759030` | bitwise | bitwise | bitwise | `9.782700` (`0.24%`) | `9.799900` (`0.42%`) | `10.063880` (`3.1%`) | `9.755270` (`0.039%`) | `9.676980` (`0.84%`) | `10.043420` (`2.9%`) |
| 10 | `4.309360` | bitwise | bitwise | bitwise | `4.162170` (`3.4%`) | `4.155300` (`3.6%`) | `4.614790` (`7.1%`) | `4.133930` (`4.1%`) | `4.068740` (`5.6%`) | `4.127090` (`4.2%`) |

Grad norm, same runs: tp=1 rows bitwise (`25.625`, `19.875`, `4.75`); tp=2 SP on `25.625` (`0%`) / `20.875` (`5.0%`) / `4.5625` (`3.9%`) on partial_dtensor and `25.625` / `21.125` / `4.3438` on spmd_types; tp=2 SP off `25.375` (`0.98%`) / `23.625` / `6.5938`; tp=4 SP on `25.0` (`2.4%`) / `22.25` / `4.3438`.

The data-parallel stream (a second dp rank reads other samples, so these rows compare with dp2 only), 512 tokens per step:

| Step | dp2 | dp2 x tp2 (diff) | dp2 x tp2, spmd_types (diff) | dp2 x ep2 (diff) | dp2 x ep2 x tp2 (diff) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `12.427210` | `12.446900` (`0.16%`) | `12.446900` (`0.16%`) | `12.427330` (`0.001%`) | `12.467330` (`0.32%`) |
| 3 | `9.635420` | `9.419580` (`2.2%`) | `9.513450` (`1.3%`) | `9.659570` (`0.25%`) | `9.538430` (`1.0%`) |
| 10 | `4.026080` | `4.004540` (`0.54%`) | `3.954370` (`1.8%`) | `4.056090` (`0.75%`) | `3.849490` (`4.4%`) |

Reading the tables: the four tp=1 rows are #4500's control, nothing changes at tp=1 on either backend; with SP on the two backends read the same step-1 loss and grad norm, at tp=2 and at dp2 x tp2. The later steps move by percents on this flavor for any step-1 difference (the same cell twice is bitwise), so correctness is the float32-compute comparison below.

Float32 masters and float32 compute (`--training.mixed_precision_param float32`, a float32 loop for the experts), full 24-layer model, step-1 gradients of the 750 parameters against dp1 with the sharded ones gathered, on this branch. The control row is dp1 itself with the KDA projections perturbed by `1 + 3e-7 * N(0, 1)`, the colwise matmul's fp32 roundoff, so it is the floor a correct TP sits inside:

| cell | median | p90 | max | within 1e-4 | within 1e-3 | within 1e-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 + 3e-7 on the KDA projections (control) | 2.1e-4 | 1.9e-3 | 1.6e-2 | 370 | 468 | 738 |
| tp=2 SP on, partial_dtensor | 1.2e-4 | 2.0e-3 | 1.8e-2 | 366 | 466 | 737 |
| tp=2 SP off, partial_dtensor | 1.2e-4 | 2.0e-3 | 1.6e-2 | 364 | 466 | 736 |
| tp=2 SP on, spmd_types | 1.1e-4 | 2.0e-3 | 1.8e-2 | 366 | 466 | 737 |
| tp=2 SP off, spmd_types (one routing flip, below) | 9.8e-3 | 1.4e-2 | 5.7e-2 | 0 | 4 | 402 |

Every TP cell holds the same gradient on both ranks for all 750 parameters. The tp=2 SP-off spmd_types row is one token's top-4 routing: at layer 17, token 6's fourth and fifth routing scores in dp1 are `0.6296397` and `0.6296365` (`3.3e-6` apart), the router's input differs from dp1 by `1.2e-5` relative under either backend, and spmd_types' rounding flips the order (expert 12 for 17) where partial_dtensor's keeps it. Layers 0 to 16 match dp1 at `5.4e-5` on that micro-batch and the other micro-batch matches through the last layer; from the flip on, that token routes differently in every later layer, which is the whole of the cell's step-1 loss (`5.5e-5`) and gradient difference. Three of the 5888 (router, token) pairs in the micro-batch sit within `1e-5`, and the control row moves the router scores by the same amount (`7e-6` relative) without flipping any: the discrete floor of top-k routing, not the sharding.

--- PASTE END ---
