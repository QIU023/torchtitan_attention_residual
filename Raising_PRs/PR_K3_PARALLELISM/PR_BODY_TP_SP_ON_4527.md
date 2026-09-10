# PR title: [Kimi K3] Tensor parallelism with sequence parallel, stacked on the multimodal spmd fix

For PR 4499 (keep it; close 4492 with the comment in PP_RUNTIME_ENGAGEMENT §7), re-pointed at fork branch `k3_tp_sp` = `tp_sp_on_main` = `9a62f5229` (11 commits on main `ac10ca48f`, the bottom one the Kimi K2.5 table retype filed as its own small PR, `PR_BODY_K27_TABLES.md`). Every number in this body is from 8 x A100 (`matrix_scripts/tp_a100/`, `TP_SP_ON_4500_2026-09-09.md`); the rework's bitwise checks are in `TP_SP_REWORK_2026-09-10.md`.

--- PASTE BEGIN ---

## PR stack

- #4527 (the multimodal spmd annotations and the KDA local map; this PR's base)
- the Kimi K2.5 vision-table retype under type checking (`kimi_k2_7/vision_encoder.py`, one commit at the bottom of this branch, filed as its own small PR): the shared position tables named a tp destination as well as the dp one, which the checker refuses at tp > 1 once the tower's parameters carry a tp declaration; the same check fails for K2.5 itself at tp2 with type checking on

Stacked on #4527, which carries the K3 spmd declarations this PR builds on; the CP-side declarations #4492 made are #4500's now, so #4492 is closed and this PR holds the TP/SP delta only (the eight commits after `d1e3979c7`; the same delta rebased onto #4500 is on the fork as `tp_sp_on_4500`). One commit accepts any CUDA capability of 8.0 or newer for the KDA kernels, which Attention Gym's default Triton path requires (drop it if the SM100 gate is deliberate); one makes `RouterGateLinear` run on the local shards when its weight or input is a DTensor, since `aten.mm.dtype` has no DTensor sharding strategy and the gate is replicated on tp.

## Summary

Enable tensor parallelism, with and without sequence parallel, for Kimi K3's hybrid KDA/MLA decoder and its multimodal input path, on the spmd_types backend (the default; with partial_dtensor, tensor parallelism is refused, see Limitations).

- KDA and MLA are head-parallel: the projections that produce or consume the head axis are colwise / rowwise, the per-head KDA state (`A_log`, `dt_bias`, the depthwise convolutions) shards with the heads, and the kernel runs on the local heads behind the `local_map` that #4527 installs, re-declared head-sharded on tp. The two rank-sized compressions (`wq_a` / `wkv_a`, `forget_a`) stay whole.
- Sequence parallel carries the tp-axis `Shard(0)` of the token stream between modules: norms on the shard, the attention boundaries gather, the rowwise outputs reduce-scatter (the llama3 template); the MoE internals take and return the shard (`set_moe_sharding_config(enable_sp=True)`).
- The multimodal splice under SP indexes global token positions, so it gathers the shard for the scatter and re-shards after; under spmd_types the splice learns the tp group from `parallelize` (`_sp_group`).
- No core change to gradient clipping: under spmd_types the tower's parameters are declared on the tp axis with the rest of the model, so every gradient lives on one mesh and `clip_grad_norm_` is the single call it always was. Under partial_dtensor the tower is sharded by FSDP alone and its gradients would sit on the fsdp mesh next to the decoder's on (fsdp, tp), which the gradient-norm stack refuses to mix; rather than teach core to group by mesh, `parallelize_kimi_k3` refuses tensor parallelism on that backend.
- Under EP the routed experts are whole on every tp rank, so `routed_down` follows the stream's rule (invariant without SP, replicated with it) instead of the replicated declaration the TP-sharded experts need; the 36-cell matrix caught the replicated version by its step-1 grad norm (+37% at tp=2, +131% at tp=4, SP off, spmd_types).

## Implementation

`sharding.py` keeps #4527's `set_kimi_k3_sharding_config` (the KDA local map, the vision buffer, the MoE with `enable_ep`) and passes it `enable_sp`; `set_tensor_parallel_sharding_config` adds the head-parallel and stream declarations, and under spmd_types declares the tower invariant on tp (it runs whole on every rank, as under partial_dtensor). `update_from_config` issues the TP declarations at tp > 1. The head splits read the local head count from the projection width (`local_head_split` in KDA, the MLA reshapes), so the model code has no tp-degree arithmetic.

## Limitations

- TP x CP is not exercised (CP is #4500's; the two are not combined).
- EP x TP is measured at dp2 x ep2 with tp=2 and tp=4 (eight GPUs here); larger degrees are not.
- The tower runs whole on every rank (invariant on tp).
- Tensor parallelism needs `--parallelism.spmd_backend spmd_types` (the default): with partial_dtensor the tower's gradients live on the fsdp mesh alone and the gradient-norm stack refuses the mix with the decoder's (fsdp, tp) gradients; `parallelize_kimi_k3` raises with that reason. Placing the tower on the tp mesh under partial_dtensor would need its inputs and outputs wrapped as DTensors around a flex-attention forward, which is left for later.

## Tests

```text
pytest tests/unit_tests/cpu/test_kimi_k3_sp_splice.py tests/unit_tests/test_kda_attention.py tests/unit_tests/gpu/test_kimi_k3.py
```

Result: `test_kimi_k3_sp_splice.py` 1 passed; `gpu/test_kda_attention.py` + `gpu/test_kimi_k3.py` 2 passed, 2 skipped (8 x A100: 3 passed, 2 skipped, the same split). pre-commit passes on the touched files with the pinned pyrefly 0.45.1; `torchtitan/distributed/utils.py` is untouched. The Kimi K2.5 debug model at dp2 x tp2 with spmd_types type checking (AdamW, activation checkpointing off, 3 steps) fails on main at the shared vision tables with `mutate_type: expected current type R on axis mesh_tp, got I`, and with the bottom commit of this stack gets past them (it then hits a K2.5 tensor-parallel problem of its own in `qk_clip.py`, `QK clip scales do not match the MLA weight shape`, reported separately).

## Results

Same protocol as #4500: `seed=42`, deterministic, one seed checkpoint per stream, bf16, 100 steps, tp=1 on the parent commit (main `ac10ca48f`) as the reference, percentages relative to it, loss and grad norm side by side. Measured on 8 x A100-SXM4-40GB (the A100's gate kernel is attn-gym's eager reference below capability 9.0, its chunk kernels the portable Triton ones, as on H100). Tensor parallelism runs on spmd_types (partial_dtensor with TP is refused, see Limitations); the tp=1 rows on both backends are the control.

dp1 stream, 256 tokens per step (this head `9a62f5229` against main `ac10ca48f`):

| cell | step 1 loss (diff) | step 10 loss (diff) | step 100 loss (diff) | mean abs loss diff, steps 1-100 | step 1 grad norm (diff) | step 10 (diff) | step 100 (diff) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp=1, parent, partial_dtensor | `12.584470` | `3.900930` | `2.969010` | 0% | `25.875` | `9.125` | `5.3438` |
| tp=1, parent, spmd_types | bitwise | bitwise | bitwise | 0% | bitwise | bitwise | bitwise |
| tp=1, this branch, partial_dtensor | bitwise | bitwise | bitwise | 0% | bitwise | bitwise | bitwise |
| tp=1, this branch, spmd_types | bitwise | bitwise | bitwise | 0% | bitwise | bitwise | bitwise |
| tp=2 SP on, spmd_types | `12.593710` (`0.073%`) | `3.989400` (`2.27%`) | `2.954350` (`0.49%`) | 4.08% | `25.25` (`2.4%`) | `6.0625` (`33.6%`) | `5.5312` (`3.5%`) |
| tp=2 SP off, spmd_types | `12.626320` (`0.333%`) | `4.111170` (`5.39%`) | `2.924570` (`1.50%`) | 3.90% | `25.625` (`1.0%`) | `7.2188` (`20.9%`) | `5.7812` (`8.2%`) |
| tp=4 SP on, spmd_types | `12.608530` (`0.191%`) | `4.060410` (`4.09%`) | `2.952170` (`0.57%`) | 6.26% | `25.875` (`0%`) | `6.3438` (`30.5%`) | `5.6562` (`5.9%`) |
| tp=4 SP off, spmd_types | `12.593140` (`0.069%`) | `3.980570` (`2.04%`) | `2.915250` (`1.81%`) | 5.29% | `25.625` (`1.0%`) | `6.125` (`32.9%`) | `5.5312` (`3.5%`) |

dp2 stream, 512 tokens per step (dp2 on this branch as the reference, dp2 on the parent as the control): running, pasted when done.

The later-step percentages are the class this model reads on every card measured with the debug recipe (256 tokens per step, lr 8e-4): #4500's own CP cells read +8.6% / +9.7% at step 10 on this A100 (`TP_SP_ON_4500_2026-09-09.md`), against the 0.15% its H100 table reports; llama3 under the same tensor-parallel code reads 0.017%. The dp2 stream at 100 steps is running (the kit's first pass passed a 256-token train step to two ranks).


## CI/CD

A new cell in the b200 suite: `kimi_k3_mm_tp2` (`torchtitan_recipes/tests/b200.py: kimi_k3_debugmodel_mm_tp2`, the multimodal debug model at tp=2 with sequence parallel on spmd_types with type checking, 2 GPUs), next to the existing `kimi_k3_mm_fsdp` cell.
