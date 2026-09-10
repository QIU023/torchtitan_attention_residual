# PR title: [Kimi K3] Tensor parallelism with sequence parallel, stacked on the multimodal spmd fix

For PR 4499 (keep it; close 4492 with the comment in PP_RUNTIME_ENGAGEMENT §7), re-pointed at fork branch `k3_tp_sp` = `tp_sp_on_main` = `e6bed4bc9` (10 commits on main `ac10ca48f`, the bottom one the Kimi K2.5 table retype filed as its own small PR, `PR_BODY_K27_TABLES.md`). The 10-step tables below were measured on this box on the 4527-based twin of the same delta before the rework; the rework changed no spmd_types number (verified bitwise, `TP_SP_REWORK_2026-09-10.md`) and removed the partial_dtensor TP rows, which the branch now refuses. The A100 100-step table (bf16, `run_bf16_100.sh`, 4527 base) is in `TP_SP_ON_4500_2026-09-09.md`; its spmd_types rows are pasted below as the 100-step section.

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

Result: `test_kimi_k3_sp_splice.py` 1 passed; `gpu/test_kda_attention.py` + `gpu/test_kimi_k3.py` 2 passed, 2 skipped (this box, 8 x RTX 5060 Ti). pre-commit passes on the touched files with the pinned pyrefly 0.45.1; `torchtitan/distributed/utils.py` is untouched. The Kimi K2.5 debug model at dp2 x tp2 with spmd_types type checking (AdamW, activation checkpointing off, 3 steps) fails on main at the shared vision tables with `mutate_type: expected current type R on axis mesh_tp, got I`, and with the bottom commit of this stack gets past them (it then hits a K2.5 tensor-parallel problem of its own in `qk_clip.py`, `QK clip scales do not match the MLA weight shape`, reported separately).

## Results

Same protocol as #4500: `seed=42`, deterministic, one seed checkpoint per stream, tp=1 on the parent commit as the reference, percentages relative to it, loss and grad norm side by side; 10 steps on this box (the 100-step run follows on A100; the debug config decays its learning rate from step 5 of a 10-step run, so these rows are a 10-step schedule, not the first ten steps of a 100-step one), 36 cells: dp1, dp2 and dp2 x ep2 streams x tp=1/2/4 x SP on/off x both SPMD backends, with the parent tp=1 cell of each stream. The debug config trains in bf16 end to end.

dp1 stream, 256 tokens per step:

| cell | loss 1 | grad norm 1 | loss 3 | grad norm 3 | loss 10 | grad norm 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp=1, parent, partial_dtensor | `12.600330` (bitwise) | `25.625` (bitwise) | `9.759030` (bitwise) | `19.875` (bitwise) | `4.309360` (bitwise) | `4.75` (bitwise) |
| tp=1, this branch, partial_dtensor | `12.600330` (bitwise) | `25.625` (bitwise) | `9.759030` (bitwise) | `19.875` (bitwise) | `4.309360` (bitwise) | `4.75` (bitwise) |
| tp=1, parent, spmd_types | `12.600330` (bitwise) | `25.625` (bitwise) | `9.759030` (bitwise) | `19.875` (bitwise) | `4.309360` (bitwise) | `4.75` (bitwise) |
| tp=1, this branch, spmd_types | `12.600330` (bitwise) | `25.625` (bitwise) | `9.759030` (bitwise) | `19.875` (bitwise) | `4.309360` (bitwise) | `4.75` (bitwise) |
| tp=2 SP on, spmd_types | `12.603320` (`0.0237%`) | `25.625` (bitwise) | `9.799900` (`0.419%`) | `21.125` (`6.29%`) | `4.155300` (`3.58%`) | `4.3438` (`8.55%`) |
| tp=2 SP off, spmd_types | `12.607430` (`0.0563%`) | `25.5` (`0.488%`) | `9.755270` (`0.0385%`) | `27` (`35.8%`) | `4.133930` (`4.07%`) | `4.4062` (`7.24%`) |
| tp=4 SP on, spmd_types | `12.641470` (`0.326%`) | `25` (`2.44%`) | `9.592080` (`1.71%`) | `21.25` (`6.92%`) | `4.094390` (`4.99%`) | `4.3125` (`9.21%`) |
| tp=4 SP off, spmd_types | `12.589200` (`0.0883%`) | `26.625` (`3.9%`) | `9.590550` (`1.73%`) | `21.625` (`8.81%`) | `4.570760` (`6.07%`) | `5.7188` (`20.4%`) |

dp2 stream, 512 tokens per step (a second dp rank reads other samples; compare within the stream):

| cell | loss 1 | grad norm 1 | loss 3 | grad norm 3 | loss 10 | grad norm 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp=1, parent, partial_dtensor | `12.427210` (bitwise) | `23.375` (bitwise) | `9.635420` (bitwise) | `20.875` (bitwise) | `4.026080` (bitwise) | `4.1875` (bitwise) |
| tp=1, this branch, partial_dtensor | `12.427210` (bitwise) | `23.375` (bitwise) | `9.635420` (bitwise) | `20.875` (bitwise) | `4.026080` (bitwise) | `4.1875` (bitwise) |
| tp=1, parent, spmd_types | `12.427210` (bitwise) | `23.375` (bitwise) | `9.635420` (bitwise) | `20.875` (bitwise) | `4.026080` (bitwise) | `4.1875` (bitwise) |
| tp=1, this branch, spmd_types | `12.427210` (bitwise) | `23.375` (bitwise) | `9.635420` (bitwise) | `20.875` (bitwise) | `4.026080` (bitwise) | `4.1875` (bitwise) |
| tp=2 SP on, spmd_types | `12.446900` (`0.158%`) | `23.75` (`1.6%`) | `9.513450` (`1.27%`) | `18.25` (`12.6%`) | `3.954370` (`1.78%`) | `3.6094` (`13.8%`) |
| tp=2 SP off, spmd_types | `12.480910` (`0.432%`) | `23.875` (`2.14%`) | `9.637730` (`0.024%`) | `18` (`13.8%`) | `4.043920` (`0.443%`) | `4.9062` (`17.2%`) |
| tp=4 SP on, spmd_types | `12.459320` (`0.258%`) | `24.25` (`3.74%`) | `9.743860` (`1.13%`) | `21.125` (`1.2%`) | `4.014260` (`0.294%`) | `4.25` (`1.49%`) |
| tp=4 SP off, spmd_types | `12.472210` (`0.362%`) | `23.5` (`0.535%`) | `9.736010` (`1.04%`) | `17.625` (`15.6%`) | `4.228030` (`5.02%`) | `4.4688` (`6.72%`) |

dp2 x ep2 stream, 512 tokens per step:

| cell | loss 1 | grad norm 1 | loss 3 | grad norm 3 | loss 10 | grad norm 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp=1, parent, partial_dtensor | `12.427330` (bitwise) | `23.375` (bitwise) | `9.659570` (bitwise) | `21.125` (bitwise) | `4.056090` (bitwise) | `4.125` (bitwise) |
| tp=1, this branch, partial_dtensor | `12.427330` (bitwise) | `23.375` (bitwise) | `9.659570` (bitwise) | `21.125` (bitwise) | `4.056090` (bitwise) | `4.125` (bitwise) |
| tp=1, parent, spmd_types | `12.427330` (bitwise) | `23.375` (bitwise) | `9.659570` (bitwise) | `21.125` (bitwise) | `4.056090` (bitwise) | `4.125` (bitwise) |
| tp=1, this branch, spmd_types | `12.427330` (bitwise) | `23.375` (bitwise) | `9.659570` (bitwise) | `21.125` (bitwise) | `4.056090` (bitwise) | `4.125` (bitwise) |
| tp=2 SP on, spmd_types | `12.467330` (`0.322%`) | `23.625` (`1.07%`) | `9.394440` (`2.74%`) | `19.125` (`9.47%`) | `3.871160` (`4.56%`) | `3.7812` (`8.33%`) |
| tp=2 SP off, spmd_types | `12.446770` (`0.156%`) | `23.625` (`1.07%`) | `9.412070` (`2.56%`) | `21.875` (`3.55%`) | `4.206590` (`3.71%`) | `5.7812` (`40.2%`) |
| tp=4 SP on, spmd_types | `12.413060` (`0.115%`) | `23.875` (`2.14%`) | `9.434100` (`2.33%`) | `21.5` (`1.78%`) | `4.271710` (`5.32%`) | `4.6562` (`12.9%`) |
| tp=4 SP off, spmd_types | `12.452520` (`0.203%`) | `23.625` (`1.07%`) | `9.967720` (`3.19%`) | `19.75` (`6.51%`) | `4.021300` (`0.858%`) | `4.1875` (`1.52%`) |

Reading the tables: the tp=1 rows are #4500's control, nothing changes at tp=1 (the two partial_dtensor tp=1 rows were measured before tensor parallelism was restricted to spmd_types; tp=1 has no tensor parallelism and still runs on either backend); with SP on the two backends read the same step-1 loss and grad norm, at tp=2 and at dp2 x tp2. The later steps spread by percents, and that spread is this model's on this box, not TP's: upstream's llama3 debugmodel under the same protocol reads tp=2 within `0.017%` of tp=1 at every one of the 10 steps (grad norm within `0.67%`), while for K3 the two backends of one TP configuration, bitwise at step 1, are `11%` apart by step 8; dp1 with a fixed `1e-3` perturbation on its KDA projections and no parallelism is `24%` off dp1 at step 7; and the 9-layer alias in float32 compute takes tp=2 from `3e-5` at step 1 to `10%` at step 10 (float32 masters with bf16 compute: `7.5%` at step 4). #4500's own CP=2 recipes on this box read `9.6%` at step 7 and `18.7%` at step 9 (all-gather) and `12%` at step 8 (Ulysses) against its dp1, where its H100 table reads `0.15%` at step 10. The same cell twice is bitwise in every regime. So on this hardware the 10-step column measures the flavor's sensitivity, and correctness is the step-1 comparison and the float32 gradient comparison below; the A100 run will show whether the sensitivity is Blackwell-specific.

Float32 masters and float32 compute (`--training.mixed_precision_param float32`, a float32 loop for the experts), full 24-layer model, step-1 gradients of the 750 parameters against dp1 with the sharded ones gathered, on this branch. The control row is dp1 itself with the KDA projections perturbed by `1 + 3e-7 * N(0, 1)`, the colwise matmul's fp32 roundoff, so it is the floor a correct TP sits inside:

| cell | median | p90 | max | within 1e-4 | within 1e-3 | within 1e-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 + 3e-7 on the KDA projections (control) | 2.1e-4 | 1.9e-3 | 1.6e-2 | 370 | 468 | 738 |
| tp=2 SP on, spmd_types | 1.1e-4 | 2.0e-3 | 1.8e-2 | 366 | 466 | 737 |
| tp=2 SP off, spmd_types (one routing flip, below) | 9.8e-3 | 1.4e-2 | 5.7e-2 | 0 | 4 | 402 |

100 steps, 8 x A100-SXM4-40GB (bf16, seed checkpoint, 256 tokens per step, tp=1 on the parent as the reference; the A100's gate kernel is attn-gym's eager reference below capability 9.0, its chunk kernels the portable Triton ones, as on H100):

| cell | step 1 loss (diff) | step 10 loss (diff) | step 100 loss (diff) | mean abs loss diff, steps 1-100 | step 1 grad norm (diff) | step 10 (diff) | step 100 (diff) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp=1, parent | `12.584470` | `3.900930` | `2.969010` | 0% | `25.875` | `9.125` | `5.3438` |
| tp=1, parent, spmd_types | bitwise | bitwise | bitwise | 0% | bitwise | bitwise | bitwise |
| tp=1, this branch, spmd_types | bitwise | bitwise | bitwise | 0% | bitwise | bitwise | bitwise |
| tp=2 SP on, spmd_types | `12.593710` (`0.073%`) | `4.085840` (`4.74%`) | `3.011340` (`1.43%`) | 6.83% | `25.375` (`1.9%`) | `5.6875` (`37.7%`) | `6.0625` (`13.4%`) |
| tp=2 SP off, spmd_types | `12.626320` (`0.333%`) | `4.111170` (`5.39%`) | `2.924570` (`1.50%`) | 3.90% | `25.625` (`1.0%`) | `7.2188` (`20.9%`) | `5.7812` (`8.2%`) |
| tp=4 SP on, spmd_types | `12.608530` (`0.191%`) | `4.072260` (`4.39%`) | `2.736770` (`7.82%`) | 9.98% | `25.875` (`0%`) | `4.9375` (`45.9%`) | `5.8125` (`8.8%`) |
| tp=4 SP off, spmd_types | `12.593140` (`0.069%`) | `4.223250` (`8.26%`) | `2.774510` (`6.55%`) | 4.94% | `25.625` (`1.0%`) | `12.9375` (`41.8%`) | `5.5` (`2.9%`) |

The later-step percentages are the class this model reads on every card measured with the debug recipe (256 tokens per step, lr 8e-4): #4500's own CP cells read +8.6% / +9.7% at step 10 on this A100 (`TP_SP_ON_4500_2026-09-09.md`), against the 0.15% its H100 table reports; llama3 under the same tensor-parallel code reads 0.017%. The dp2 stream at 100 steps is running (the kit's first pass passed a 256-token train step to two ranks).

Every TP cell holds the same gradient on both ranks for all 750 parameters. The tp=2 SP-off spmd_types row is one token's top-4 routing: at layer 17, token 6's fourth and fifth routing scores in dp1 are `0.6296397` and `0.6296365` (`3.3e-6` apart), the router's input differs from dp1 by `1.2e-5` relative under either backend, and spmd_types' rounding flips the order (expert 12 for 17) where partial_dtensor's keeps it. Layers 0 to 16 match dp1 at `5.4e-5` on that micro-batch and the other micro-batch matches through the last layer; from the flip on, that token routes differently in every later layer, which is the whole of the cell's step-1 loss (`5.5e-5`) and gradient difference. Three of the 5888 (router, token) pairs in the micro-batch sit within `1e-5`, and the control row moves the router scores by the same amount (`7e-6` relative) without flipping any: the discrete floor of top-k routing, not the sharding.

--- PASTE END ---

## CI/CD

A new cell in the b200 suite: `kimi_k3_mm_tp2` (`torchtitan_recipes/tests/b200.py: kimi_k3_debugmodel_mm_tp2`, the multimodal debug model at tp=2 with sequence parallel on spmd_types with type checking, 2 GPUs), next to the existing `kimi_k3_mm_fsdp` cell.
