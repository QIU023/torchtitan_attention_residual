# PR title: [Kimi K3] Tensor parallelism with sequence parallel

Body for PR 4499. Review branch `tp_sp_on_main` = `22eeec412` (2026-09-12): four commits on upstream main `7e7f271e0` (#4419, which removed the `spmd_backend` option and the partial_dtensor backend) -- `94c6f7949` (K2.5 tables), `b9c2135ee` (the MoonViT tensor-parallel plan public, the projector norm named by the caller), `91f76f986` (the declarations), `22eeec412` (the b200 cell). The PR branch `k3_tp_sp` is still `bd55160a8` and is not synced.

**Needs a pass before pasting:** Summary, Implementation and Stack below were written for the five commits on `da2f82670` (e.g. "the common `set_vision_transformer_block_sharding_config`"; the new stack makes K2.5's plan public instead). Results, Tests and Limitations are updated for `22eeec412`.

--- PASTE BEGIN ---

## Stack

- The Kimi K2.5 vision-table retype under type checking (`kimi_k2_7/vision_encoder.py`, the bottom commit): the shared position tables' `mutate_type` names a tp destination as well as the dp one, which the checker refuses at tp > 1 because the table is declared invariant on tp by `pos_embed`'s state layout; K2.5 itself fails this way at dp2 x tp2 with type checking on main. Can be its own PR.

## Summary

Enable tensor parallelism, with and without sequence parallel, for Kimi K3's hybrid KDA/MLA decoder and its multimodal input path, on the spmd_types backend.

- KDA and MLA are head-parallel: the projections that produce or consume the head axis are colwise / rowwise, the per-head KDA state (`A_log`, `dt_bias`, the depthwise convolutions) shards with the heads, the kernel runs on the local heads behind the `local_map` of #4527 re-declared head-sharded on tp, and the head splits go through core's `local_head_split`. The two rank-sized compressions (`wq_a` / `wkv_a`, `forget_a`) stay whole.
- Sequence parallel carries the tp-axis `Shard(0)` of the token stream between modules: norms on the shard, the attention boundaries gather, the rowwise outputs reduce-scatter (the llama3 template); the MoE internals take and return the shard.
- The vision tower is tensor-parallel through Kimi K2.5's MoonViT plan (the common `set_vision_transformer_block_sharding_config`, colwise / scaled-bias-rowwise linears, invariant norms and tables); K3 customises only its projector, whose norm follows the second linear.
- The multimodal splice follows Kimi K2.5 / Qwen3.5: with a tower present the token embedding stays replicated on TP, the scatter indexes the whole sequence on every rank, and layer 0's input boundary (the stream and the empty block-residual stack cut from it) restores the decoder's layout: the sequence shard under SP, invariant otherwise.
- Under EP the routed experts are whole on every tp rank, so `routed_down` follows the stream's rule (invariant without SP, replicated with it) instead of the replicated declaration the TP-sharded experts need.
- No core change: `distributed/utils.py`, `common/attention.py` and `common/linear.py` are main's.

## Implementation

`sharding.py`: `set_kimi_k3_sharding_config(config, enable_ep, enable_tp, enable_sp)` is the one entry point, called after `Decoder.Config.update_from_config`. Without TP it declares the vision buffers, the KDA kernel's DP-local boundary and the experts (#4527). With TP it adds core's decoder declarations, the head-parallel MLA / KDA declarations with the stream boundaries (invariant stream, replicated attention body), the dense FFN and latent-MoE projections around core's expert sharding, the tower plan, and, with a tower, the replicated-embedding / layer-0 boundary pattern. `model.py`: MLA's head splits use `local_head_split`; `_prepare_multimodal_embeds` is embed / encode / scatter. `parallelize.py` refuses tensor parallelism on partial_dtensor (the tower's gradients would sit on the fsdp mesh next to the decoder's on (fsdp, tp), which the gradient-norm stack refuses to mix).

## Limitations

- TP x CP is not exercised (CP is #4500's; the two are not combined).
- EP x TP (dp2 x ep2 x tp2) is not in the table; its type-checking smoke needs a four-GPU box. Larger degrees are not planned.
- The tower runs without sequence parallel (its patch sequence is short), as in Kimi K2.5.
- The tensor-parallel degree must divide the vision tower's head count, the check Kimi K2.5 carries. The debug tower has 6 heads, so the cells here stop at tp=2; the released tower has 12, where tp=2, tp=3, tp=4, tp=6 and tp=12 divide it and tp=8 does not.

## Tests

```text
pytest tests/unit_tests/cpu/test_integration_test_definitions.py
pytest tests/unit_tests/gpu/test_kimi_k3.py tests/unit_tests/gpu/test_kda_attention.py
```

Result on `22eeec412`: `test_integration_test_definitions.py`, `test_no_new_cli_options.py` and `test_deepseek_v4_hash_sharding.py` 19 passed. The GPU tests and pinned pyrefly have not been rerun on this head.

## Results

Measured on 2 x H100 80GB HBM3: this PR (`22eeec412`) against main (`7e7f271e0`), #4500's protocol -- dp1, `seed=42`, `--debug.deterministic`, bf16, 256 tokens per step, one seed checkpoint, one inductor cache per cell, 100 steps. Percentages are relative to tp=1 on main.

| cell | loss, step 1 | step 10 | step 100 |
| --- | ---: | ---: | ---: |
| tp=1, main | `12.624810` | `4.009670` | `2.942050` |
| tp=1, this PR | bitwise | bitwise | bitwise |
| tp=1, this PR, fresh inductor cache | bitwise | bitwise | bitwise |
| tp=2, SP on | `12.631220` (+0.051%) | `4.041600` (+0.796%) | `2.962110` (+0.682%) |
| tp=2, SP off | `12.628260` (+0.027%) | `4.111430` (+2.54%) | `2.836290` (-3.59%) |

| cell | grad norm, step 1 | step 10 | step 100 |
| --- | ---: | ---: | ---: |
| tp=1, main | `25.625` | `6.5` | `5.5625` |
| tp=1, this PR | bitwise | bitwise | bitwise |
| tp=1, this PR, fresh inductor cache | bitwise | bitwise | bitwise |
| tp=2, SP on | `25.875` (+0.976%) | `6.0312` (-7.21%) | `5.4688` (-1.68%) |
| tp=2, SP off | `26.0` (+1.46%) | `7.4062` (+13.9%) | `5.3125` (-4.49%) |

At tp=1 this PR is bitwise with main on all 100 steps, loss and grad norm, and a fresh inductor cache reproduces it. At tp=2 the sharded bf16 matmuls reduce in a different order: a few hundredths of a percent at step 1, single digits after. #4500's CP=2 rows on H100 read +3.8% and +15.1% loss at step 10.

A 3-step tp=2 run with `--debug.spmd_typechecking` (activation checkpointing off, as the b200 recipe sets it) passes.

```bash
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 100 \
  --training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.tensor_parallel_degree 2   # add --parallelism.no-enable-sequence-parallel for SP off
```

Box: 2 x H100 80GB HBM3 (capability 9.0), torch `2.15.0.dev20260906+cu130`, Attention Gym upstream main `499404b`; the KDA capability guard was widened locally to admit SM 9.0 and is not part of this PR.

## CI/CD

A new cell in the b200 suite: `kimi_k3_mm_tp2` (`torchtitan_recipes/tests/b200.py: kimi_k3_debugmodel_mm_tp2`, the multimodal debug model at tp=2 with sequence parallel on spmd_types with type checking, 2 GPUs), next to the existing `kimi_k3_mm_fsdp` cell.
