# PR title: [Kimi K3] Tensor parallelism with sequence parallel

Body for PR 4499. The PR branch `k3_tp_sp` = `tp_sp_on_main` = `bd55160a8` (pushed 2026-09-11 with lease on `9a62f5229`), five commits on upstream main `da2f82670` -- the fifth, `bd55160a8`, drops the tensor-parallel disjunct the dispatcher condition cannot reach. The first four: `ddb306332` (K2.5 tables), `fc44fbb80` (the declarations), `bb2dad9da` (spmd_types required), `d4d6e774c` (the b200 cell). The Results section of the previous body (8 x A100, head `9a62f5229`) does not transfer: the tower is tensor-parallel now and the multimodal cells move; the table has to be re-measured on the new stack (cells listed below). TODO-NUMBERS marks what this box measured meanwhile.

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
- EP x TP (dp2 x ep2 x tp2) and the other four-GPU cells are part of the pending 8 x A100 rerun; larger degrees are not planned.
- Tensor parallelism needs `--parallelism.spmd_backend spmd_types` (the default); partial_dtensor is refused with the reason above.
- The tower runs without sequence parallel (its patch sequence is short), as in Kimi K2.5.
- The tensor-parallel degree must divide the vision tower's head count, the check Kimi K2.5 carries. The debug tower has 6 heads, so the cells here stop at tp=2; the released tower has 12, where tp=2, tp=3, tp=4, tp=6 and tp=12 divide it and tp=8 does not.

## Tests

```text
pytest tests/unit_tests/cpu/test_integration_test_definitions.py
pytest tests/unit_tests/gpu/test_kimi_k3.py tests/unit_tests/gpu/test_kda_attention.py
```

Result: `test_integration_test_definitions.py` 13 passed; `gpu/test_kimi_k3.py` + `gpu/test_kda_attention.py` 3 passed, 2 skipped; the CPU suite's failure set is main's own (environment: transformers, torch_checkpointing). pre-commit passes on the touched files; pinned pyrefly (0.45.1) reports the same error set as main `da2f82670` (13 errors, none in the touched files). The Kimi K2.5 debug model at dp2 x tp2 with type checking fails on main at the shared vision tables (`mutate_type: expected current type R on axis mesh_tp, got I`) and gets past them with the bottom commit.

## Results

**8 x A100: pending rerun after revision.** The previous head's 8 x A100 table does not transfer -- the vision tower is tensor-parallel on this stack, where it was replicated before -- and will be re-measured here in the same format.

Measured on the revised stack, 2 x H100 PCIe: this PR (`bd55160a8`) against main (`da2f82670`), same protocol as #4500 -- `seed=42`, `--debug.deterministic`, bf16, one seed checkpoint per batch shape, one inductor cache per cell, 100 steps. Loss, then total gradient norm; percentages are relative to the reference row.

dp1 stream, 256 tokens per step (reference: tp=1 on main):

| cell | step 1 | step 10 | step 20 | step 1 grad norm | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp=1, main | `12.624810` | `4.351560` | `3.631110` | `25.625000` | `4.843800` | `9.875000` |
| tp=1, this PR, spmd_types | bitwise | bitwise | bitwise | bitwise | bitwise | bitwise |
| tp=1, this PR, spmd_types, fresh cache | bitwise | bitwise | bitwise | bitwise | bitwise | bitwise |
| tp=1, this PR, partial_dtensor | bitwise | bitwise | bitwise | bitwise | bitwise | bitwise |
| tp=2, SP on | `12.631220` (+0.0508%) | `4.108150` (-5.59%) | `3.323500` (-8.47%) | `25.875000` (+0.976%) | `6.718800` (+38.7%) | `5.875000` (-40.5%) |
| tp=2, SP off | `12.628260` (+0.0273%) | `4.057510` (-6.76%) | `3.379290` (-6.94%) | `26.000000` (+1.46%) | `7.093800` (+46.5%) | `6.062500` (-38.6%) |

dp2 stream, 512 tokens per step (reference: dp2 on this PR, since a second data-parallel rank reads other samples):

| cell | step 1 | step 10 | step 20 | step 1 grad norm | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 | `12.463530` | `4.039750` | `3.340510` | `23.750000` | `6.218800` | `4.875000` |
| dp2 x ep2 | bitwise | `3.609350` (-10.7%) | `3.340420` (-0.00269%) | `23.625000` (-0.526%) | `4.593800` (-26.1%) | `3.734400` (-23.4%) |

At tp=1 this PR changes nothing on either backend: every tp=1 row is bitwise with main for 100 steps, and a fresh inductor cache reproduces it. One partial_dtensor tp=1 run on a cold cache diverged from step 2 and did not reproduce; the row above is its rerun on the same seed checkpoint, and the parent's rerun is bitwise too. The tp=2 rows move at step 1 by a few hundredths of a percent -- the sharded bf16 matmuls reduce in a different order -- and by percents at steps 10 and 20, the class the debug model reads for any change in summation order.

Box: 2 x H100 PCIe (capability 9.0), torch `2.15.0.dev20260906+cu130`, Attention Gym upstream main `b16d6d3`; the KDA capability guard was widened locally to admit SM 9.0 and is not part of this PR. The four-GPU cells (dp2 x tp2, dp2 x ep2 x tp2) need the pending A100 rerun.

## CI/CD

A new cell in the b200 suite: `kimi_k3_mm_tp2` (`torchtitan_recipes/tests/b200.py: kimi_k3_debugmodel_mm_tp2`, the multimodal debug model at tp=2 with sequence parallel on spmd_types with type checking, 2 GPUs), next to the existing `kimi_k3_mm_fsdp` cell.
