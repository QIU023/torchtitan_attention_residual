# PR title: [Kimi K3] Tensor parallelism with sequence parallel

Draft body for the rebuilt stack after Shuhua's review (review branch `tp_sp_on_main`; the PR branch `k3_tp_sp` is updated by the user). Four commits on upstream main `da2f82670`: `ddb306332` (K2.5 tables), `fc44fbb80` (the declarations), `bb2dad9da` (spmd_types required), `d4d6e774c` (the b200 cell). The Results section of the previous body (8 x A100, head `9a62f5229`) does not transfer: the tower is tensor-parallel now and the multimodal cells move; the table has to be re-measured on the new stack (cells listed below). TODO-NUMBERS marks what this box measured meanwhile.

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
- EP x TP is measured at dp2 x ep2 with tp=2 and tp=4 (eight GPUs); larger degrees are not.
- Tensor parallelism needs `--parallelism.spmd_backend spmd_types` (the default); partial_dtensor is refused with the reason above.
- The tower runs without sequence parallel (its patch sequence is short), as in Kimi K2.5.

## Tests

```text
pytest tests/unit_tests/cpu/test_integration_test_definitions.py
pytest tests/unit_tests/gpu/test_kimi_k3.py tests/unit_tests/gpu/test_kda_attention.py
```

Result: `test_integration_test_definitions.py` 13 passed; `gpu/test_kimi_k3.py` + `gpu/test_kda_attention.py` 3 passed, 2 skipped; the CPU suite's failure set is main's own (environment: transformers, torch_checkpointing). pre-commit passes on the touched files; pinned pyrefly (0.45.1) reports the same error set as main `da2f82670` (13 errors, none in the touched files). The Kimi K2.5 debug model at dp2 x tp2 with type checking fails on main at the shared vision tables (`mutate_type: expected current type R on axis mesh_tp, got I`) and gets past them with the bottom commit.

## Results

To re-measure on the new stack (the previous head's 8 x A100 table does not transfer: the tower is tensor-parallel now): `seed=42`, deterministic, one seed checkpoint per stream, bf16, 100 steps, tp=1 on main as the reference, percentages relative to it, loss and grad norm side by side; dp1 stream (256 tokens per step): tp=1 main / tp=1 this branch / tp=2 SP on / tp=2 SP off / tp=4 SP on / tp=4 SP off; dp2 stream (512 tokens per step, steps 1 and 10): dp2 / dp2 x tp2 / dp2 x ep2 / dp2 x ep2 x tp2.

Measured on the rebuild (8 GPUs, 3 steps, seed 42, deterministic, spmd_types, 4096 tokens per step, one seed checkpoint; loss at steps 1 / 3):

| cell | step 1 | step 3 |
| --- | ---: | ---: |
| tp=1 (bitwise with main) | `12.53584` | `6.63797` |
| tp=2 SP on | `12.56396` | `6.53527` |
| tp=2 SP off | `12.55333` | `6.81346` |
| dp2 x tp2 SP on | `12.53445` | `7.02513` |
| dp2 x ep2 x tp2 SP on | `12.52623` | `7.00102` |

With type checking on (the b200 cell's settings, the recipe's batch): tp=2 SP on `12.49262 / 11.38085 / 10.23243`, tp=2 SP off `12.48463 / 11.41571 / 9.89427`, dp2 x tp2 `12.56306 / 11.38314 / 9.64091`. At the 4096-token batch with a seed-42 init per cell: tp=1 `12.51269 / 9.93869 / 7.14302`, tp=2 SP on `12.45116 / 9.86473 / 7.14307`, tp=2 SP off `12.44463 / 9.90965 / 7.12468`. tp=4 needs a tower whose head count it divides (the debug tower has 6), so the tp=4 rows move to a text-only or larger-tower configuration.

## CI/CD

A new cell in the b200 suite: `kimi_k3_mm_tp2` (`torchtitan_recipes/tests/b200.py: kimi_k3_debugmodel_mm_tp2`, the multimodal debug model at tp=2 with sequence parallel on spmd_types with type checking, 2 GPUs), next to the existing `kimi_k3_mm_fsdp` cell.
