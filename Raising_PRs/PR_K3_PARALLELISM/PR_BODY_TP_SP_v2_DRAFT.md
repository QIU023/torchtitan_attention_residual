# PR title: [Kimi K3] Tensor parallelism with sequence parallel

Body for PR 4499 (checked against the code 2026-09-12). Review branch `tp_sp_on_main` = `22eeec412`: four commits on upstream main `7e7f271e0` -- `94c6f7949` (K2.5 tables), `b9c2135ee` (MoonViT plan public), `91f76f986` (the declarations), `22eeec412` (the b200 cell). The PR branch `k3_tp_sp` was synced to `22eeec412` on 2026-09-12 (GitHub: 4 commits, 9 files, +336/-117, no longer `dirty`).

Notes, not for pasting: the Results columns are steps 1 / 10 / 20. The runs are 100 steps, but both references start memorising the debug set from about step 40 (dp1 loss 0.84 at step 60 and 0.59 at 90; dp2 1.27 at 60 and 0.82 at 90), so no later step is reported (CLAUDE.md numerics-table rule); the 100 steps back only the "identical" claims. Not rerun on `22eeec412`: `tests/unit_tests/gpu/test_kimi_k3.py`, `test_kda_attention.py`. Pinned pyrefly (0.45.1) on the touched files: 0 errors (CPU box, 2026-09-11). "identical" in the tables is the logged values (loss to 5 decimals, grad norm to 4), not a bit dump. The PR is `dirty` against main until the sync; `22eeec412` merges cleanly onto today's main `56a721b64`. Logs: `phase13_k3like_48b_posttrain/tp_h100x4_logs_2026-09-12/`.

--- PASTE BEGIN ---

### Summary

Tensor parallelism, with and without sequence parallel, for Kimi K3 on spmd_types: KDA and MLA become head-parallel, the latent MoE takes core's expert sharding, and the MoonViT tower reuses Kimi K2.5's plan. At tp=1 nothing changes: loss and grad norm match main on all 100 steps.

- Before: `parallelize_kimi_k3` refuses tensor parallel, and MLA / KDA reshape to the global head count (`view(num_tokens, n_heads, ...)`). After: `local_head_split` splits the rank-local heads; `wq_b`, `wkv_b`, `gate` and KDA's `q_proj`, `k_proj`, `v_proj`, `forget_b`, `beta`, `output_gate` are colwise; `wo` and `output_proj` are rowwise, reduce-scattering under SP.
- Two Kimi K2.5 commits sit under it. `kimi_k2_7/vision_encoder.py`: the packed position tables retype only the dp axis under type checking; retyping tp as well is refused at tp > 1, where the tables are already invariant on tp. `kimi_k2_7/sharding.py`: K2.5's MoonViT plan is public as `set_moonvit_sharding_config`, with the projector norm named by the caller (`pre_norm` in K2.5, `post_norm` in K3). The first can go as its own PR.
- No core file changes.

### Design

- `kimi_k3/sharding.py`: `set_kimi_k3_sharding_config(config, *, enable_sp, enable_ep)` fills every sub-config unconditionally, as DeepSeek V3 does; `Module.parallelize` drops the disabled axes.
  - MLA: DeepSeek V3's plan without RoPE; `wq_a`, `wkv_a` and their norms replicated; inner attention through core's `set_gqa_inner_attention_local_spmd`; the headless rope key is broadcast to the local heads under `spmd.local()`.
  - KDA: Qwen3.5 GatedDeltaNet's head-sharded plan; `forget_a` and `output_norm` replicated; the conv weights, `A_log` and `dt_bias` shard with the heads; the kernel is one `local_spmd` region on the local heads.
  - Latent MoE: core's `set_moe_sharding_config`; `routed_down` replicated when the experts are TP-sharded and on the stream's rule under EP, where the experts are whole on every tp rank; without SP the experts' partial output is reduced at `routed_norm` and `routed_up` re-enters partial, so the MoE exit reduces once.
  - Multimodal: `tok_embeddings` stays TP-replicated for the vision scatter; layer 0's input boundary puts the stream and the block-residual stack back in the decoder's layout (sequence shard under SP, invariant otherwise).
- `kimi_k3/model.py`: `update_from_config` runs core's first, then checks that tp divides the tower's head count (as K2.5 does), then declares; the batch uses core's `multimodal_input_sharding()`.
- Not covered: TP x CP (#4500's); tp above 2 on the debug model, whose tower has 6 heads (the released tower has 12).

### Results

```bash
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 100 \
  --training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  --parallelism.tensor_parallel_degree 2   # --parallelism.no-enable-sequence-parallel for SP off
```

dp1, 256 tokens per step, against tp=1 on main; "identical" means the same logged loss and grad norm on every one of the 100 steps.

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp=1, main | `12.624810` | `4.351560` | `3.631110` | `25.625` | `4.8438` | `9.875` |
| tp=1, this PR | identical | identical | identical | identical | identical | identical |
| tp=1, this PR, fresh inductor cache | identical | identical | identical | identical | identical | identical |
| tp=2, SP on | `12.631220` (+0.051%) | `4.108150` (-5.59%) | `3.323500` (-8.47%) | `25.875` (+0.98%) | `6.7188` (+38.7%) | `5.875` (-40.5%) |
| tp=2, SP off | `12.628260` (+0.027%) | `4.057510` (-6.76%) | `3.379290` (-6.94%) | `26.0` (+1.46%) | `7.0938` (+46.5%) | `6.0625` (-38.6%) |

dp2, **512 tokens per step** (256 per rank; a multimodal row does not fit in 128), against dp2 on this PR.

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp2 | `12.463530` | `4.039750` | `3.340510` | `23.75` | `6.2188` | `4.875` |
| dp2 x ep2 (no TP) | identical | `3.609350` (-10.7%) | `3.340420` (-0.003%) | `23.625` (-0.53%) | `4.5938` (-26.1%) | `3.7344` (-23.4%) |
| dp2 x tp2 | `12.459580` (-0.032%) | `3.797220` (-6.0%) | `3.396810` (+1.69%) | `23.75` (same to 4 decimals) | `5.7812` (-7.0%) | `5.4062` (+10.9%) |
| dp2 x ep2 x tp2 | `12.470960` (+0.060%) | `3.760900` (-6.9%) | `3.450880` (+3.30%) | `23.75` (same to 4 decimals) | `5.125` (-17.6%) | `5.5312` (+13.5%) |

Kimi K2.5, dp2, tp=1, 4096 tokens per step (its debug config), this PR against main; K2.5 refuses tp > 1 on main (DistMuon, #3353).

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main | `7.967150` | `4.952920` | `3.445640` | `8.4513` | `5.1867` | `2.9887` |
| this PR | identical | identical | identical | identical | identical | identical |

Type checking, 3 steps with `--debug.spmd_typechecking` and activation checkpointing off (as the b200 recipe sets it): tp=2 on 2 GPUs and dp2 x ep2 x tp2 on 4 GPUs pass.

4 x H100 PCIe, torch `2.15.0.dev20260906+cu130`, Attention Gym main `499404b`; KDA's capability guard widened locally to admit SM 9.0, not part of this PR. For scale, #4500's CP=2 rows on H100: +3.8% and +15.1% loss at step 10.

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py                                              +264/-68  the TP/SP declarations: MLA, KDA, latent MoE, the multimodal boundary
      model.py                                                 +34/-31   MLA head splits through local_head_split; update_from_config order and the tower head check; core's multimodal_input_sharding
      kda.py                                                   +9/-6     KDA head splits through local_head_split
      parallelize.py                                           +0/-1     tensor parallel off the unsupported list
    torchtitan/models/kimi_k2_7/
      sharding.py                                              +8/-5     set_moonvit_sharding_config public, projector norm named by the caller
      vision_encoder.py                                        +3/-6     the packed position tables retype the dp axis only
    tests/integration_tests/b200.py                            +6/-0     the kimi_k3_mm_tp2 cell
    tests/unit_tests/cpu/test_integration_test_definitions.py  +1/-0     registers it in the b200 suite
    torchtitan_recipes/tests/b200.py                           +11/-0    kimi_k3_debugmodel_mm_tp2

### CI/CD Coverage

`kimi_k3_mm_tp2` in the b200 suite (dp1 x tp2, SP on, type checking, 2 GPUs), next to `kimi_k3_mm_fsdp`. `test_integration_test_definitions.py` and `test_no_new_cli_options.py` pass on this head.
