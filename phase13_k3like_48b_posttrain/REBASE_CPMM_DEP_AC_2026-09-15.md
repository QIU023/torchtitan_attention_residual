# AttnRes AC reuse against the report paragraph; Dynamic CP onto PR 4639; DEP onto PR 4312 (2026-09-15)

## 1. AttnRes AC reuse: what the branch does against the report's paragraph

The paragraph (report sec 5.2.2): "The block representation is generated once at the boundary layer and shared by all subsequent layers, residing directly on the GPU. The AttnRes computation is entirely wrapped with checkpointing, so the activation saved for the backward pass at each layer is identical to that of the standard residual architecture. For pipeline parallelism, we adopt cache-based pipeline communication [57] ..."

| sentence | code today | verdict |
| --- | --- | --- |
| 1: block representation generated once at the boundary layer, shared by all subsequent layers, on the GPU | main, 4312 and `k3_ac_reuse_attention` re-`cat` a new `[T, k, D]` stack at every block boundary (`model.py`, `first_layer_in_block` / `opens_block`), and every block's stack stays saved (it is the input of the residual checkpoint, or of the block's AC), so K stacks are alive at once: sum of k*T*D, quadratic in blocks. Megatron 6840 keeps a list appended once per block start (`AttnResStageSources.graph_sources`, `append_block_start`), linear. | not implemented on the branch; incomplete, not wrong. Changing it means the stack becomes per-block entries, which touches 4312's stage assembly and CP's stack sharding: a design decision, not a fixup. |
| 2: AttnRes computation entirely wrapped with checkpointing, per-layer saved activation identical to a standard residual | branch commit `fd0a9b6c6`: `_checkpointed_attention_residual` under `remat.checkpoint`, cleared (`checkpoint_residual = False`) when selective/full AC already checkpoints the whole block, a `remat.region` under RegionAC. 6840 gets the same effect inside the op: `_AttnResAggregation` saves per-token stats only and recomputes the fp32 upcasts and normalized keys in backward (its docstring: "saves only references to the incoming values plus per-token fp32 statistics ... and recomputes everything else in backward"); its compile path lets AOTAutograd build the recomputing backward; its FLA path passes `checkpoint_level=1`. | matches. |
| 3: cache-based pipeline communication | PR 4312. | matches. |
| not in the paragraph | branch commit `615908fcd` (MLA/KDA `torch_remat` regions so RegionAC can keep attention and recompute the MoE) and the whole body of PR 4656. | this is where "AC reuse" went astray: it reads "reuse" as reusing attention activations under AC. 6840 has nothing of the kind and never calls anything AC reuse. |

PR 4656 (open, 0 comments, head `362b6cc70`, title "[DO NOT review, stack on AttnRes AC ReUse] Declare torch_remat regions so RegionAC can keep attention and recompute the MoE"): the base "AttnRes AC ReUse" PR was never opened, the branch carries all three commits, so 4656's diff shows the report item while its body describes only the regions. Options: (a) repurpose 4656: retitle to the report item, body from `PR_BODY_AC_REUSE.md` without the region paragraphs, push the branch without `615908fcd` (and its share of the test); (b) close 4656 and open the report-item PR fresh. Either way the branch is 18 behind main (`0ae22167c`, merge-tree clean) and the body's numbers are 5060 numbers, so a rebase, a line-by-line diff audit of `fd0a9b6c6` and H100 tables come before filing. Decision (user, 2026-09-15): option (a). Review branch `ac_review3` = `ea1316606`: main `0ae22167c` plus the recompute (`d1c829f06`, the old `fd0a9b6c6` cherry-picked clean) and its CPU test (`tests/unit_tests/cpu/test_kimi_k3_attention_residual_recompute.py`, the recompute half of the old `test_kimi_k3_remat_regions.py` with the RegionAC tests and the call counters gone; 3 passed). The regions commit `615908fcd` and its tests are out. Body draft: `Raising_PRs/PR_K3_PARALLELISM/PR_BODY_AC_REUSE.md` (title, paste section without the region material; the 5060 table stays until the H100 rerun on this head). To do on GitHub by the user: retitle 4656, paste the body, sync `k3_ac_reuse_attention` from `ac_review3`.
5060 smoke of `ea1316606` (`int0915_logs/run_ac4656.sh`, dp1, 2048 tokens per step in 512-token micro-batches, seed 42, deterministic, 3 steps, cold caches): AC none / selective / full all rc 0 and read the same three losses (`12.63048` / `10.80023` / `8.26401`, grad norm `20.6250` / `15.1250` / `11.2500`); step 1 and the peaks (14.17 / 12.68 / 12.52 GiB) equal the old body table's PR column, so the rebase moved nothing visible. ufmt clean; pyrefly 0 errors on the three files (main: 0).

## 2. Dynamic CP rebased onto PR 4639

- Source: PR 4380's branch `k3_cp_mm`, head `a063a3d0e` (one commit on the text CP PR 4313's head `61a73ca6c`, main `390e2985b` of 09-04). Target: PR 4639 head `e06dcbee3` (two commits on main `4a0d8dab3`, unchanged since 09-11; 4639 replaced 4500, closed 09-13).
- Worktree `/tmp/wt_cpmm_4639`, pushed to `cpmm_review1` = `774e0b9b5`: `7bfa51503` (the CP transform's `exclude_fqn_prefixes` and the two K3 mm CP recipes excluding `vision_encoder`, the integration fix `9c5876776`; without it 4639's own mm CP recipes stop in the tower's flex attention, see `K3_INT_20260915.md`) then `774e0b9b5` (dynamic CP, +828/-9 over 8 files). No trailers.
- Conflicts: three hunks, all positional (an import line; `get_attention_masks` next to the new encode block; the `skip_dp` return next to the sub-group build).
- Ports, each found by a GPU run that went a different way than it looked:
  - 4313's `_context_parallel_group()` read the cp group off the active SPMD mesh; on 4639 the tower encodes inside `spmd_local_context("dp")`, where `spmd_mesh_group("cp")` is None, so the first GPU run took the replicated path without a word (zero "Dynamic CP" log lines; allgather `12.56631`, ulysses `12.42911` at step 1). Now `parallelize_kimi_k3` stores the cp group on the model next to the sub-groups and `encode_images` reads it from there. `torchtitan.tools.logging` is gone on 4639's base: `logging.getLogger(__name__)`.
  - With the cp group in place the path still did not cut anything: the debug batch's image is 16 x 12 = 192 patches, under the 256 default (the 09-10 note's batch had one above it). The seeded cells below lower the bar to 96 through a scratch recipe (`int0915_logs/cpmm_probe.py`); the default stays 256.
  - Under SPMD type checking (the b200 recipes turn it on) the cut path raised twice: `_AllGather.apply` (the differentiable `torch.distributed.nn.functional.all_gather` behind gather-KV) and `_PlainGradBoundary.apply` are autograd Functions the checker does not know, now registered with `spmd.register_local_autograd_function` (the tower runs in the dp-local context, so the local rule applies), and the padded-key `BlockMask`'s mask_mod indexes the keep vector (`ModIndex`), now built under `spmd.no_typecheck()` the way Kimi K2.5's tower builds its block-diagonal mask. On 4313 these cells never ran with type checking on.
- CPU (venv_bfx9): 34 passed (`test_kimi_k3_vit_cp_plan` 7, the K3 tests, definitions, transforms); `test_no_new_cli_options` fails only on 4639's own `context_parallel_load_balancer.load_balancer_type`, nothing from this port.
- GPU: below.

## 3. DEP rebased onto PR 4312

- Source: PR 4381's branch `k3_pp_mm`, head `c87097ae5` = 19 commits of the 09-04 text PP branch (on main `6e2ac3dcd`) plus two DEP commits (`f15c88ff6`, `c87097ae5`). Target: PR 4312 head `de6f29514` (= `k3_pp_text` = `pp_review4`), 101 commits past that base.
- Rebasing the two DEP commits directly conflicts in eight hunks against 4312's rewritten split (`_kimi_k3_split`, `kimi_k3_module_fqns_per_model_part(num_stages, num_layers, ...)`), so the port takes the integration tree's DEP commits instead (`39a063732`, `874953b98`, `ca9ad1726`), which the 09-15 replay had already adapted to `de6f29514`'s layout. Their eight conflict hunks were integration context only, all dropped: TP's `padding_mask` / `kda_cp_routing` / `vision_bank_indices_T` / `cu_seqlens` parameters, `attn_res_cache_offload`, `pp_balance`, the two CP recipes and their b200 entries, `_apply_ac_outside_attention`. The integration's later import move (`fd5e515d0`) is not taken: 4312's tree still has `torchtitan.tools.logging`.
- Folded in: `_kimi_k3_vit_dep_split`'s return annotation (`Any` was never imported on 4312: `ParallelismConfig`), and `kimi_k3_pp8_vp4_vit_dep` in the b200 expected set of `test_integration_test_definitions.py`. Trailers stripped.
- Worktree `/tmp/wt_dep_4312`: `c58084968`, `503d77fc0`, `384d576dc` on `de6f29514`; +1641/-4 over 11 files.
- CPU: 53 passed (`test_kimi_k3_dep_bubble`, `test_kimi_k3_vit_dep_split`, `test_kimi_k3_pp_layout`, `test_kimi_k3_pp_stage`, `test_kimi_k3_stage_swap`, `test_kimi_k3_pp_exact_block_grads`, definitions, frozen CLI). `test_model_td_layout`'s three failures need `flash_attn` and fail the same on plain `de6f29514`.
- GPU: below.

## 4. GPU smokes (RTX 5060 Ti box, KDA guard lifted locally in both worktrees, 3 steps, cold caches: rc and steps only)

DEP on 4312 (`/tmp/wt_dep_4312` = `384d576dc`, pushed to `dep_review1`), 8 GPUs, the b200 recipes, unseeded (each rank initialises its own parts, so the two splits draw different weights and the losses are not a pair; the seeded pairing is in `DEP_NEW_TREE_2026-09-10.md`):

| cell | rc | loss / grad norm, step 1 | step 2 | step 3 |
| --- | --- | --- | --- | --- |
| pp8 x vp4, ViT DEP (`kimi_k3_debugmodel_pp8_vp4_vit_dep`) | 0 | 12.41330 / 15.9375 | 10.77271 / 14.6875 | 8.33396 / 10.6875 |
| pp8 x vp4 (`kimi_k3_debugmodel_pp8_vp4`) | 0 | 12.55091 / 17.7500 | 11.24661 / 14.3750 | 8.99473 / 10.6875 |

Dynamic CP on 4639 (`cpmm_review1` = `774e0b9b5`), 2 GPUs per cell, the b200 mm CP recipes, `--debug.seed 42 --debug.deterministic`, the tower either cut over the pair (threshold 96, "Dynamic CP: 1 large image(s) of 1 over 1 sub-CP group(s) of 2 rank(s)" in the log) or forced replicated (threshold 10^9). Same seed and data in all four, so step 1 pairs:

| cell | rc | loss / grad norm, step 1 | step 2 | step 3 |
| --- | --- | --- | --- | --- |
| allgather-KV cp2, tower cut | 0 | 12.27696 / 27.0000 | 11.10394 / 35.0000 | 10.31077 / 27.5000 |
| allgather-KV cp2, tower replicated | 0 | 12.27654 / 27.2500 | 10.89869 / 33.0000 | 10.03317 / 21.5000 |
| Ulysses cp2, tower cut | 0 | 12.27696 / 27.0000 | 11.10394 / 35.0000 | 10.31164 / 27.6250 |
| Ulysses cp2, tower replicated | 0 | 12.27654 / 27.2500 | 10.89869 / 33.0000 | 10.02553 / 21.5000 |

The two CP flavours read the same step 1 and 2 in each pairing (the tower path is the same, the text-side flavours part at step 3). Cut against replicated at step 1: +0.00042 loss (3.4e-5 relative) and 27.00 against 27.25 grad norm, in bf16 with the tower's attention and its merged tokens reduced in another order; the 09-10 fp32 probe of the same tower code read 1e-6 forward and 4e-6 worst gradient (`CPMM_NEW_TREE_2026-09-10.md`), and has not been rerun on this branch, so the 3.4e-5 is not located beyond that. Unseeded runs of the b200 recipes (default threshold, tower replicated) pass too: allgather `12.59487`, ulysses `12.38626` at step 1 (the recipes set no seed, so those move run to run).

Not run: the tower parity probe on this branch, TP x dynamic CP, more than one large image per batch.

## 5. What of the dynamic CP findings is PR 4639's own (user, 2026-09-15: record, do not raise; fix inside the dynamic CP PR)

- 4639's bug: `ContextParallelTransform` replaces inner attention by config type, so its two K3 mm CP recipes swap the MoonViT tower's `FlexInnerAttention` too and stop at step 1 in the tower's flex attention (`block_mask was created for a smaller length`); reproduced on 4639's own head `e06dcbee3` on this box (`K3_INT_20260915.md`). The dynamic CP PR carries the fix as its first commit (`7bfa51503`: `exclude_fqn_prefixes` on the transform, the recipes excluding `vision_encoder`), since a partitioned tower needs its own attention untouched anyway.
- Not 4639's: the tower encoding inside `spmd_local_context("dp")` is 4639's design (every rank encodes the whole batch, `gather_vision_embeds` picks the rank's tokens), and the cp-group lookup that returned None was this port's assumption from 4313; the 192-patch debug image under the 256 threshold is data; the type-checker registrations and the `no_typecheck` mask are for this port's own `_AllGather` use, `_PlainGradBoundary` and mask_mod, which 4639 never runs.
