# Fork branch inventory (QIU023/torchtitan, 2026-09-16)

114 remote branches, 130 local (about 40 local-only: `pr4xxx` mirrors of upstream PR heads, `k3_on_4025_pre_rebase_*` and `k3_on_4025_backup` snapshots, `wip_*`, `*clean`, `*_review` scratch), 288 worktrees under `/tmp` (93 with an August HEAD, 195 September). Distances are against upstream main `1949c297f`. Nothing here is deleted without the user's word; the proposal is the group heading.

## A. PR branches (open upstream PRs)

| branch | last commit | head | ahead of main | note |
| --- | --- | --- | ---: | --- |
| `k3_pp_text` | 2026-09-14 | `de6f29514` | 37 | PR 4312, review branch pp_review4 (same head); pp_review4_consume6840 holds the held-back fixes |
| `k3_ac_reuse_attention` | 2026-09-15 | `ea1316606` | 2 | PR 4656 (draft), = ac_review3, main tip + recompute + test |
| `k3_cp_mm` | 2026-09-15 | `774e0b9b5` | 4 | PR 4380 (draft), = cpmm_review1, on PR 4639 |
| `k3_pp_mm` | 2026-09-15 | `384d576dc` | 40 | PR 4381 (draft), = dep_review1, on PR 4312 |
| `k3_cp_text` | 2026-09-06 | `61a73ca6c` | 5 | PR 4313, ours; superseded by the maintainers' 4639 (kept open, not touched); = cp_review5 |

## B. Merged or closed upstream: deletable

| branch | last commit | head | ahead of main | note |
| --- | --- | --- | ---: | --- |
| `k3_tp_sp` | 2026-09-15 | `c0dad3e91` | 12 | PR 4499 merged 09-16 as 3ce746c71; = tpsp_review4 |
| `tpsp_review4` | 2026-09-15 | `c0dad3e91` | 12 | review of 4499 |
| `tpsp_review3` | 2026-09-05 | `c0ab32d07` | 4 | review of 4499 |
| `tp_review1` | 2026-09-02 | `9ff6f8635` | 28 | review of 4499 |
| `tp_review2` | 2026-09-03 | `80d7e0951` | 4 | review of 4499 |
| `tpsp_spmd_review1` | 2026-09-05 | `2e2230cbb` | 6 | review of 4499 |
| `k27_vision_tables_tp` | 2026-09-11 | `d8120354e` | 1 | one of the two K2.5 commits inside 4499 |
| `tp_sp_on_main` | 2026-09-11 | `22eeec412` | 4 | 4499 scratch rebase |
| `tp_sp_on_main_da2f82670` | 2026-09-11 | `2a57b3fc3` | 8 | 4499 scratch rebase |
| `tp_sp_on_4527` | 2026-09-09 | `54ebe13ee` | 9 | 4499 scratch |
| `tp_sp_on_4500` | 2026-09-09 | `3fdd8c40d` | 7 | 4499 scratch |
| `k3_qb` | 2026-09-09 | `895f4d6e9` | 4 | PR 4412 closed in favour of 4577 |
| `qb_review4` | 2026-09-09 | `895f4d6e9` | 4 | review of 4412 |
| `k3_spmd_decl` | 2026-09-05 | `dbc60701d` | 1 | PR 4492 closed |
| `spmd_decl_review1` | 2026-09-05 | `dbc60701d` | 1 | review of 4492 |
| `spmd_review1` | 2026-09-03 | `4b88ada6b` | 30 | review of 4492 |
| `spmd_review2` | 2026-09-04 | `8955c0778` | 6 | review of 4492 |
| `spmd_probe` | 2026-09-03 | `b151fcf06` | 31 | 4492 probe |

## C. Superseded review branches: deletable once the PR they served moved on

| branch | last commit | head | ahead of main | note |
| --- | --- | --- | ---: | --- |
| `pp_review1` | 2026-09-02 | `1c0c1416c` | 19 | 4312 |
| `pp_review2` | 2026-09-03 | `3af70c9ee` | 23 | 4312 |
| `pp_review3` | 2026-09-05 | `a3be242bf` | 17 | 4312 |
| `pp_review5` | 2026-09-10 | `6042863a4` | 19 | 4312, a later scratch than review4 |
| `cp_review1` | 2026-09-01 | `b85c2a078` | 12 | 4313 |
| `cp_review2` | 2026-09-03 | `adc012ce4` | 40 | 4313 |
| `cp_review3` | 2026-09-04 | `e63f54d6b` | 7 | 4313 |
| `cp_review4` | 2026-09-04 | `f6ca064be` | 11 | 4313 |
| `cp_review5` | 2026-09-06 | `61a73ca6c` | 5 | 4313, = k3_cp_text |
| `ac_review1` | 2026-09-02 | `ca9e497f8` | 2 | old flag design |
| `ac_review2` | 2026-09-05 | `a02b5e195` | 3 | old flag design |
| `lora_review1` | 2026-09-04 | `93f78b5ab` | 8 | LoRA, no PR yet |
| `lora_review2` | 2026-09-10 | `72bbcb639` | 13 | LoRA, no PR yet |
| `ep_review1` | 2026-09-03 | `cd8856107` | 26 | EP, = k3_ep |
| `pp_balance_review1` | 2026-09-05 | `3bb3fc6cf` | 18 | pp_balance (integration only) |
| `pp_offload_review1` | 2026-09-05 | `240c3205b` | 18 | cache offload (integration only) |
| `k3_pp_mm_v2` | 2026-09-10 | `c87097ae5` | 21 | old DEP head, replaced by k3_pp_mm |
| `k3_cp_mm_v2` | 2026-09-10 | `a063a3d0e` | 6 | old dynamic CP head, replaced by k3_cp_mm |

## D. Integration trees: keep the current one, the rest are tagged history

| branch | last commit | head | ahead of main | note |
| --- | --- | --- | ---: | --- |
| `k3_on_4025` | 2026-09-15 | `ae0c7372c` | 82 | current integration tree, tag k3_int_20260915 |
| `k3_int_20260912_cp` | 2026-09-13 | `dddade302` | 57 | tagged |
| `k3_int_20260912` | 2026-09-13 | `f201b9ac0` | 50 | tagged |
| `k3_int_20260910` | 2026-09-10 | `acf7c2ae6` | 58 | tagged |
| `k3_int_20260906` | 2026-09-07 | `4f7c87fc9` | 51 | tagged |
| `mm18_int` | 2026-09-05 | `40bb5bbb7` | 26 | 09-05 mm18 scratch |
| `mm18_lora_int` | 2026-09-06 | `4716f8ee6` | 36 | 09-05 mm18 scratch |

## E. Feature branches without a PR yet (our own, on a recent main): keep, each is a future PR or a port source

| branch | last commit | head | ahead of main | note |
| --- | --- | --- | ---: | --- |
| `k3_compile_blocks` | 2026-09-14 | `2d7b85225` | 1 | compile PR kit (PR_BODY_COMPILE.md) |
| `k3_moonep_seam` | 2026-09-14 | `a706a881d` | 5 | MoonEP on titan seams |
| `k3_mx_qat` | 2026-09-10 | `a8a850860` | 2 | MXFP4 QAT |
| `k3_mtp_layers` | 2026-09-10 | `937fe9276` | 1 | MTP |
| `k3_dist_muon` | 2026-09-10 | `1f9831580` | 1 | DistMuon (ours, not 4596) |
| `k3_linear_graft` | 2026-09-10 | `3b6f03347` | 3 | 48B graft |
| `k3_released_format` | 2026-09-10 | `65badf428` | 1 | released checkpoint format |
| `k3_empty_optimizer` | 2026-09-10 | `1c075f212` | 2 | LoRA with nothing trainable |
| `k3_lora_extras` | 2026-09-02 | `2cd6a35e0` | 6 | LoRA extras |
| `k3_qat` | 2026-09-02 | `99defeab8` | 3 | older QAT head |
| `k3_mtp` | 2026-09-02 | `5ce30dbe1` | 3 | older MTP head |
| `k3_pp_transport` | 2026-09-11 | `8126172f8` | 26 | PP transport (neighbor P2P) |
| `pp_runtime_client` | 2026-09-08 | `464421e13` | 20 | PP runtime client |
| `k3_ep` | 2026-09-03 | `cd8856107` | 26 | EP head (EP merged upstream earlier; check before deleting) |
| `k3_ep_cell` | 2026-09-04 | `b20138da5` | 1 | EP cell |
| `k3_kda_impl` | 2026-08-29 | `3b34e3068` | 1 | KDA impl |
| `pp_fp64_probe` | 2026-09-13 | `b3d2166ef` | 38 | 4312 numerics probe |
| `k3_pr_classified_v2` | 2026-09-09 | `a2e0c0f74` | 23 | PR classification scratch |
| `grad-norm-fp32` | 2026-08-13 | `5e88ff897` | 1 | one commit, 08-13 |

## F. Old tree (August and earlier, hundreds of commits off main): archive as tags and delete

| branch | last commit | head | ahead of main | note |
| --- | --- | --- | ---: | --- |
| `main` | 2026-08-29 | `ae5fcd3f9` | 110 | fork main, the old integration alias, 08-29 |
| `gb200_fixes` | 2026-08-27 | `507c34fac` | 435 |  |
| `k3_tp_declarative` | 2026-08-27 | `507c34fac` | 435 |  |
| `k3_full_tree_draft` | 2026-08-21 | `afc3e4287` | 409 |  |
| `k3_merge_upstream` | 2026-08-21 | `afc3e4287` | 409 |  |
| `k3_cp_declarative` | 2026-08-21 | `7dff28ddb` | 406 |  |
| `k3_pr_classified` | 2026-08-21 | `856a9aff4` | 15 |  |
| `k3_pr_pp_clean` | 2026-08-21 | `703f26840` | 1 |  |
| `k3_pr_ep_clean` | 2026-08-21 | `58d5834ee` | 1 |  |
| `k3_pr_tp_clean` | 2026-08-21 | `2ff652ff1` | 1 |  |
| `k3_pr_ep` | 2026-08-19 | `89355489b` | 384 |  |
| `k3_pr_tp` | 2026-08-19 | `1e0ddb73b` | 384 |  |
| `k3_pr_base` | 2026-08-19 | `1f5baf9a5` | 383 |  |
| `origin` | 2026-08-18 | `f489aca86` | 381 | a branch literally named origin |
| `attention_residual_dev` | 2026-08-18 | `f489aca86` | 381 | retired per CLAUDE.md |
| `dep_exp_impl` | 2026-08-18 | `f489aca86` | 381 |  |
| `migrate_stream_dtensor` | 2026-08-15 | `7f08efeb2` | 350 |  |
| `migrate_tp_attnres_tail` | 2026-08-14 | `5e5f20e4c` | 332 |  |
| `align_4025` | 2026-08-14 | `643ba7543` | 319 |  |
| `migrate_carrier_tensor` | 2026-08-14 | `643ba7543` | 319 |  |
| `migrate_step3_cp` | 2026-08-14 | `56255db22` | 319 |  |
| `core_revert_probe` | 2026-08-14 | `066c2b453` | 310 |  |
| `tp_declarative_refactor` | 2026-08-13 | `4069c9871` | 311 |  |
| `k3_pr_a_tp_kda` | 2026-08-05 | `a146d1bf2` | 222 |  |
| `k3_pr_b_ep_grouped` | 2026-08-05 | `a146d1bf2` | 222 |  |
| `k3_pr_c_pp_attnres` | 2026-08-05 | `a146d1bf2` | 222 |  |
| `k3_pr_d_cp_ulysses` | 2026-08-05 | `a146d1bf2` | 222 |  |
| `k3_models_move` | 2026-08-03 | `22d36d8ed` | 198 |  |
| `k3_config_tree` | 2026-08-03 | `ddfb3612b` | 193 |  |
| `k3_refactor` | 2026-08-02 | `1e9493198` | 193 |  |
| `kimi_k3_upstream_pr` | 2026-07-27 | `b5abac262` | 2 | 07-27 |
| `experiments_rl_unmerged` | 2026-07-17 | `be1a2a57c` | 83 | 07-17, the SGLang/Monarch RL the CLAUDE.md says to keep |
| `k3_pp` | 2026-08-24 | `768a80d80` | 1 | 08-24, one commit |
| `k3_cp` | 2026-08-24 | `135abcf36` | 1 | 08-24, old PR 4380 base |
| `k3_tp` | 2026-08-27 | `771e75e73` | 3 | 08-27 |
| `qwen25_vl_video_vla` | 2026-05-20 | `5ee6380dd` | 79 | May, VLA |
| `pixelshuffle_projector` | 2026-05-20 | `b627122f8` | 79 | May |
| `perceiver_resampler_projector` | 2026-05-20 | `0026ea5cd` | 79 | May |
| `step_decay_lr_scheduler` | 2026-05-20 | `b521799d2` | 78 | May |
| `old_tree_full_impl` | 2026-05-12 | `a7ab26117` | 62 | May |
| `vlm-sglang-overlay` | 2026-05-10 | `b1decbe35` | 56 | May |
| `phase11_kimi_linear_447m_aligned` | 2026-05-09 | `82b08e740` | 54 | May |

## G. Live review branches of the open PRs: keep

| branch | last commit | head | ahead of main |
| --- | --- | --- | ---: |
| `ac_review3` | 2026-09-15 | `ea1316606` | 2 |
| `cpmm_review1` | 2026-09-15 | `774e0b9b5` | 4 |
| `dep_review1` | 2026-09-15 | `384d576dc` | 40 |
| `pp_review4_consume6840` | 2026-09-15 | `e4955d2d5` | 40 |
| `pp_review4` | 2026-09-14 | `de6f29514` | 37 |

## Worktrees

288 `/tmp/wt_*` and similar worktrees. Removing a worktree directory does not touch its branch; the risk is uncommitted work inside one (the KDA guard lifts and the `rl` flavor are deliberate uncommitted edits in the live ones). Proposal: a `git status --porcelain` sweep, then `git worktree remove` for every clean worktree whose HEAD is older than a week and not one of the live ones (`wt_int0915`, `wt_int0915_rl`, `wt_cpmm_4639`, `wt_dep_4312`, `wt_ac_4656`, `wt_verl_0915`, `wt_4639`, `wt_pr4312_plain`, `wt_main_plain`); dirty ones listed for the user first.
