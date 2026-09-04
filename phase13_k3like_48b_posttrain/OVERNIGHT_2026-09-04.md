# Overnight 2026-09-04: PR-head matrices with the DP / EP meshes, and the near-ready branches rebased

Goal (user, evening of 09-04): the matrices the two live PR heads need, with data parallel and expert parallel on and off in every table; then the near-ready branches rebased onto upstream/main `6e2ac3dcd` and their numerics re-run with the SM120 guard-lift hack local to the run tree. Priority: TP/SP with the spmd declarations (one PR), PP cache balance, PP cache offload, AttnRes AC reuse, QB, LoRA.

## GPU queue (one chain, 8 cards, in this order)

| # | job | tree | cells | state |
|---|---|---|---|---|
| 1 | CP cp8 | `wt_cprun5` (323cf86fa) | cp8 | running at 22:xx |
| 2 | PP step-1 sign census | `wt_ppprobe8` (0e7cc5ea1 + dump hack) | dp1 x2 fresh caches, pp2 x vp4, pp8 x vp4 | queued |
| 3 | PP matrices on the rebased head | `wt_pprun3` / `wt_pprun3gn` (0e7cc5ea1) | bf16: dp1, pp2/pp4/pp8 x vp4, pp8 naive, even split, dp2, dp2 x ep2, dp2 x pp2, dp2 x ep2 x pp2, dp2 x pp4, dp2 x ep2 x pp4; fp32 norm: the five original cells | queued |
| 4 | PP 100-step curves | `wt_pprun3` | dp1, pp2 x vp4, pp8 x vp4, pp8 naive | queued |
| 5 | LoRA | `wt_lora` (cdedd17c9) | dp1, dp2, dp2 x ep2, for lora and qlora_mxfp4 | queued |
| 6 | TP/SP + spmd | `wt_tpsprun` (8e7d4998d) | dp1 (partial_dtensor), dp1/dp2/dp2 x ep2 (spmd_types), tp2, tp2 no-SP, tp4, dp2 x tp2, dp2 x ep2 x tp2 | queued |
| 7 | AC reuse | `wt_acrun` (7d2e0dd51 + alias) | dp1, dp2, dp2 x ep2, flag off and on | queued |
| 8 | QB | `wt_qbrun2` (2fe12cc32) | dp1, dp2, dp2 x ep2, sign-step and quantile balancing | queued |

Scripts: `matrix_scripts/rebase_main_pp3.sh`, `pp_probe_signs.sh`, `rebase_main_pp3_curves100.sh`, `rebase_main_lora.sh`, `rebase_main_tpsp.sh`, `rebase_main_ac2.sh`, `rebase_main_qb2.sh`; the chain lives in the session scratchpad (`bridge_pp_all.sh`, `bridge_after_lora.sh`).

## Branches (all on upstream/main `6e2ac3dcd` unless noted)

| branch | head | content | CPU checks | pushed |
|---|---|---|---|---|
| `pp_review3` = `k3_pp_text` | `0e7cc5ea1` | PP, review round 2 | pyrefly 0 on the PP files; 29 passed / 2 skipped | yes (PR branch synced 09-04) |
| `cp_pr_candidate` = `k3_cp_text` | `223e97a23` (on `af9b6b195`) | marked stack copy + declarations + CP layer | 63 tests; pyrefly 0 on our files | yes (PR branch synced 09-04) |
| `lora_review1` | `cdedd17c9` (on `af9b6b195`) | LoRA export, QLoRA, packed TP; typed bases; mx_qat flavor out | 30 passed; pyrefly = main | yes |
| `tpsp_spmd_review1` | `8e7d4998d` | tp_review2 + spmd_review2 (6 commits), clean rebase | 16 passed; pyrefly = main | yes |
| `ac_review2` | see log | ac_review1 rebased (2 commits), typing fixes | 15 passed | yes |
| `qb_release` | see log | k3_qb content rebased, typing fixes | 26 passed | yes (force, lease on 0902c7a24) |
| PP cache offload / pp_balance | not yet | live on the old `pp_review1` line (`e8897274d`, `1c0c1416c`); need a port onto `pp_review3`'s stage class | | |

## Results as they land

(filled by the session as rows arrive; the PR bodies get the tables)
