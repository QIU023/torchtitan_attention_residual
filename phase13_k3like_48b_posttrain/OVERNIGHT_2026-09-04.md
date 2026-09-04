# Overnight 2026-09-04: PR-head matrices with the DP / EP meshes, and the near-ready branches rebased

Goal (user, evening of 09-04): the matrices the two live PR heads need, with data parallel and expert parallel on and off in every table; then the near-ready branches rebased onto upstream/main `6e2ac3dcd` and their numerics re-run with the SM120 guard-lift hack local to the run tree. Priority: TP/SP with the spmd declarations (one PR), PP cache balance, PP cache offload, AttnRes AC reuse, QB, LoRA.

## GPU queue (one chain, 8 cards, in this order)

| # | job | tree | cells | state |
|---|---|---|---|---|
| 1 | CP cp8 | `wt_cprun5` (323cf86fa) | cp8 | done: 12.54963 / 7.28587 / 2.95883 (step 1 is 1e-2 above cp2/cp4; the check is queued at the chain's end) |
| 2 | PP step-1 sign census | `wt_ppprobe8` (0e7cc5ea1 + dump hack) | dp1 x2 fresh caches, pp2 x vp4, pp8 x vp4 | done: dp1 x2 bitwise; pp2 0.223% / pp8 0.227% flips = 9.5% of the first update; all four reproduce the 09-03 rows bitwise |
| 3 | PP matrices on the rebased head | `wt_pprun3` / `wt_pprun3gn` (0e7cc5ea1) | bf16: dp1, pp2/pp4/pp8 x vp4, pp8 naive, even split, dp2, dp2 x ep2, dp2 x pp2, dp2 x ep2 x pp2, dp2 x pp4, dp2 x ep2 x pp4; fp32 norm: the five original cells | done: every PP row bit-identical at step 1 to its mesh without PP; dp1-based rows reproduce the previous head bitwise |
| 4 | PP 100-step curves | `wt_pprun3` | dp1, pp2 x vp4, pp8 x vp4, pp8 naive | done: the 32-sample set is memorized by all four (0.04-0.05 at step 100); delta pp8 crosses 1.0 at step 41 vs 26-31 |
| 5 | LoRA | `wt_lora` (cdedd17c9) | dp1, dp2, dp2 x ep2, for lora and qlora_mxfp4 | lora rows done (dp1/dp2 bitwise the old base; dp2 x ep2 12.48346 / 11.90808 / 10.50562); the qlora_mxfp4 seed build raised on main's SpmdType (`entry.axis_types`), fixed on the branch, rerun queued as item 15 |
| 6 | TP/SP + spmd | `wt_tpsprun` (8e7d4998d) | dp1 (partial_dtensor), dp1/dp2/dp2 x ep2 (spmd_types), tp2, tp2 no-SP, tp4, dp2 x tp2, dp2 x ep2 x tp2 | done: dp1/dp2/dp2 x ep2 bitwise partial_dtensor; tp2 12.54164 (= pre-rebase), tp2 no-SP 12.55332, tp4 12.52816, dp2 x tp2 12.53383, dp2 x ep2 x tp2 12.53826 at step 1 |
| 7 | AC reuse | `wt_acrun` (24aa8c08d + alias) | dp1, dp2, dp2 x ep2, flag off and on | done: all six bitwise main's rows; flag engaged (log line); peak dp1 off 12.68, dp1 on 12.81, dp2 off 7.48, dp2 on 8.17, dp2_ep2 off 7.59, dp2_ep2 on 8.28 GiB |
| 8 | QB | `wt_qbrun2` (47ec648b4) | dp1, dp2, dp2 x ep2, sign-step and quantile balancing | queued |
| 9 | PP cache offload | `wt_ppoffrun` (eb665b1b1 + alias) | pp2 x vp4, pp8 x vp4 with the store on pinned host memory | queued |
| 10 | PP balance | `wt_ppbalrun` (54a9e81ee + alias) | pp2 x vp4, rank 0 parks on rank 1 (Mooncake TE over TCP), as designed and with K3_PPBAL_KEEP_LOCAL=1 | queued |
| 11 | cp8 check | `wt_cprun5` | cp8 on the generic kernel; step-1 gradients cp8 vs dp1 / cp2 / generic | queued |
| 12 | census set 2 | `wt_ppprobe8` | dp1 vs FSDP dp2, dp1 vs 512-token micro-batches | queued |
| 13 | 100-step curves on streamed cc12m | `wt_pprun3` (+ cc12m alias) | dp1, pp2 x vp4, pp8 x vp4, pp8 naive | queued |
| 14 | 100-step curves at seed 43 | `wt_pprun3` | dp1, pp2 x vp4, pp8 x vp4 (mx3 got a SEED knob) | queued |
| 15 | QLoRA rerun | `wt_lora` (93f78b5ab) | dp1, dp2, dp2 x ep2 on qlora_mxfp4 | done: dp1 / dp2 bitwise the old base; dp2 x ep2 12.50176 / 11.99613 / 10.51754 |

Scripts: `matrix_scripts/rebase_main_pp3.sh`, `pp_probe_signs.sh`, `rebase_main_pp3_curves100.sh`, `rebase_main_lora.sh`, `rebase_main_tpsp.sh`, `rebase_main_ac2.sh`, `rebase_main_qb2.sh`; the chain lives in the session scratchpad (`bridge_pp_all.sh`, `bridge_after_lora.sh`).

## Branches (all on upstream/main `6e2ac3dcd` unless noted)

| branch | head | content | CPU checks | pushed |
|---|---|---|---|---|
| `pp_review3` | `fe34932ee` | PP, review round 2, the any-layer-count split, the 33-layer debug model | pyrefly 0; 28 PP tests | yes (`k3_pp_text` still at `0e7cc5ea1`, sync pending approval) |
| `cp_pr_candidate` = `k3_cp_text` | `223e97a23` (on `af9b6b195`) | marked stack copy + declarations + CP layer | 63 tests; pyrefly 0 on our files | yes (PR branch synced 09-04) |
| `lora_review1` | `93f78b5ab` (on `af9b6b195`) | LoRA export, QLoRA, packed TP; typed bases; mx_qat flavor out; packed experts carry main's SpmdType entry through | 19 LoRA tests + definitions; pyrefly = main | yes |
| `tpsp_spmd_review1` | `8e7d4998d` | tp_review2 + spmd_review2 (6 commits), clean rebase | 16 passed; pyrefly = main | yes |
| `ac_review2` | `24aa8c08d` | ac_review1 rebased (2 commits), typing fixes | 15 passed; pyrefly = main | yes |
| `qb_release` | `47ec648b4` | k3_qb content rebased, typing fixes | 26 passed; pyrefly = main | yes (force, lease on 0902c7a24) |
| `pp_offload_review1` | `20d83a8cb` (on fe34932ee) | `attn_res_cache_offload` ported onto `pp_review3`'s `RankStore` (old `e8897274d`) | 25 passed; pyrefly 0 on the files | yes |
| `pp_balance_review1` | `fd97aba9b` (on fe34932ee) | `pp_balance.py` and its pool test copied from `1c0c1416c`, knobs as a record on `pipeline_kimi_k3`; mooncake imports with the cu12 runtime wheel | 21 passed; pyrefly 0 on the files | yes |

## Results as they land

(filled by the session as rows arrive; the PR bodies get the tables)

- CP cp8: 12.54963 / 7.28587 / 2.95883 -> `PR_BODY_CP.md`; the CP table is complete, the cp8 check (generic kernel at cp8, step-1 gradients vs dp1/cp2) runs last in the chain.
- PP census: dp1 vs dp1 (fresh caches) bitwise; dp1 vs pp2 x vp4 0.223% sign flips, 9.45% first-update difference; dp1 vs pp8 x vp4 0.227%, 9.53%; every group ~1.1% element-wise, norms 2e-4 -> `PR_BODY_PP.md`, `PP_STEP10_SPREAD` sec 6, `REPLY_4312`.
- PP fp32-norm matrix on the rebased head complete: dp1 3.37903, pp2 3.47015, pp4 3.46001, pp8 3.44950, pp8 naive 3.58862 at step 10 (spread 6.2%; bf16 5.6%) -> `PP_STEP10_SPREAD` sec 2. bf16 rows dp1/pp2/pp4/pp8/pp8n reproduce the previous head bitwise -> `PR_BODY_PP.md`.
- PP mesh rows: dp2 12.49684 / 7.75700 / 3.44594; dp2 x ep2 12.49422 / 7.70749 / 3.55892; dp2 x pp2 12.49684 / 7.69817 / 3.46918; dp2 x ep2 x pp2 12.49422 / 7.70147 / 3.63188; dp2 x pp4 12.49684 / 7.75872 / 3.42694; dp2 x ep2 x pp4 12.49422 / 7.71806 / 3.52509 -> `PR_BODY_PP.md` (table complete). dp2 reads a different batch (dataset sharded by dp rank), so step 1 compares within a dp group.
- PP 100-step curves (debug set, seed 42): dp1 / pp2 / pp8 / pp8 naive cross 1.0 at steps 31 / 26 / 41 / 30, end at 0.043 / 0.044 / 0.047 / 0.046 -> `PR_BODY_PP.md`, `PP_STEP10_SPREAD` sec 7; cc12m and seed-43 runs queued.
- LoRA: lora dp1 / dp2 bitwise the old base, dp2 x ep2 12.48346 / 11.90808 / 10.50562 -> `PR_BODY_LORA.md`. qlora_mxfp4: the packed-experts sharding translation read `entry.axis_types`, gone on main (`SpmdType.local_type`), and its TP inner-dim refusal fired on main's always-declared expert TP layout; the entry now passes through (the model refuses TP anyway), with a trainer-path CPU test; rerun queued.
- TP/SP + spmd: dp1 12.52977 / 7.27107 / 2.98077 (both backends); dp2 12.53137 / 7.31248 / 3.15823; dp2 x ep2 12.53146 / 7.20212 / 3.10296; tp2 12.54164 / 7.35554 / 3.16327; tp2 no-SP 12.55332 / 7.38015 / 3.00522; tp4 12.52816 / 7.03474 / 3.09639; dp2 x tp2 12.53383 / 7.26831 / 3.16722; dp2 x ep2 x tp2 12.53826 / 7.30627 / 3.10813 -> `PR_BODY_TP_SP_SPMD.md` (table complete).
- AC reuse: six cells bitwise main (dp1 / dp2 / dp2 x ep2, flag off and on); peaks dp1 off 12.68 GiB, dp1 on 12.81 GiB, dp2 off 7.48 GiB, dp2 on 8.17 GiB, dp2_ep2 off 7.59 GiB, dp2_ep2 on 8.28 GiB -> `PR_BODY_AC.md`.
- Correction: the PP row labelled 'even split' was the `first/last_stage_less_layers=0` cell, a habit from the retired 32-layer flavor; on the 30-layer model at pp2 x vp4 it cuts 4 x 6 stages then 3 and 3 (the trainer's log), not 8 x 4. Relabelled in the body. The debug model is 30 layers (`72a2b5344`), 32 units with the embedding and the head, so the DEFAULT less_layers=1 is the split every shape up to 32 stages divides.

## Correction, evening of 09-04: the debug model is 33 layers, and the PP matrix reruns

The user's design: the debug model is 33 layers (12 x 2 + 9, the 93-layer model's partial block; 35 units with the embedding and the head), which no pipeline shape divides. The 30-layer model of `72a2b5344` was chosen because its 32 units divide every shape up to 32 stages, which is the opposite of what the pipeline has to handle, so every PP row above was measured on splits that never exercised the uneven case. Two commits on `pp_review3`: the split rule (the multiple of pp nearest to units / layers_per_stage, core given the split and not the knob, since its ceiling refuses 35 units at 4 per stage) and the 33-layer flavor. `pp_offload_review1` and `pp_balance_review1` rebased on top. The PP matrix (13 cells), the census, the offload and balance cells, and the curves rerun on the 33-layer model; the 30-layer rows stay in this log's results section as history.

Queue after QB (re-chained 09-04 evening): cp8 check -> PP33 matrix (11 cells) -> PP33 census -> offload33 -> balance33 -> qlora rerun -> census set 2 (33) -> cc12m curves (33) -> seed-43 curves (33). Run trees: `wt_pprun33`, `wt_ppprobe33`, `wt_ppoffrun33`, `wt_ppbalrun33`; seeds under `.mx3_seeds_main33`.
- PP33 matrix (33 layers, fe34932ee): dp1 12.41967 / 7.56783 / 3.45908; pp2 x vp4 12.41967 / 7.47862 / 3.42131; pp4 x vp4 12.41967 / 7.57579 / 3.36337; pp8 x vp4 12.41967 / 7.51825 / 3.37366; pp8 whole-stack 12.41967 / 7.60614 / 3.42516; dp2 12.40417 / 7.37116 / 3.30135; dp2 x ep2 12.40257 / 7.45076 / 3.38303; dp2 x pp2 12.40417 / 7.28299 / 3.40680; dp2 x ep2 x pp2 12.40257 / 7.49486 / 3.24775; dp2 x pp4 12.40417 / 7.48020 / 3.33841; dp2 x ep2 x pp4 12.40257 / 7.39910 / 3.24169 -> `PR_BODY_PP.md` (table complete); splits read from the trainer log: 4/5/5/4/4/4/4/3, 2/3/3/2...2/1, 1/2/2/1...1/0.
- User (evening): the fp32 grad-norm matrix runs again on the 33-layer model (the maintainer reads the spread as a problem); `wt_pprun33gn` = fe34932ee + SM120 hack + PR 4135's reduction; queued right after the 33-layer census, before offload/balance. The body carries both tables, one caption line each.
- PP33 census: dp1 vs pp2 0.267% flips (10.3% of the first update), dp1 vs pp8 0.277% (10.5%); norms 2e-4; every group ~1.3% element-wise -> `PR_BODY_PP.md`, `PP_STEP10_SPREAD` sec 8, `REPLY_4312`. The census script died on a quoting slip after the dumps; the old chain then started the offload cells, so the queue is re-armed behind them: fp32-norm matrix -> balance -> cp8 check -> qlora -> QB rest -> census2 -> curves.
- PP offload (33 layers, 20d83a8cb): pp2 x vp4 and pp8 x vp4 bitwise the on-device rows; peaks {'pp2 off': 13.98, 'pp2 on': 13.98, 'pp8 off': 8.67, 'pp8 on': 8.67} -> `PR_BODY_PP_OFFLOAD.md`.
- User: every vp cell in both matrices gets its whole-stack twin (`_pp_naive`): 6 cells x 2 trees queued right after the fp32-norm matrix (`pp33_naive_matrix.sh`, `pp33gn_naive_matrix.sh`); rows added to both tables.
- PP33 fp32-norm matrix (wt_pprun33gn): dp1 12.41967 / 7.57490 / 3.34752; pp2 x vp4 12.41967 / 7.49055 / 3.33238; pp4 x vp4 12.41967 / 7.57446 / 3.43256; pp8 x vp4 12.41967 / 7.49769 / 3.49425; pp8 whole-stack 12.41967 / 7.51799 / 3.30288; dp2 12.40417 / 7.37116 / 3.30122; dp2 x ep2 12.40257 / 7.45076 / 3.37020; dp2 x pp2 12.40417 / 7.29014 / 3.42404; dp2 x ep2 x pp2 12.40257 / 7.49486 / 3.24388; dp2 x pp4 12.40417 / 7.49642 / 3.31773; dp2 x ep2 x pp4 12.40257 / 7.40208 / 3.31594 -> `PR_BODY_PP.md`. Step 1 identical to the bf16 table cell by cell (the bf16 norm is quantised to 0.125 at magnitude 16, the clip coefficient equal in every cell); step-10 spread of the five dp1 cells 3.30-3.49 (5.8%) against 3.36-3.46 (2.8%) in bf16.
- PP33 bf16 whole-stack twins: pp2 12.41967 / 7.66420 / 3.32480; pp4 12.41967 / 7.64929 / 3.49334; dp2 x pp2 12.40417 / 7.32403 / 3.36641; dp2 x ep2 x pp2 12.40257 / 7.39084 / 3.34226; dp2 x pp4 12.40417 / 7.60047 / 3.25333; dp2 x ep2 x pp4 12.40257 / 7.30184 / 3.25535 -> `PR_BODY_PP.md` (bf16 table complete, 17 rows); every twin's step 1 equals its delta row's.
- PP33 fp32-norm whole-stack twins: pp2 12.41967 / 7.61479 / 3.35875; pp4 12.41967 / 7.68891 / 3.43122; dp2 x pp2 12.40417 / 7.32403 / 3.36488; dp2 x ep2 x pp2 12.40257 / 7.39084 / 3.34489; dp2 x pp4 12.40417 / 7.61220 / 3.25558; dp2 x ep2 x pp4 12.40257 / 7.30184 / 3.26341 -> `PR_BODY_PP.md` (both 17-row tables complete).
- PP balance (33 layers, fd97aba9b, Mooncake TE over TCP): pp2 x vp4 as designed and with KEEP_LOCAL both bitwise the unbalanced row (12.41967 / 7.47862 / 3.42131); 1,440 tensors / 1,760 MiB parked over the run; rank 0 peak 13.98 GiB unchanged -> `PR_BODY_PP_BALANCE.md`.
- cp8 check: generic Ulysses at cp8 reads the same step 1 (12.54963; 7.30435 / 3.01451); gradients dp1 vs cp8 1.18e-2 / 6.25e-2 / 3.31e-1, dp1 vs cp8 generic 1.17e-2 / 6.1e-2 / 3.27e-1, cp2 vs cp8 9.5e-3 / 6.4e-2 / 3.7e-1, cp8 packed vs generic 1.76e-4 / 1.31e-3 / 3.1e-2 (24 of 750 identical) -> `PR_BODY_CP.md`. The offset is the CP degree's, not the packing's.
- QLoRA (packed MXFP4) after the fix: dp1 12.48328 / 12.00891 / 10.42474 and dp2 12.50176 / 12.00203 / 10.45663 bitwise the old base; dp2 x ep2 12.50176 / 11.99613 / 10.51754 (packed experts under EP: step 1 equal to dp2) -> `PR_BODY_LORA.md` (table complete).
