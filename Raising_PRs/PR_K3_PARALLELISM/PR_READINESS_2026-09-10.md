# PR readiness, 2026-09-10 (end of day)

Every branch below is on the fork `QIU023/torchtitan`; unless a stack is named, the branch sits directly on main `ac10ca48f` (#4527) and is independent of the parallelism PRs. Each body under `Raising_PRs/PR_K3_PARALLELISM/` has a `--- PASTE BEGIN ---` marker; paste from there. Commit messages carry no trailers, no cherry-pick lines, no PR URLs. "pyrefly" below means: the pinned 0.45.1 (`/tmp/venv_pyrefly/bin/pyrefly`) reports the same 21 project errors as clean main, or the system 1.2.0 the same 58/64, with no delta in the touched files (the hook itself is `language: system`, so whatever `pyrefly` is on PATH runs project-wide and strips ignore comments from ~27 unrelated files; that collateral was reverted by list before every commit).

## Published PRs (heads pushed, bodies ready to paste)

| PR | branch = head | base / stack | body | state |
| --- | --- | --- | --- | --- |
| #4312 text PP | `k3_pp_text` = `75045fed5` | main; carries the transport round 3 (warm-up default, Elfie's isolation opt-in) | `PR_BODY_PP.md` (transport section) + `REPLY_4312_2026-09-10.md` (two threads for Tianyu) | do not rebase; the user pastes the replies |
| #4499 TP/SP | `k3_tp_sp` = `9a62f5229` (PR branch, untouched); review branch `tp_sp_on_main` = `d4d6e774c` (4 commits on main `da2f82670`, Shuhua's 18 comments addressed 2026-09-11) | main `da2f82670`; bottom commit = the K2.5 tables PR | `REPLY_4499_2026-09-11.md` (top-level + 18 replies), `PR_BODY_TP_SP_v2_DRAFT.md` (A100 100-step table to re-measure on the new stack; tp4 rows need a tower whose heads divide by 4) | waiting for the A100 rerun before the PR branch moves; note `k3_tp_sp`'s bottom commit `c0e1584df` is mis-attributed to Shuhua Yu (my identity slip on 2026-09-10) and gets corrected by that same force-push |
| #4381 mm PP | `k3_pp_mm` = `c87097ae5` | #4312, restacked DEP | `PR_BODY_PP_MM_v2.md` | ready |
| #4380 mm CP | `k3_cp_mm` = `a063a3d0e` | #4500 stack, dynamic vision CP (carries the fsdp zero-valued dependency guard) | `PR_BODY_CP_MM_v2.md` | ready |

## New standalone branches (pushed today; bodies written; no PR opened yet)

| branch = head | commits | body | verification | state |
| --- | --- | --- | --- | --- |
| `k27_vision_tables_tp` = `d8120354e` (author corrected 2026-09-11; content = `c0e1584df`) | 1 | `PR_BODY_K27_TABLES.md` | K2.5 dp2 x tp2 type check gets past the tables with it (fails on main) | ready, small |
| `k3_empty_optimizer` = `1c075f212` | 2 (change + tests) | `PR_BODY_EMPTY_OPTIMIZER.md` | 33 CPU tests (4 new); pyrefly no delta | ready, core-only |
| `k3_released_format` = `65badf428` | 1 | `PR_BODY_RELEASED_FORMAT.md` | 27 tests, 0 skipped against the released-layout artifact; `report_arch` loads 658 keys and trains | ready |
| `k3_compile_blocks` = `5ee84f0c4` | 1 | `PR_BODY_COMPILE.md` | dp1 3 steps compiled bitwise with eager, same 12.64 GiB peak; main without it raises NotImplementedError | ready |
| `k3_mx_qat` = `a8a850860` | 2 (port + scope test) | `PR_BODY_QAT.md` | 18 tests; dp1 3 steps QAT vs plain | ready |
| `k3_moonep_seam` = `3756a97d5` | 1 | `PR_BODY_MOONEP.md` | 17 tests; standard backend bitwise with main at dp1 and dp2 x ep2; moonep EP=1 fallback bitwise; ep2 refused by the not-installed guard (no MoonEP package here) | draft (transport not exercised on this box) |
| `lora_review2` = `72bbcb639` | 9 (the user's 8 + one flake8 fix) | `PR_BODY_LORA.md` | 19 LoRA tests; dp1/fsdp2 3 steps bitwise before/after the fix | ready; the user decides whether the fix commit stays |
| `k3_mtp_layers` = `937fe9276` | 1 | `PR_BODY_MTP.md` | 37 tests; weight-0 MTP identical to plain under the same CE; MTP layer costs 1.08 GiB | ready; MTP x PP stays on the integration tree |
| `k3_dist_muon` = `1f9831580` | 1 | `PR_BODY_MUON.md` | 390 Muon layouts cover 390 params; step-1 bitwise with AdamW at dp1 / fsdp2 / ep2 x fsdp2 | draft, waits for #4353 |
| `k3_linear_graft` = `3b6f03347` | 3 (Kimi-Linear shapes + `kimi_linear_48b` / `kimi_linear_debugmodel`; alpha-gated reads + `_gated` suffix; text-only checkpoint spelling) | `PR_BODY_LINEAR_GRAFT.md` | 20 CPU tests + 2 GPU; K3 debug model bitwise with main; gated at alpha=0 equals the plain backbone bitwise on GPU and CPU; official-spelling load leaves the alphas at init | fork-internal (option A) until the A/B + upstream decision; the unbounded Kimi-Linear decay runs on attn-gym's eager reference path (the fused chunk kernels take decays in about [-5.9, 0]) |
| `k3_ac_reuse_attention` = `6ef880995` | 3 | `PR_BODY_AC_REUSE.md` | 16 tests; dp1 10 steps bitwise with main for both flavors; `ac_reuse_attention` 246 -> 305 tps at +0.13 GiB on the debug model | ready |

## Not extracted

- `attn_res_cache_offload` (763e24083) and `pp_balance` (222108864): stack on #4312's runtime; wait for the transport decision on that PR.
- Kimi-Linear-48B graft: ported under option A as `k3_linear_graft` (table above); whether it goes upstream in the K3 folder or as a `kimi_linear` folder (option B) is still the decision in `phase13_k3like_48b_posttrain/GRAFT_48B_DESIGN_2026-09-10.md`; a graft train needs the 48B weights and a bigger box, and a training kernel for the unbounded decay.
- The integration tree `k3_int_20260910` = `da7f9e348` stays on `ac10ca48f`; the trial rebase onto `d398a8fb9` is not pushed (#4572 moves pp2 numerics from step 2).
