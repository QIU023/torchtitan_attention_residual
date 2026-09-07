# Review branches against upstream main, 2026-09-07

Upstream main tip is `d263ca0a1` (2026-09-06). Its last four commits matter here:

| commit | what | effect on this box |
| --- | --- | --- |
| `23ebdfe87` #4484 (09-05) | router fp32 backward, bf16x9 matmuls | `enable_fp32_matmul_emulation_with_bf16x9` raises on any cc >= 10.0 GPU whose torch lacks BFX9 (pytorch#195301); the 5060 Ti trips it, neither venv's nightly has it, no override -- **main tip cannot run here** (mx3 shows `ASSERT-SEED-FAIL` on every cell) |
| `966befce6` #4493 | cudagraph capture of single-stage PP steps (`trainer.py`) | conflicts with the CP branch's copied CP-stack lines in `trainer.py` |
| `a423ddeb0` #4498, `d263ca0a1` | DeepSeek V4 MTP conversion; rl renderers | none |

`aecbb8199` (#4494, 09-05, after the EP merge of 09-03) is the last commit this box can run; rebases for
numerics use it and name it.

| branch | head | rebase onto `aecbb8199` | rebase onto tip `d263ca0a1` |
| --- | --- | --- | --- |
| `spmd_decl_review1` (4492) | `dbc60701d` | clean | clean |
| `tpsp_review3` (4499) | `c0ab32d07` | clean | clean |
| `cp_review5` = `k3_cp_text` (4313) | `61a73ca6c` | clean | conflict in `trainer.py` (#4493 vs the copied 4322/4449/4450 lines; drops when that stack lands) |
| `pp_review3` (4312) | `a3be242bf` | conflict in `kimi_k3/parallelize.py` at the first PP commit (base `6e2ac3dcd` predates the EP merge; rerere holds an earlier resolution) | same |
| `pp_balance_review1`, `pp_offload_review1` | on `pp_review3` | same as PP | same |
| `ac_review2` | `a02b5e195` | clean | clean |
| `lora_review1` | `93f78b5ab` | clean | clean |
| `qb_release` (4412) | `a4658eefe` | clean -> `qb_review2` = `d0d75fa8b` on `aecbb8199`, QB tests 14/14, one-cache matrix below | clean but cannot run (the gate) |

QB: #4412 was waiting on the EP merge; EP merged 09-03. `qb_review2` carries the two QB commits on
`aecbb8199`; `matrix_scripts/qb_rebase_onecache.sh` runs the sign-step control and quantile balancing on
one compile cache (dp1 / dp2 / dp2 x ep2, partial_dtensor per cell since the declarations PR is not in main).
Results go to `PR_BODY_QB.md` when the run ends; syncing `k3_qb` and undrafting are the user's.
