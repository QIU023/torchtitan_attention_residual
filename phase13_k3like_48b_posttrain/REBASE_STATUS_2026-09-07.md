# Review branches against upstream main, 2026-09-07

Upstream main tip is `d263ca0a1` (2026-09-06). Its last four commits matter here:

| commit | what | effect on this box |
| --- | --- | --- |
| `23ebdfe87` #4484 (09-05) | router fp32 backward, bf16x9 matmuls | `enable_fp32_matmul_emulation_with_bf16x9` raises on any cc >= 10.0 GPU whose torch lacks BFX9 (pytorch#195301); the 5060 Ti trips it, neither venv's nightly has it, no override -- **main tip cannot run here** (mx3 shows `ASSERT-SEED-FAIL` on every cell) |
| `966befce6` #4493 | cudagraph capture of single-stage PP steps (`trainer.py`) | conflicts with the CP branch's copied CP-stack lines in `trainer.py` |
| `a423ddeb0` #4498, `d263ca0a1` | DeepSeek V4 MTP conversion; rl renderers | none |

`aecbb8199` (#4494, 09-05, after the EP merge of 09-03) is the last commit this box can run; rebases for
numerics use it and name it. Every run worktree on main also carries the local lift of `kda.py`'s SM100/SM103
guard (`local_hacks/kda_sm120_guard_lift.patch`; the bodies say so), or the first step dies with
`Attention Gym KDA requires Blackwell SM100/SM103; got CUDA capability (12, 0)`.

| branch | head | rebase onto `aecbb8199` | rebase onto tip `d263ca0a1` |
| --- | --- | --- | --- |
| `spmd_decl_review1` (4492) | `dbc60701d` | clean | clean |
| `tpsp_review3` (4499) | `c0ab32d07` | clean | clean |
| `cp_review5` = `k3_cp_text` (4313) | `61a73ca6c` | clean | conflict in `trainer.py` (#4493 vs the copied 4322/4449/4450 lines; drops when that stack lands) |
| `pp_review3` (4312) | `a3be242bf` | conflict in `kimi_k3/parallelize.py` at the first PP commit (base `6e2ac3dcd` predates #4446, the spmd_types enablement of `parallelize.py`; rerere holds an earlier resolution) | same |
| `pp_balance_review1`, `pp_offload_review1` | on `pp_review3` | same as PP | same |
| `ac_review2` | `a02b5e195` | clean | clean |
| `lora_review1` | `93f78b5ab` | clean | clean |
| `qb_release` (4412) | `a4658eefe` | clean -> `qb_review2` = `d0d75fa8b` on `aecbb8199`, QB tests 14/14, one-cache matrix below | clean but cannot run (the gate) |

QB: #4412 was waiting on the EP merge; EP merged 09-03. `qb_review2` carries the two QB commits on
`aecbb8199`; `matrix_scripts/qb_rebase_onecache.sh` runs the sign-step control and quantile balancing on
one compile cache (dp1 / dp2 / dp2 x ep2, partial_dtensor per cell since the declarations PR is not in main).
Results go to `PR_BODY_QB.md` when the run ends; syncing `k3_qb` and undrafting are the user's.

QB result (2026-09-07, `mx3_qb4_control_0907_040856` + `mx3_qb4_qb_0907_043737`, one cache, gym `b19162e`):
every one of the six filed rows reproduced to the digit on `d0d75fa8b` -- control dp1 12.52977 / 7.36833 /
2.91045, dp2 12.53137 / 7.25082 / 3.15411, dp2 x ep2 12.53146 / 7.13441 / 3.09174; quantile 12.52977 /
7.38270 / 3.00769, 12.53137 / 7.30862 / 3.20259, 12.53146 / 7.57599 / 3.16632. `PR_BODY_QB.md` names the
new tree (`DIFF_PR_BODY_QB_rebase_2026-09-07.diff`); syncing `k3_qb` to `qb_review2` and undrafting #4412
are the user's.

## Main tip runs here now (venv_bfx9, 2026-09-07)

`/workspace/venv_bfx9`: torch 2.15.0.dev20260906+cu130 and torchvision 0.30.0.dev20260906 from the cu130
nightly index (pytorch#195301 landed 09-02 06:00 UTC; nightlies from 09-03 carry it), torchtitan's
`requirements.txt`, cutlass-dsl 4.6.0, cuda-python 13.3.1, torchdata 0.11.0. `mx3.sh` takes `VENV=`.
`/tmp/wt_maintip` is `d263ca0a1` plus the guard lift. Declarations protocol (8192 / 256, seed 42,
partial_dtensor, gym `b19162e`), 10 steps:

| tree | venv | dp1 | dp2 x ep2 |
| --- | --- | --- | --- |
| `aecbb8199` (+QB, control flavor) | /venv/main (torch 2.14.0.dev20260802) | 12.52977 / 7.36833 / 2.91045 | 12.53146 / 7.13441 / 3.09174 |
| same | venv_bfx9 (torch 2.15.0.dev20260906) | 12.52977 / 7.36833 / 2.91045 | 12.53146 / 7.13441 / 3.09174 |
| main tip `d263ca0a1` | venv_bfx9 | 12.51887 / 7.11252 / 3.11301 | 12.52372 / 7.50218 / 3.21130 |

The torch upgrade is bitwise neutral on the same tree (`mx3_venvneutral_0907_061221`); the step-1 move of
about 1.1e-2 on main tip is #4484's `RouterLinear` (router forward output kept fp32, backward in fp32
through bf16x9 tensor-core matmuls), an upstream numerics change every K3 table after it will carry.
Scripts: `maintip_smoke.sh`, `venv_neutrality.sh`.


## 2026-09-08: QB moved to the main tip

`qb_review3` = `0e52d7b46`: the two QB commits plus a comment/counter fix on upstream/main `f6b9152e9` (#4505).
Clean cherry-pick; QB unit tests 14/14 in `venv_bfx9` after installing the tip's new `torch_remat` dependency
(pyproject pins a meta-pytorch/remat commit; the four runtime tests import it). The "last runnable commit
before the gate" framing above is obsolete: the tip runs in venv_bfx9. Matrix and load-balance evidence on
the tip: `QB_EVIDENCE_2026-09-08.md`.
