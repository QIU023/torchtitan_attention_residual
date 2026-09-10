# TP/SP rework handoff (2026-09-10), updated after the rebuild

The user's audit items and what the rebuilt branch does about each. Branch: `k3_tp_sp` = `tp_sp_on_main` (PR 4499), the rebuilt stack on main `ac10ca48f`; the pre-rebuild head `ce10692c1` and the reviewer's local commit `8a9c129d7` (pyrefly suppressions removed, four CP TODOs) are gone from the fork; `8a9c129d7` never reached origin and is not in this clone, so its two items were redone from scratch (below) and its TODO text has to come from the reviewer.

| item | result |
| --- | --- |
| 1. core `distributed/utils.py` (+58/-17): verify, likely revert | Verified first (`TP_SP_REWORK_2026-09-10.md` §1): with `utils.py` back to main, tp2 spmd_types is bitwise the branch (12.36934 / 10.17401 / 7.89447), tp2 partial_dtensor fails at the first clip on the mixed meshes. Reverted; the choice between declaring the tower on the tp mesh under partial_dtensor and refusing partial_dtensor + TP went to refusing (`7232c1210`): the tower's inputs and outputs would have to be wrapped as DTensors around a flex-attention forward, and spmd_types is the default backend. |
| 2. `kimi_k2_7/vision_encoder.py` (+3/-6): its own small PR | Its own commit at the bottom of the stack, `c0e1584df`, pushed as `k27_vision_tables_tp`; body `PR_BODY_K27_TABLES.md`. Evidence: K2.5 dp2 x tp2 with type checking fails on main at the tables with the retype error and gets past them with the commit (then hits K2.5's own qk-clip TP problem, reported separately). |
| 3. K3 TP CI cell | `kimi_k3_mm_tp2` in the b200 suite (`e6bed4bc9`): tp2, SP on, spmd_types, type checking, 2 GPUs. |
| 4. `_local_head_split`, KDA head-parallel declarations | Kept. |
| 5. Verification | Pinned pyrefly 0.45.1: 0 errors on the touched files. Numerics: the spmd_types cells bitwise before and after; the 10-step tables (this box, 4527-based twin) and the A100 100-step spmd_types rows in the body; partial_dtensor TP rows removed. `git diff ac10ca48f <head>`: 11 files, `utils.py` absent; grep audit clean. |
| 6. Body | Reworked (`PR_BODY_TP_SP_ON_4527.md`): no "both backends", the refusal and its reason in Summary and Limitations, tp=1 partial_dtensor control rows kept with a note, the 100-step A100 section, Tests with the CI cell and the K2.5 evidence, `utils.py` untouched. |
| 7. Sync | Done on the user's word: `k3_tp_sp` = `tp_sp_on_main`. |
| reviewer's `8a9c129d7` | The pyrefly suppressions: the one the branch itself adds (`preprocess_inputs`, `bad-override`) is not needed by the pinned checker and is dropped in the top commit; the other suppressions in the touched files are main's own. The four CP TODOs: text needed from the reviewer, not reconstructed. |

Head after the top commit: see `git log -1 origin/k3_tp_sp`.
