# TP/SP review branch reworked after the diff audit (2026-09-10)

The user's audit of `git diff ac10ca48f tp_sp_on_main` flagged two hunks: `torchtitan/distributed/utils.py` (+58/-17, the per-mesh `clip_grad_norm_` grouping over two commits) and `torchtitan/models/kimi_k2_7/vision_encoder.py` (+3/-6, the table retype under type checking). Everything else stays (`_local_head_split`: core's `local_qkv_head_split` is a GQA closure on `self.head_dim`, MLA's q and kv head widths differ; the KDA head-parallel declarations: parameters on the tp mesh, the kernel on local heads behind the local map).

## 1. The clip grouping: verified, then removed

`utils.py` reverted to main in a scratch worktree, tp2 SP on, both backends, three steps, one seed, against the branch head:

| cell | head (with grouping) | reverted (main's clip) |
| --- | --- | --- |
| tp2 spmd_types | 12.36934 / 10.17401 / 7.89447 | 12.36934 / 10.17401 / 7.89447 (bitwise) |
| tp2 partial_dtensor | 12.36934 / 10.19813 / 7.91091 | fails at the first clip: `aten.stack` sharding propagation over `_NormPartial` specs on `(fsdp=1, tp=2)` and the tower's fsdp-only mesh (`torch.nn.utils.get_total_norm`) |

As expected: under spmd_types every parameter, the tower's included, carries the tp axis (`annotate_replicated_parameters` seeds the layouts), one mesh, and the grouping was the single call it always was; under partial_dtensor the tower is FSDP-only. The choice between declaring the tower on the tp mesh under partial_dtensor and refusing partial_dtensor + TP: refusing. Placing the tower on the tp mesh is not a declaration: its parameters become DTensors on tp, its inputs and outputs must be wrapped and unwrapped around a flex-attention forward that DTensor does not dispatch. spmd_types is `ParallelismConfig.spmd_backend`'s default on main. So `parallelize_kimi_k3` refuses `tp_enabled and spmd_backend != "spmd_types"` with the mesh reason in the message, and core is untouched.

## 2. The K2.7 tables: their own commit at the bottom of the stack

`c0e1584df` (`k27_vision_tables_tp` on the fork): the two `mutate_type` calls name the dp axis only; on tp the table's declaration holds. It runs only under spmd_types type checking, so it cannot move a number. The K3 branch depends on it (the tower shares the helpers), so it sits at the bottom of the stack and the body names it as a separate small PR.

Evidence for the small PR: K2.5 debug model, dp2 x tp2, spmd_types with type checking, main against main + the commit. The cell needed an AdamW variant of the K2.5 debug config (the DistMuon container refuses TP), `activation-checkpoint:none` (type checking rejects selective AC with flex attention) and dp >= 2 (the qk-clip hook asks for the `loss` mesh). Result: main fails at the tables with `SpmdTypeError: mutate_type: expected current type PerMeshAxisLocalSpmdType.R on axis mesh_tp, got PerMeshAxisLocalSpmdType.I` (every rank), the check the K3 branch hit; with the commit the run gets past the tables and fails later in `kimi_k2_7/qk_clip.py` with `ValueError: QK clip scales do not match the MLA weight shape` (the qk-clip hook against a tp-sharded MLA weight), a Kimi K2.5 tensor-parallel problem of its own that no CI cell exercises. So the small PR's claim is exactly the first error, and the second is reported as a separate finding, not fixed here.

## 3. The rebuilt stack: `tp_sp_on_main` = `e6bed4bc9`, 10 commits on `ac10ca48f`

`c0e1584df` K2.7 tables; `4ebdbf23c` TP with SP on the declarations; `417714d4b` vision encoder under type checking at tp > 1; `fd6a46cde` KDA on capability 8.0+; `4dd722615` RouterGateLinear on local shards; `9b91f8e2e` the declarations' comments trimmed (what remained of the second clip commit); `5e1ccb608` routed_down under EP; `2492fb4c1` RouterGateLinear narrowing; `7232c1210` TP requires spmd_types; `e6bed4bc9` the b200 integration cell `kimi_k3_mm_tp2` (tp2, SP, spmd_types, type checking, 2 GPUs). `git diff ac10ca48f e6bed4bc9`: 11 files, no `torchtitan/distributed/utils.py`.

Checks on the stack: pinned pyrefly 0 errors on the touched files; `test_kimi_k3_sp_splice.py` + `test_integration_test_definitions.py` 14 passed (13 with the new cell registered); tp2 spmd_types 12.36934 / 10.17401 / 7.89447 (bitwise with the head); tp1 partial_dtensor 12.37043 / 10.23010 / 7.74434 on head and stack (bitwise); tp2 partial_dtensor refused with the message. Grep audit of the diff (logbook paths, measurements, prints): clean.

## 4. Body and sync

`PR_BODY_TP_SP_ON_4527.md`: the clip bullet replaced by "no core change" with the mesh reason, Limitations names the backend requirement, Tests drops the clipping callers and states `utils.py` untouched, the partial_dtensor TP rows are out of the tables (tp1 partial_dtensor rows stay as the reference), the PR stack names the K2.5 small PR, a CI/CD section names the new cell. Sync to `k3_tp_sp` (PR 4499) with `--force-with-lease tp_sp_on_main:k3_tp_sp` waits for the user's word.

## The A100 table and this rework

The A100 bf16 table (`TP_SP_ON_4500_2026-09-09.md`) ran the pre-rework branch: its spmd_types rows are unchanged by the rework (bitwise above), its partial_dtensor TP rows describe a configuration the reworked branch refuses and are kept as reference only. Nothing on the A100 had to be restarted.
