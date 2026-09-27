# PR A body（pp cache optimize，4312 合并后单独开 PR），2026-09-27

## 状态（不粘贴）

- **09-27 夜（用户确认"对 做"）：** `pp_review_optimize` = `439bd2088`，叠在 #4656 的 `aa6d9fedc`（列表载体）上，不经过 #4780 的 checkpoint 提交；PR A 自己的 diff 和 `e8d0a4aec` 逐行相同；干净导出树上 Test plan 的命令 68 passed。旧 head `e8d0a4aec` 备份为 `backup/pp_review_optimize_pre_20260927`。粘贴区的 GPU 结果一律写 Pending（H100），5060 的 campaign s6 留在 logbook。
- **分支（09-27 晚，按 PLAN 的更正重叠）：** `pp_review_optimize` = `e8d0a4aec`，叠在 #4656 的 `attnres_review1` = `f14d681f4` 之上的 1 个提交，由 `be9e2fa69` 重放而来：`model.py`、`sharding.py` 取 #4656 的（PR A 原来在这两个文件里的改动就是列表载体，与 #4656 的逐行相同），PP 的 5 个文件（`stage.py`、`cache.py`、`__init__.py`、两个 PP 测试）与 `be9e2fa69` 完全相同；提交说明删掉了讲列表载体的第一段。
- **（以下为 `be9e2fa69` 时的记录）** review 分支 `pp_review_optimize` = `be9e2fa69`，main `f35966713`（含 4312 的 squash `e033f7517` 和 #4617）上的 1 个提交。09-27 由 `d445b2f7f` 的 4 个提交压成，同时删了 7 处私有函数和测试辅助类的 docstring、`_placeholder`，改回了 `cache.py` 的类 docstring，树差 +7/−19，见 `PP_REVIEW_OPTIMIZE_DIFF_AUDIT_2026-09-27.md` §5。还没有 PR 分支，也还没开 PR。
- **rebase：** 从 4312 的 head `ffdd169ef` 重放到 main。一处冲突在 `model.py` 的 `forward` 签名：#4617 把 `attention_masks` 的类型改成了 `HybridAttentionMetadata`，PR A 把 `block_residual_TND` 改成了 `blocks_TD: list[torch.Tensor]`，两边都保留。rebase 前后 PR 自己的 diff（+/- 行）完全相同。
- **检查：** pyflakes、flake8、ufmt 干净。`e8d0a4aec` 的干净导出树（不带本地 torch 兼容补丁）上，Test plan 的 4 个文件 68 个通过，加上 #4656 的 `test_kimi_k3_attention_residual.py` 一共 72 个通过。带补丁的 worktree 里 core 的 `test_pipeline_parallel.py` 有 5 个挂在 `KeyError: 'max_active_stages'`：补丁把本地 torch 不认的这个参数过滤掉了，和 PR 无关。
- **数字：** 5060 实测，campaign s6（`kit_pp_lowerbound_2026-09-26/results/s6_*`，索引见 `results/INDEX.md` 末节）：#4656 `f14d681f4` 对 PR A `e8d0a4aec`，同一条 cache 血缘。计时是第 8 步各 rank 窗口的平均（旧表的 25.17 / 24.85 也是这个口径），计算是各 rank 的平均。H100 没测。
- **标题建议**（标题由用户改）：`[Kimi K3] PP rank store at the memory lower bound: one tensor per attention residual block, freed by the stage that brought it`
- **依赖：** 叠在 #4656 之上（列表载体在 #4656 里）；#4765、#4764 叠在它之上（它们的 review 分支还要重叠）。
- **结果表已按 #4656 对 #4656 加 PR A 重测**（09-27 晚）：#4656 单独已经去掉了开 block 时的 stack 拷贝，所以旧表（main 对列表载体加 PP）换掉了。
- **torch issue：** 草稿在 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md`，用户在网页上开；开好后把号填进 Design 第三段的 `<torch issue link>`。

--- PR A body: PASTE BEGIN ---

## Summary

Kimi K3 pipeline ranks keep each attention residual block once, as its own tensor, and free it at the backward of the stage that brought it onto the rank.

- The pipeline stage and the rank store (`kimi_k3/pipeline_parallel/stage.py`, `cache.py`): one tensor per block on the wire. A received block is its own receive buffer and a committed block is the model's own tensor, so the store keeps references, not copies, and drops each block at the backward of the stage that brought it, its last reader on the rank.
- Receive buffers are allocated when the receive is posted; forward sends are waited at the stage's backward and input gradient sends at the first forward that proves they arrived.

## Design

The AttnRes paper stores each block exactly once across a rank's virtual stages, and the Kimi K3 report releases a micro-batch's blocks when it finishes, calling that the lower bound. A block's last reader on a rank, in backward order, is the stage that brought it there, so this PR releases it at that stage's backward, earlier than the micro-batch's end.

The model already carries its blocks as a list (#4656). With one tensor per block on the wire, a received block needs no row in a shared buffer, and nothing forces a copy when a hop forwards a block it did not make.

torch's pipelining runtime waits send works only after a step's last action, and a pending work keeps its tensor alive. It also keeps one receive buffer per micro-batch for every input and input gradient. The stage overrides both. These two behaviors hold for every pipeline model; they belong in torch (<torch issue link>), and the overrides go once torch offers them.

## Relation to #4656

Stacked on #4656, which carries the model's blocks as a list; its commit shows here until it merges.

## Results

Pending (H100).

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py -q` (68 passed)
  - `test_kimi_k3_pp_block_grads.py`: four ranks on gloo under Interleaved1F1B and under 1F1B, cache on and off, blocks opening inside stages; every block gradient bitwise with one device; each block released by the stage that brought it; the input gradient wait points follow the receiver's use.
- The B200 cell `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4`: pending (H100).

--- PASTE END ---
