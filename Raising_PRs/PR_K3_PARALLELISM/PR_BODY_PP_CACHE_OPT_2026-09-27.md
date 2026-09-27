# PR A body（pp cache optimize，4312 合并后单独开 PR），2026-09-27

## 状态（不粘贴）

- **分支：** review 分支 `pp_review_optimize` = `be9e2fa69`，main `f35966713`（含 4312 的 squash `e033f7517` 和 #4617）上的 1 个提交。09-27 由 `d445b2f7f` 的 4 个提交压成，同时删了 7 处私有函数和测试辅助类的 docstring、`_placeholder`，改回了 `cache.py` 的类 docstring，树差 +7/−19，见 `PP_REVIEW_OPTIMIZE_DIFF_AUDIT_2026-09-27.md` §5。还没有 PR 分支，也还没开 PR。
- **rebase：** 从 4312 的 head `ffdd169ef` 重放到 main。一处冲突在 `model.py` 的 `forward` 签名：#4617 把 `attention_masks` 的类型改成了 `HybridAttentionMetadata`，PR A 把 `block_residual_TND` 改成了 `blocks_TD: list[torch.Tensor]`，两边都保留。rebase 前后 PR 自己的 diff（+/- 行）完全相同。
- **检查：** pyflakes、flake8、ufmt 干净；4 个测试文件 68 个通过（`d445b2f7f` 上跑的；`be9e2fa69` 只删 docstring 和 `_placeholder`，ufmt 已过，单测要在 GPU 机器上重跑）；5060 组合格 10 步 rc=0，loss 8.10820 → 3.48003（本地 torch 兼容补丁，不在 diff 里）。
- **数字：** 5060 实测（`kit_pp_lowerbound_2026-09-26/results/s5_mem_base`、`s4_mem_pra`，base 是 `ffdd169ef`，与现在的 main 只差 #4617）；CPU 计数来自 `PR32_torchtitan_kimi_k3_attnres_recompute/stack_vs_list_2026-09-27.py`；H100 没测。
- **标题建议**（标题由用户改）：`[Kimi K3] PP rank store at the memory lower bound: one tensor per attention residual block, freed by the stage that brought it`
- **依赖：** #4656 在它之上 rebase（见 `PR32_torchtitan_kimi_k3_attnres_recompute/PLAN_4656_4780_2026-09-27.md`）；#4765、#4764 叠在它之上。
- **待定（09-27 用户指出 shuhuayu 那条是 eager AttnRes 的 stack）：** 如果列表载体改放进 #4656，粘贴区 Summary 第一条（block residual 改成列表）和 Design 第二段（21 对 6）要删掉，PR A 只剩 PP 的传输和存储。
- **torch issue：** 草稿在 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md`，用户在网页上开；开好后把号填进 Design 第三段的 `<torch issue link>`。

--- PR A body: PASTE BEGIN ---

## Summary

Kimi K3 pipeline ranks keep each attention residual block once, as its own tensor, and free it at the backward of the stage that brought it onto the rank; the model carries its blocks as a list, so no layer's checkpoint keeps a copy of the stack.

- The block residual (`kimi_k3/model.py`): a list of [T, D] blocks instead of a [T, N, D] stack. A block opening appends to the list; it used to concatenate a longer stack, and every earlier layer's checkpoint kept the shorter one it had been called with until backward. The aggregation stacks the list inside the call, so its arithmetic is unchanged.
- The pipeline stage and the rank store (`kimi_k3/pipeline_parallel/stage.py`, `cache.py`): one tensor per block on the wire. A received block is its own receive buffer and a committed block is the model's own tensor, so the store keeps references, not copies, and drops each block at the backward of the stage that brought it, its last reader on the rank.
- Receive buffers are allocated when the receive is posted; forward sends are waited at the stage's backward and input gradient sends at the first forward that proves they arrived.

## Design

The AttnRes paper stores each block exactly once across a rank's virtual stages, and the Kimi K3 report releases a micro-batch's blocks when it finishes, calling that the lower bound. A block's last reader on a rank, in backward order, is the stage that brought it there, so this PR releases it at that stage's backward, earlier than the micro-batch's end.

The list matters without pipeline parallelism too. With the stack, a 24 layer model in blocks of 4 keeps 21 [T, D] units of stack alive until backward, where the longest stack holds 6; with the list it keeps 6. That holds under full, selective and region AC alike, since each checkpoint takes the blocks as an input. With one tensor per block on the wire, a received block needs no row in a shared buffer, and nothing forces a copy when a hop forwards a block it did not make.

torch's pipelining runtime waits send works only after a step's last action, and a pending work keeps its tensor alive. It also keeps one receive buffer per micro-batch for every input and input gradient. The stage overrides both. These two behaviors hold for every pipeline model; they belong in torch (<torch issue link>), and the overrides go once torch offers them.

## Results

8 x RTX 5060 Ti, the released layout at a smaller width: 93 layers in blocks of 12, pp8 x vp4 (32 stages), dim 2048, seq 2048, 16 micro-batches, full AC. Peak allocated memory per PP rank in GiB at step 5 (steps 3 to 10 agree within 0.01):

| rank | main | this PR |
|---:|---:|---:|
| 0 | 10.42 | 5.52 |
| 1 | 10.68 | 6.02 |
| 2 | 10.68 | 6.00 |
| 3 | 10.78 | 6.80 |
| 4 | 12.00 | 6.11 |
| 5 | 11.48 | 6.15 |
| 6 | 11.47 | 6.11 |
| 7 | 11.20 | 7.11 |
| max | 12.00 | 7.11 |
| mean | 11.09 | 6.23 |

Loss and grad norm are identical on all 10 steps. The traced step takes 25.17 s on main and 24.85 s with this PR, with 7.44 s and 7.37 s of compute. H100 numbers go here once measured.

The 21 and 6 above are counted on CPU by a script that builds the two carriers at [T, D] = [64, 32] and counts the block storages alive after the forward.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py -q` (68 passed)
  - `test_kimi_k3_pp_block_grads.py`: four ranks on gloo under Interleaved1F1B and under 1F1B, cache on and off, blocks opening inside stages; every block gradient bitwise with one device; each block released by the stage that brought it; the input gradient wait points follow the receiver's use.
- The B200 cell `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4` on 8 x RTX 5060 Ti: 10 steps, loss 8.10820 to 3.48003 (no fixed seed, so only that it runs).

--- PASTE END ---
