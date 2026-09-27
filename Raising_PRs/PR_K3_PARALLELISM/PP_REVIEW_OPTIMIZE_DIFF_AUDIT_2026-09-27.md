# pp_review_optimize（PR A）审 diff 和 body（2026-09-27，4312 合并后）

范围：`git diff f35966713 d445b2f7f`。`pp_review_optimize` 已由 GPU 那边 rebase 到 upstream main `f35966713`，有 4 个提交，改了 7 个文件，+564/−314。body 部分审的是 `PR_BODY_4765_2026-09-27.md` 里属于 PR A 的内容：Summary 的前两条、Design 的前三段。PR A 目前没有单独的 body。

跑过的检查：
- 按 pre-commit 的版本（black 22.12.0、usort 1.0.5）跑 ufmt，7 个文件都已格式化。
- 没有超过 120 列的行。
- 测试类都在 `if __name__ == "__main__":` 之前。09-26 审计 §2 的前两项已修。
- 这台机器上没有跑 CPU 单测：kimi_k3 在 Windows 上 import 不了。

## 1. DEP 和 PR A 的关系

- **互不依赖。** `dep_review1` 是 `456fa195f`，`pp_review_optimize` 是 `d445b2f7f`，两者都直接叠在 main `f35966713` 上，谁也不包含谁。
- **放在一起会有两处文本冲突**（`git merge-tree` 实测），都是两边各加一段，合并时两边都保留即可：
  - `model.py` 的 `KimiK3Model.forward` 签名：DEP 加了 `vision_embeds` 参数，PR A 把返回类型改成了 `list[torch.Tensor]`。
  - `pipeline_parallel/__init__.py`：import 一行冲突；另一处是 PR A 给 `set_routing` 加了参数，DEP 在它后面接了 `install_vision_dep`。
- **语义上能组合：**
  - `VisionDepPipelineStage` 继承 `AttnResPipelineStage`，只通过 `super()` 包 `forward_one_chunk`、`backward_one_chunk` 和 `backward_weight_one_chunk`。PR A 改的都是这三个方法内部和别的方法。
  - DEP 推迟的是 tower 所在的第一个 stage 的反向。第一个 stage 不发输入梯度，所以 PR A 的输入梯度等待点不受影响。
- **还需要一次 GPU 验证：** 两者都合进来以后，要在 GPU 上跑一次 vit_dep 格子。
- `k3_pp_mm`（DEP 的 PR 分支）还是旧 base 的 `232834a4d`，里面带着 4312 的提交。同步要等用户同意。

## 2. diff：行为

逐项推了一遍，没有发现错误：
- block 列表：模型每开一个 block 只往列表里 append，不再复制整个 stack。聚合在调用内部才 stack，所以算术和原来一样。
- store 保存的是 `detach()` 之后的引用，不做拷贝；每个 block 在把它带到这个 rank 的那个 stage 的反向时释放。本 rank 最后一个反向（也就是 rank 上第一个 stage 的反向）检查 store 里没有残留。
- 接收缓冲在 post receive 时才分配，读完就换回 placeholder。deposit 原地加到梯度接收缓冲上，前提是这块缓冲只属于当前这个 micro-batch，代码里有一行注释写明了这个约束。
- 两种 send 的等待时机：
  - 前向 send 在本 stage 对同一 micro-batch 做反向时 wait；
  - 输入梯度 send 在"证明点"wait：本 rank 上第一个前向，它的输入是接收方在消费完这份梯度之后才产出的。
  - 找不到证明点时，这个 send 交还给 core，按原来的方式在 step 末 wait。
- `sharding.py` 删掉了多模态路径上 layer 0 对 `block_residual_TND` 的边界 redistribute。现在列表从空开始，layer 0 放进去的第一个 block 就是已经 redistribute 过的 `x_TD`，所以不再需要。

## 3. diff：进 PR 前要改

1. **提交历史过时。**
   - `678447260` 描述的是 [N, T, D] 行缓冲和 [K, T, D] 线上格式，这两样都被 `d445b2f7f` 删掉了。
   - `5dda7b67b` 说单 stage schedule 下 payload 会复制一份，现在的代码里没有这份复制。
   - 建议压成两个提交：一个是 block 列表加按 block 传输和释放；另一个是 torch 通用行为的覆盖（按需分配接收缓冲、send 的等待点）。
2. **torch 私有接口。**
   - 新用到的私有接口：`_batch_p2p`、`_ComputationType`、`_make_tensor_from_meta`、`_PipelineScheduleRuntime`。
   - 新覆盖的私有方法：`_setup_forward_recv_info`、`_setup_backward_recv_info`、`backward_maybe_with_nosync`。
   - body 里说这些"belong in torch"，但计划 §6 的第一步（先提 torch issue）还没做。按 CLAUDE.md 的抽象规则，body 要写明缺的是哪个 seam，并附上 issue 链接。
3. **私有辅助函数的 docstring**（#4577 的标准是不写）：`_outgoing_blocks`、`_grad_send_wait_points`（两行而且是硬换行）、`_GradSendWaits`、`_brought`、`_assemble`、`_route_input_grads`。main 上原来的 `_assemble_stack` 等也有 docstring，所以删不删要统一：要删就一起删。
4. **无关的改动：** `cache.py` 的类 docstring 从 "Blocks a rank holds" 改成了 "The blocks a rank holds"，改回原样。
5. **`_placeholder` 里的 `torch.bfloat16` 回退分支走不到：** 调用方传进来的 `info.buffer` 都是 tensor。直接用 `like.dtype`，把回退删掉。

## 4. body（`PR_BODY_4765_2026-09-27.md` 里 PR A 的部分）

- 4312 已经合并，以下几处过时：
  - 标题里的 "stack on #4312"；
  - "Relation to #4312 and #4764" 一节的 "Stacked on #4312"；
  - 状态栏的 base `ffdd169ef`；
  - 结果表的 "#4312" 列应改叫 main。
- 表里的数字是在 4312 的 `ffdd169ef` 上测的，现在 base 换成了 main `f35966713`（中间合进来了 CP #4617 等）。按数值表规则，要在新 base 上、共用一份暖缓存重测以后才能沿用这些数字。在重测之前，这些数字先不写成本 PR 的数字。
- PR A 要不要单独开 PR 还没定（计划 §6 的决定 1）。如果单独开，Summary 的前两条和 Design 的前三段移到 PR A 的 body 里，#4765 只留 storage 的部分。

## 5. 处理结果（2026-09-27，用户："全改了，推"）

- **`pp_review_optimize`：** 从 `d445b2f7f` 变为 `be9e2fa69`，force-with-lease 推送。
  - 与 `d445b2f7f` 的树差只有 +7/−19 行：6 个私有函数的 docstring 删掉；测试里 `_LoggedStore` 的 docstring 删掉；`cache.py` 的类 docstring 改回原样；删掉 `_placeholder`，改成 `info.buffer.new_empty(0)`，forward 那两处加了 `is not None` 判断。
  - 4 个提交压成 1 个，提交说明重写，只描述现在的代码。原计划压成两个，没有这样做：`stage.py` 里两部分交织在一起，这台机器跑不了 kimi_k3 的测试，拆出来的中间状态没法验证；upstream 也是 squash 合并。
  - ufmt 通过（black 22.12.0）。CPU 单测在 Windows 上 import 不了，没有跑，等 GPU 机器跑 `test_kimi_k3_pp_block_grads.py` 和 `test_kimi_k3_pp_stage.py`。
- **torch 私有接口（§3 第 2 条）：** issue 草稿在 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md`，需要用户在网页上开；#4765 正文里留了 `<torch issue link>` 待填。
- **body：** `PR_BODY_4765_2026-09-27.md` 去掉了 stack on #4312，结果表的列名改成 main，状态栏写明表里的数字要在新 base 上重测之后才能粘贴。
- **还没做：** `pp_offload_review1`、`pp_balance_review1` 还要 rebase 到 `be9e2fa69` 上，交给 GPU 那边。
