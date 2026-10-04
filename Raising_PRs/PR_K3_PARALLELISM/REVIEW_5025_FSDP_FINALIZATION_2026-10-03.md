# Shuhua 的 #5025（K3 PP stage 的 FSDP 手动 finalization）审阅，2026-10-03

用户 10-03："我看到Shuhua有这个针对我们pp的fix https://github.com/pytorch/torchtitan/pull/5025，我们之前合并的4312为什么没有捕捉到这个问题？？？明明是有FSDP PP混合测试的，这个fix是对的吗？有什么需要comment的地方吗？"

## #5025 是什么

- draft，作者 shuhuayu，10-02 17:21 UTC 开，head `fc6302123`，基于 main `bc86435b7`。暂时没有评论和 review。
- 改动 2 个文件 +37：
  - `kimi_k3/pipeline_parallel/stage.py` +4：`AttnResPipelineStage.forward_one_chunk` 里，在 `forward_maybe_with_nosync` 之前加 `if isinstance(self.submod, FSDPModule) and self.has_backward: self.submod.set_manual_backward_finalization(True)`。
  - `tests/unit_tests/cpu/test_kimi_k3_pp_stage.py` +33：用一个带 `set_manual_backward_finalization` 的 `nn.Linear` 桩替换 stage 模块里的 `FSDPModule` 名字，跑一次 forward，断言被调了一次 `True`。

## 结论：改法是对的

- 它和 torch main 基类 `PipelineStage.forward_one_chunk` 里新加的那一行逐字相同，位置也一样（`composite_kwargs` 之后、forward 之前），条件也一样（有 FSDP 且 `has_backward`，eval 不开）。
- 配对关系成立：关掉的那一半在基类里，K3 不覆写。
  - `perform_reduce_grad` / `start_gradient_reduction` 调 `finalize_backward(async_op=True)`；
  - `wait_for_gradient_reduction` 和 `clear_runtime_states` 把手动模式设回 `False`；
  - K3 的 `backward_one_chunk` 调 `super()`，所以反向一侧本来就跟着基类走。
- 不改的后果（读 torch main 源码得出，没有实际跑过）：
  - 没打开手动模式时，FSDP 走自动模式，每个 micro-batch 的反向结束时 root 回调就把这一轮 finalize 并清空状态。
  - 到 schedule 的 `REDUCE_GRAD`，`finalize_backward` 检查到不是手动模式，直接抛 `RuntimeError: ... finalize_backward requires manual backward finalization. Call set_manual_backward_finalization(True) before forward`。
  - 所以 K3 的 FSDP × PP 在新 torch 上第 1 步就会报错，不是悄悄算错。body 里写的 "could let FSDP finalize backward before the pipeline schedule reaches its gradient-reduction boundary" 是同一件事的前半句。

## 4312 为什么没抓到

- **合入时 torch 里还没有这个要求。** 时间线（UTC）：
  - 09-24 12:06：pytorch#196640（FSDP2 的 gradient accumulation finalization API）第一次合入，同时改了 `PipelineStage`：基类 forward 打开手动模式，`perform_reduce_grad` 改调 `finalize_backward()`。
  - 09-25 23:19：#196640 被 revert（"breaks internal builds"）。
  - 09-26 21:55：#4312 合入 titan main，正好在 revert 的窗口里。
  - 09-29 02:45：#196640 重新合入；09-29 17:47 #196725 把 `REDUCE_GRAD` 拆成 start / wait。
  - 在此之前，`perform_reduce_grad` 是直接对每个 param group 调 `post_backward()` 和 `_root_post_backward_final_callback()`，不需要任何模式，4312 的覆写在那时是完整的。
- **nightly 断了几天。** download.pytorch.org 的 cu130 nightly 列表：`dev20260926`、`dev20260928` 之后，下一个就是 `dev20261003`，09-29 到 10-02 都没有。所以：
  - 第一个带这个要求的 nightly 是今天的 `dev20261003`（`dev20260925` 也带第一次合入的版本，但那个已被 revert）。
  - #4731 在 10-02 05:58 的 `ciflow/b200`（run 36971319111）包含 4312 的 K3 FSDP × PP 格子，跑的是 `dev20260928`，所以通过了。
- **我们自己的验证也都在旧 torch 上。** 4312 的 GPU smoke 用 `torch 2.15.0.dev20260906`（`PP_REVIEW5_REBASE_2026-09-26.md`），之后 5060 上的 CI 格复测用 `dev20260928`。
- **B200 通道本身信号很弱：**
  - 它对 PR 是 opt-in（`ciflow/b200` 标签），4312 没有跑过（Actions 里没有 `ciflow/b200/4312` 的 run）；FSDP × PP 的 K3 格子只在 B200 那套里。
  - main 上的 B200 夜间任务从 09-23 起天天红（09-22 是最后一次绿），比 4312 合入还早，所以就算以后这个格子在 main 上开始报错，也会被淹没。10-03 那次（run 37083455002）6 分钟就失败，日志要 admin 权限，原因看不到。
- **CPU 单测不涉及 FSDP：** `test_kimi_k3_pp_stage.py` 和 `test_kimi_k3_pp_block_grads.py` 用 gloo，没有 `fully_shard`。
- **结构上的原因（我们的）：** `AttnResPipelineStage.forward_one_chunk` 是基类 forward 的完整拷贝（它要在 forward 前拼 block stack、forward 后打包 payload），所以基类 forward 以后再加的任何东西都不会自动继承。反向调 `super()`，所以只有 forward 这一侧出了问题。

## 这个覆写还漏了基类的什么（4312 合入时就已经漏了）

对照 torch main 的基类 `forward_one_chunk`，除了 #5025 补的这一行，K3 的覆写还跳过了：
- `register_forward_context` 注册的 forward context（基类用 `_call_with_forward_context` 包住 `forward_maybe_with_nosync`）；
- `TORCH_DISTRIBUTED_DEBUG=DETAIL` 时的 `_runtime_validate`：forward 输入和输出的 DTensor 元数据校验；
- 出错时带 args / kwargs 调试信息的 `RuntimeError` 包装，以及 debug 日志。

titan 现在没人用 forward context，DETAIL 也不是默认，所以目前不会出问题。09-25 的 torch（`3b6c63e7`）里这两项已经在基类里，4312 当时就没跟上。

要从根上解决，得让覆写不再拷贝基类 forward。可行的方向是只覆写 `_retrieve_recv_activations`（返回拼好的 stack）和 `forward_maybe_with_nosync`（forward 后打包 payload）。但 DETAIL 校验会拿 `(hidden, stack)` 去对照接收元数据 `(hidden, delta)`，所以要连拼装的位置一起改，不是小改动。这个由用户决定要不要做成 follow-up，不放进对 #5025 的评论里承诺。

## 对我们在开的 PR 的影响

- trial merge（`git merge-tree`）：
  - #4381 DEP：和 #5025 合并干净。`VisionDepPipelineStage.forward_one_chunk` 调 `super().forward_one_chunk`，会自动带上这一行。
  - #4963（PR A）：`stage.py` 自动合并，冲突只在 `model.py`（它基于较老的 main，本来就要 rebase）。
  - #4764 / #4765：`stage.py` 和 `test_kimi_k3_pp_stage.py` 冲突，它们重写了 `forward_one_chunk`。#5025 合入后 rebase 时要把这一行带上。
- 本地 torch：5060 的 `venv_0928` 和这台 Windows 机器都没有 `set_manual_backward_finalization`。#5025 合入后，K3 的 FSDP × PP 要 `dev20261003` 及以后才能跑；在旧 nightly 上会是 `AttributeError`。基类 stage 在新 torch 上本来也这样要求，titan 跟最新 nightly，不需要兼容处理，但我们自己的 GPU 机器要换 nightly。

## 要不要评论

不评论也可以：改法正确，CI 上的 B200 K3 PP 格子会在 `dev20261003` 及以后真正覆盖它。

建议发一条短评论，作者身份（4312 的作者）：确认改法、补上症状和时间线（方便 maintainer 理解为什么 4312 的格子是绿的），并提一句覆写还漏了 forward context 和 DETAIL 校验。最后那句是否保留由用户定；保留就意味着要做 follow-up。

--- PASTE BEGIN ---

Thanks, this is the right fix: it is the same line, under the same condition, that the base `PipelineStage.forward_one_chunk` gained when pytorch/pytorch#196640 relanded, and the matching reset stays in the base `wait_for_gradient_reduction` / `clear_runtime_states`, which `AttnResPipelineStage` does not override.

For the record on why the B200 `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4` cell stayed green: #4312 merged on 09-26 while #196640 was reverted, and the cu130 nightlies jump from `dev20260928` to `dev20261003`, so every run so far (including the `ciflow/b200` run on #4731) used a torch without manual finalization. From `dev20261003` on, without this change `REDUCE_GRAD` raises "finalize_backward requires manual backward finalization" at the end of step 1.

The root cause is on my side: `AttnResPipelineStage.forward_one_chunk` re-implements the base forward, so it also skips `register_forward_context` and the `TORCH_DISTRIBUTED_DEBUG=DETAIL` input/output validation. I will send a follow-up that makes the override reuse the base forward instead of copying it.

--- PASTE END ---
