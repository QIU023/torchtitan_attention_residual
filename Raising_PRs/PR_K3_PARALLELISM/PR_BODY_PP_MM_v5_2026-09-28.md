# PR 4381 body v5（draft PR，`k3_pp_mm`），2026-09-28：按 K2.5 DEP 原文和 K3 §5.2.3 重写

## 状态（不粘贴）

- **09-28 夜 GPU 验证（5060）：** B200 格子第 1 步死锁（步首挂出的 NCCL irecv 碰上塔的 kernel 第一次加载），数值、trace、显存都没测成；详见 `OVERNIGHT_RESULTS_2026-09-28.md` 的 T5 节。修好之前，粘贴区里的 B200 格子和 GPU 结果保持 pending。
- **分支：** PR 分支 `k3_pp_mm` 和 review 分支 `dep_review1` 都是 `bb3e38d4a`，即 main `f35966713` 上的 3 个提交。09-28 按你的话（"直接推送，这是draft PR"）force-with-lease 推送，替换了旧实现 `31f372593`。旧 head 没有另建备份分支，通过 PR 的 force-push 记录和提交哈希仍能找到。
- **重写的原因：** 旧实现把塔放成 rank 0 上单独的一个 PP stage，不是 DEP，见 `DEP_VS_REPORT_AUDIT_2026-09-27.md`。旧的 4 个提交全部弃用，optimizer 那个提交（只有塔的 stage 匹配不到 Muon 矩阵）也不再需要，因为不存在只有塔的 stage 了。
- **本机验证：**
  - 21 个测试通过：K3 PP 的旧测试，加上新的规划器测试（8 个）和 gloo 测试（4 个，其中 1 个专测 TP 分片的同步和梯度回写）。
  - 09-28 按用户指出修正：副本原先没有并行化，TP > 1 时各 TP rank 都算完整的塔；现在副本用模型自己的 TP 方案（#4499 的 ViT TP），同一次编码由 TP 组分担。执行点也从副流改成"锚点动作的 send 发出之后在主流上跑"，因为副流上的视觉 TP 集合通信会和文本的集合通信在同一个通信器上排队。
  - 这台 Windows 机器 import 不了 kimi_k3 的模型（缺 CuTeDSL），torch 也只有 2.13，所以测试是在 scratchpad 的 harness 下跑的：对模型包打桩，并补上 nightly 的 `step(arg_mbs=...)` 接口。harness 不进任何提交。GPU 机器上要用正常方式重跑一遍。
- **GPU 上待验证：** 见审计文档 §8。GPU 的数字出来之前，粘贴区的结果一律写 pending。
- **标题建议**（标题由你改）：`[Kimi K3] Decoupled encoder process under pipeline parallelism, with the vision work in pipeline bubbles`

--- PR 4381 body v5: PASTE BEGIN ---

## Summary

Kimi K3 trains its vision tower with the decoupled encoder process (DEP) of Kimi K2.5, and runs the tower's work in pipeline bubbles (report sec 5.2.3); this PR implements both for the Kimi K3 pipeline.

- `kimi_k3/pipeline_parallel/dep_plan.py`: `plan_dep`, the rank that encodes and the rank that backpropagates each micro-batch, and the point in the schedule where each runs.
- `kimi_k3/pipeline_parallel/vision_dep.py`: `VisionDep`, one rank's copy of the tower with the transport of features and gradients; `VisionDepPipelineStage`, which feeds stage 0 and runs the planned work after each action; `VisionDepSchedule`, which runs the vision phases around each step.
- `kimi_k3/model.py`: a `vision_dep` config (`enabled`, `bubble`, `bubble_cost_ratio`) and a `vision_embeds` argument on `forward`.
- `torchtitan_recipes/tests/b200.py`: `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep`.

## Design

The text split is core's, unchanged. The tower stays on stage 0 as the owner of its parameters, so the optimizer, the checkpoint, FSDP and the gradient norm see it as before, but it no longer runs in training. Every pipeline rank holds a copy of the tower, parallelized with the model's own tensor-parallel plan (#4499) and refreshed from stage 0's parameters at the start of each step; the copies are not wrapped by FSDP.

A micro-batch is the unit of work and its patch count is the load. Every pipeline rank already holds the pixel values of every micro-batch of its data-parallel replica, so the features are the only thing that moves to stage 0, and the gradient at the features the only thing that moves back. An encode runs under `no_grad` and keeps only its output; the rank that runs a micro-batch's backward recomputes the tower and backpropagates. Stage 0 splices the features in as a leaf and reads their gradient through a tensor hook, because the stage backward clears input gradients.

Without `bubble`, every encode runs before the schedule and every backward after it, balanced across the ranks by patch count, which is K2.5's form. With `bubble`, the plan reads every rank's `pipeline_order`: the first pipeline-degree micro-batches stage 0 consumes are encoded before the schedule, and the others in an idle slot of any rank ahead of the forward that reads them; a backward runs in an idle slot after stage 0's backward of the micro-batch; what fits no idle slot joins the balanced prologue or epilogue. Every rank derives the same plan from the action order and `grid_thw`, so no metadata is exchanged.

Planned work runs after the action it is anchored to. When that action sends to another rank, the stage issues the send first and returns no ops to the runtime, so the send does not wait for the vision work. Features and gradients travel on two process groups created per pipeline group, so their order is independent of the schedule's sends and receives. Each rank sums its copy's gradients in fp32; at step end they are reduced to stage 0, all-reduced over data parallel, gathered over tensor parallel and added into the tower's sharded gradients, before the gradient norm.

`pipeline_kimi_k3` returns the schedule wrapped so that its `step` runs the vision phases around the core step; the engine's pipeline step body has no model hook for them. The runtime lives in the model's pipeline package, next to the attention residual stage it extends.

## Relation to earlier revisions of this PR

The earlier revisions gave the tower a pipeline stage of its own on the first rank and placed its encodes in that rank's idle slots. That kept the tower inside the pipeline split, on one rank, with its autograd graph alive until its backward; this revision replaces it.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_dep_plan.py tests/unit_tests/cpu/test_kimi_k3_vision_dep.py -q` (12 passed).
  - `test_kimi_k3_dep_plan.py`: on the Interleaved1F1B action order at pp4 x vp2 and pp8 x vp4, every encode in an idle slot finishes before its consumer, every backward starts after its gradient arrives, planned work sits in idle slots without overlap, each rank sends its features in the order stage 0 receives them, and with a cheap encode every micro-batch after the upfront ones is hidden.
  - `test_kimi_k3_vision_dep.py`: four ranks on gloo, pp4 x vp2, eight micro-batches with two text only, with the work before and after the schedule and with it in idle slots: the step-1 loss and every gradient are bitwise with one device, the step-2 loss and text gradients are bitwise and the tower gradients agree to fp32 summation order; a frozen tower gets no gradient; eval between steps matches one device. At pp2 x tp2, each copy receives its tensor-parallel shard of the tower and the tower's gradient is the sum of every rank's shards.
- The B200 cell `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4_vision_dep`: pending.
- DEP on and off on one warm compile cache, and `bubble` on and off, loss, gradients and step time: pending (H100).

--- PASTE END ---
