# PR 4381 body v5（draft PR，`k3_pp_mm`），2026-09-28：按 K2.5 DEP 原文和 K3 §5.2.3 重写

## 状态（不粘贴）

- **09-29 由 GPU 这边接手（用户："你直接接管所有改动"）：** `k3_pp_mm` = `dep_review1` = `a03f74981`，旧 head `3c461fdf1` 备份在 `backup/k3_pp_mm_pre_takeover_20260929`。只改了测试：GPU 上第 1 步之后的所有梯度都用放宽的容差（`rtol=1e-5, atol=1e-4`），这 3 行并进了引入测试的提交（`1b6d500e3`），三个提交的结构不变，没有 trailer。5060 上的结果：Test plan 第一条 15 passed，CPU 48 passed，GPU 测试连跑两次都是 2 passed。传输和副本的代码和 `3c461fdf1` 相同，所以那次的 B200 格子和逐位一致的结论照样成立。粘贴区不用改，GPU 结果仍写 pending。
- **09-29 复测 `3c461fdf1`（5060）：** 副本随机数已修好，K2.5、bubble 都和 DEP 关完全逐位一致，B200 格子跑完 10 步。GPU 测试的两个用例都卡在第 2 步 `0.embed` 梯度的 fp32 默认容差上（差 1.53e-5，来自第 1 步塔梯度的求和顺序），需要 CPU 那边把放宽的容差用到第 1 步之后的所有梯度上。见 `DEP_GPU_CHECK_2026-09-29.md` 末节。
- **09-29 GPU 验证（5060，`55e4274c4`）：** 死锁已修好，B200 格子跑完 10 步。副本初始化多用了随机数，改变了模型的初始权重，所以还不能拿 DEP 开和 DEP 关比；用 `fork_rng` 探针对齐以后，两种模式都和 DEP 关逐位一致。bubble 模式的 GPU 测试有一处容差没过。详见 `DEP_GPU_CHECK_2026-09-29.md`。粘贴区的 GPU 结果仍写 pending。
- **09-29 下午，两处补丁（GPU 那边在 `DEP_GPU_CHECK_2026-09-29.md` 里提的）：** 分支现在是 `3c461fdf1`，`k3_pp_mm` 和 `dep_review1` 已从 `55e4274c4` force-with-lease 推送到这里。
  - 副本的初始化包进了 `torch.random.fork_rng`（CUDA 时连同本卡），不再推进全局随机数。同一种子下，DEP 开和 DEP 关的初始权重现在应当相同。新增 CPU 测试：建副本前后抽到的随机数逐位相同；去掉 `fork_rng` 这个测试就失败，已验证。
  - GPU 测试：学习率改成 1e-3，玩具模型的数值不再涨到 1e17；塔梯度的容差按求和顺序放宽到相对 1e-5；每个 rank 先收集自己的失败，所有 rank 汇总一次后再一起失败，失败的断言内容会直接显示，不再让其他 rank 卡在收尾。
  - 本机（harness）24 个通过，GPU 测试在本机跳过。
- **09-29 传输修复：** 分支现在是 `55e4274c4`，在 main `5dc97a3e7`（#4905 把 ParallelDims 改名为 ParallelismContext）上，一共 3 个提交。PR 分支 `k3_pp_mm` 和 review 分支 `dep_review1` 都已从 `bb3e38d4a` force-with-lease 推送到这里。
  - 死锁修法：不再在步首挂出 receive。每一对 send 和 receive，两边都在调度的同一个槽边界上挂出，这个边界两边都能在不依赖对方之后工作的情况下走到；receive 在使用前等，send 在步末等。
  - K2.5 模式：特征在预编码后于步首交换，梯度在调度结束后于步尾交换。
  - bubble 模式：特征取发送方空闲段结束的边界，梯度取执行方空闲段开始的边界。
  - 规划器测试里加了一个最坏情况模型：每个 kernel 都要等本 rank 所有未配对的操作。torch 调度自己在这个模型下能跑完，新 plan 也能跑完；只要把 receive 挪到步首就会卡住。新增 NCCL 下的 GPU 测试 `tests/unit_tests/gpu/test_kimi_k3_vision_dep.py`（4 卡），里面的塔带 GELU，它的 kernel 在 step 进行中才第一次加载。
  - 本机（harness）23 个通过，GPU 测试在本机跳过。
- **09-28 夜 GPU 验证（5060，`bb3e38d4a`）：** B200 格子第 1 步死锁，见 `OVERNIGHT_RESULTS_2026-09-28.md` 的 T5 节。修好以后按 `t5b.sh` 补测，在这之前粘贴区的 GPU 结果保持 pending。
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

Planned work runs after the action it is anchored to. When that action sends to another rank, the stage issues the send first and returns no ops to the runtime, so the send does not wait for the vision work. Features and gradients travel on a process group created per pipeline group, and both ends of a transfer post it at one slot boundary of the schedule, a point each reaches without the other's later work: the prologue's features at the start of the step, the epilogue's gradients at its end, a feature encoded in an idle slot when its rank's next action starts, and a gradient when the idle slot of the rank that uses it begins. A posted send or receive therefore never waits on work its own rank has yet to do; receives are waited where their data is used and sends at the end of the step. Each rank sums its copy's gradients in fp32; at step end they are reduced to stage 0, all-reduced over data parallel, gathered over tensor parallel and added into the tower's sharded gradients, before the gradient norm.

`pipeline_kimi_k3` returns the schedule wrapped so that its `step` runs the vision phases around the core step; the engine's pipeline step body has no model hook for them. The runtime lives in the model's pipeline package, next to the attention residual stage it extends.

## Relation to earlier revisions of this PR

The earlier revisions gave the tower a pipeline stage of its own on the first rank and placed its encodes in that rank's idle slots. That kept the tower inside the pipeline split, on one rank, with its autograd graph alive until its backward; this revision replaces it.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_dep_plan.py tests/unit_tests/cpu/test_kimi_k3_vision_dep.py -q` (15 passed).
  - `test_kimi_k3_dep_plan.py`: on the Interleaved1F1B action order at pp2 x vp4, pp4 x vp2 and pp8 x vp4, every encode in an idle slot finishes before its consumer, every backward starts after its gradient arrives, planned work sits in idle slots without overlap, both ends of each transfer post it at one slot boundary and every pair of ranks posts its transfers in the same order, a transfer leaves after its data exists and arrives before its use, and with a cheap encode every micro-batch after the upfront ones is hidden. Replaying the schedule's own sends and receives with every kernel waiting for its rank's unmatched transfers, no rank is left stuck, with the process in either placement.
  - `test_kimi_k3_vision_dep.py`: four ranks on gloo, pp4 x vp2, eight micro-batches with two text only, with the work before and after the schedule and with it in idle slots: the step-1 loss and every gradient are bitwise with one device, the step-2 loss and text gradients are bitwise and the tower gradients agree to fp32 summation order; a frozen tower gets no gradient; eval between steps matches one device. At pp2 x tp2, each copy receives its tensor-parallel shard of the tower and the tower's gradient is the sum of every rank's shards. Building the tower's copy leaves the seeded random stream where it was, so the model initializes the same with the process on or off.
- `pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q`, the same four ranks under NCCL with a tower whose kernels first load inside the step, in both placements: pending.
- The B200 cell `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4_vision_dep`: pending.
- DEP on and off on one warm compile cache, and `bubble` on and off, loss, gradients and step time: pending (H100).

--- PASTE END ---
