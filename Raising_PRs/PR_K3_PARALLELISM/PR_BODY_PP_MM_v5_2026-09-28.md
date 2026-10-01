# PR 4381 body v5（draft PR，`k3_pp_mm`），2026-09-28：按 K2.5 DEP 原文和 K3 §5.2.3 重写

## 状态（不粘贴）

- **10-01：** 规划器偏离报告原文（装不下就退回前面或后面），修正方案在 `DEP_FIX_PLAN_2026-10-01.md`，等用户确认。修正后粘贴区的 Design 段（"what fits no idle slot joins the balanced prologue or epilogue"、`bubble_cost_ratio` 那句）要改写。
- **09-30 H100（115.124.123.240，`d27839459`）：** 粘贴区只改了一处：GPU 测试那条从 pending 改成 "3 passed on 4 H100s"（同一台机器上 CPU 那条也是 18 passed）。B200 格子要 8 卡，仍是 pending。"DEP on and off ... loss, gradients and step time" 那条仍写 pending：数值对照里 DEP 开和关第 1 步 loss 不是逐位一致（8.12804 对 8.12706），还在定位（`DEP_H100_2026-09-30.md` 数值对照一节），定位前不进粘贴区。步时和效率（pp4 × vpp4，dim 6144 的 debug 模型加 K3 塔，1008 px）：K2.5 比 DEP 关快 15% 到 29%，大头是 DEP 关时 stage 0 给每个 micro-batch 跑塔（纯文本的用假图）；相对不带塔的纯文本模型，K2.5 的效率 96 : 4 时 85% 到 91%，90 : 10 时 84%，84 : 16 时 74% 到 75%；bubble 对 K2.5 在 ±2% 以内，实测填充率 0% 到 1%（一次编码是 3.6 个 stage 前向，放不进 pp4 × vpp4 的空闲段）。要不要在 body 里放这些数字、怎么放，等第 1 步的差异定位后再和你商量。
- **09-30 复查（CPU 这边，用户："检查DEP和MoonEP body和diff，现在这两个可以去H100跑了吗？"）：** `d27839459` 把梯度的挂出边界改成 `max(run.start, ready[m])`（`dep_plan.py:342`），发送方在 B0.m 之后挂出，执行方的边界落在自己的空闲段里、按动作顺序等同于段首，两端仍在同一个槽边界，每对 rank 的次序不变。粘贴区 Design 里传输那句还写着"梯度在执行方空闲段开始时换 rank"，已改成两者取较晚。
- **09-30（用户："DEP没藏到反向？马上debug修复 并且最终汇报气泡填充率"）：** review 分支 `dep_review1` 和 PR 分支 `k3_pp_mm` 都推到 `d27839459`（force-with-lease，旧 head `a03f74981` 在 `backup/k3_pp_mm_pre_20260930`），upstream `refs/pull/4381/head` 已是它。整条线换到 main `46ec3f232` 上（`a00bf6e03`、`22fe32202`、`f85bbb04a` 和原来逐字相同），加一个修复提交 `d27839459`。
  - 原因一（bug）：规划器跳过所有"在梯度就绪之前开始"的空闲段，哪怕它后半段放得下。修复：梯度在"就绪时刻"和"空闲段起点"中较晚的那个边界换 rank，反向可以用这段的后半截。
  - 原因二（模型）：空闲段按格数算，前向、反向都算 1 格，而且把所有 rank 同时空着的格也算成气泡。pp2 × vp4 冷却段那些空格，另一个 rank 其实也空着，真实时间里长度为 0。改成一格的时长等于这一格里最长的动作（前向 1，反向 2），`bubble_cost_ratio` 的单位改成一次文本 stage 前向。
  - 结论：pp2 × vp4 的真实气泡只有一步的 2% 到 8%，梯度就绪之后几乎没有空闲，所以 5060 上那个布局本来就藏不住反向；深 PP 修复后能多放进去（比如 pp8 × vp4 M16、r = 1 时 11 → 14 次，M32、r = 0.5 时 27 → 30 次）。
  - 检查：CPU 33 passed（含 B200 格子的注册）；5060 上 GPU 单测 3 passed（NCCL，含新加的"反向在空闲段里等晚到的梯度"）；pre-commit 干净。pyrefly 比 main 多 10 个错误，都是分支原有的（修复前后错误集合相同），转正式前要修。
  - 粘贴区改了：Design 里 bubble 那段加一句"用空闲段后半截、按动作时间量"；Test plan 通过数 15 → 18，加上新测试。GPU 那行仍是 pending。
  - 填充率（模型值）见 `OVERNIGHT_RESULTS_2026-09-29.md` 最后一节。
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

Without `bubble`, every encode runs before the schedule and every backward after it, balanced across the ranks by patch count, which is K2.5's form. With `bubble`, the plan reads every rank's `pipeline_order`: the first pipeline-degree micro-batches stage 0 consumes are encoded before the schedule, and the others in an idle slot of any rank ahead of the forward that reads them; a backward runs in an idle slot after stage 0's backward of the micro-batch, including the rest of an idle run that began before that backward; what fits no idle slot joins the balanced prologue or epilogue. Idle time is measured in action time, a slot lasting as long as its longest action with a forward 1 and a backward 2, and `bubble_cost_ratio` is an encode in units of a text-stage forward. Every rank derives the same plan from the action order and `grid_thw`, so no metadata is exchanged.

Planned work runs after the action it is anchored to. When that action sends to another rank, the stage issues the send first and returns no ops to the runtime, so the send does not wait for the vision work. Features and gradients travel on a process group created per pipeline group, and both ends of a transfer post it at one slot boundary of the schedule, a point each reaches without the other's later work: the prologue's features at the start of the step, the epilogue's gradients at its end, a feature encoded in an idle slot when its rank's next action starts, and a gradient at the later of stage 0's backward of its micro-batch and the start of the idle slot of the rank that uses it. A posted send or receive therefore never waits on work its own rank has yet to do; receives are waited where their data is used and sends at the end of the step. Each rank sums its copy's gradients in fp32; at step end they are reduced to stage 0, all-reduced over data parallel, gathered over tensor parallel and added into the tower's sharded gradients, before the gradient norm.

`pipeline_kimi_k3` returns the schedule wrapped so that its `step` runs the vision phases around the core step; the engine's pipeline step body has no model hook for them. The runtime lives in the model's pipeline package, next to the attention residual stage it extends.

## Relation to earlier revisions of this PR

The earlier revisions gave the tower a pipeline stage of its own on the first rank and placed its encodes in that rank's idle slots. That kept the tower inside the pipeline split, on one rank, with its autograd graph alive until its backward; this revision replaces it.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_dep_plan.py tests/unit_tests/cpu/test_kimi_k3_vision_dep.py -q` (18 passed).
  - `test_kimi_k3_dep_plan.py`: on the Interleaved1F1B action order at pp2 x vp4, pp4 x vp2 and pp8 x vp4, every encode in an idle slot finishes before its consumer, every backward starts after its gradient arrives, planned work sits in idle slots without overlap, both ends of each transfer post it at one slot boundary and every pair of ranks posts its transfers in the same order, a transfer leaves after its data exists and arrives before its use, a backward uses the rest of an idle run that began before its gradient was ready, a slot lasts as long as its longest action, and with a cheap encode every micro-batch after the upfront ones is hidden. Replaying the schedule's own sends and receives with every kernel waiting for its rank's unmatched transfers, no rank is left stuck, with the process in either placement.
  - `test_kimi_k3_vision_dep.py`: four ranks on gloo, pp4 x vp2, eight micro-batches with two text only, with the work before and after the schedule, with it in idle slots, and with a cheaper tower whose backward waits inside an idle run for a gradient that is ready later: the step-1 loss and every gradient are bitwise with one device, the step-2 loss and text gradients are bitwise and the tower gradients agree to fp32 summation order; a frozen tower gets no gradient; eval between steps matches one device. At pp2 x tp2, each copy receives its tensor-parallel shard of the tower and the tower's gradient is the sum of every rank's shards. Building the tower's copy leaves the seeded random stream where it was, so the model initializes the same with the process on or off.
- `pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q`, the same four ranks under NCCL with a tower whose kernels first load inside the step, in both placements (3 passed on 4 H100s).
- The B200 cell `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4_vision_dep`: pending.
- DEP on and off on one warm compile cache, and `bubble` on and off, loss, gradients and step time: pending (H100).

--- PASTE END ---
