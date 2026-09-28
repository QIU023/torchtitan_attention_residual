# DEP 分支对照原文重审（2026-09-27）

用户："我怀疑DEP我们实现歪了，可能确实有气泡掩盖的效果，但是参照原始DEP的描述和这里的描述，重新检查DEP分支的diff和body"，附 K2.5 的 DEP 原文和 K3 报告 §5.2.3 那段。

范围：
- 分支 `dep_review1` = `k3_pp_mm` = `31f372593`，是 main `f35966713` 上的 4 个提交，12 个文件，+1290/−19；
- body `PR_BODY_PP_MM_v4_2026-09-27.md`。

## 0. 结论

- **实现歪了，歪在定义上。** 分支做的是：视觉塔单独占一个 PP stage，只在持有它的那个 rank 的空闲槽里编码，塔的反向保留计算图、延后重放。DEP 的三个要件一个都没有：
  1. 塔在每个 GPU 上都有，不属于任何 PP stage，文本的切分和纯文本训练一样；
  2. 视觉前向按图像或 patch 数摊到所有 GPU；
  3. 前向不留中间激活，只留输出，反向之前重算。
- **误读从哪来。** K3 那句 "splits ViT and text training into separate stages" 里的 stages，指的是一个训练步里的阶段。K2.5 原文写的是 "composed of three stages in each training step"，不是 pipeline stage。08-08 第一版就把塔做成了一个 stage（`phase13_k3like_48b_posttrain/DEP_ALIGNMENT_2026-08-09.md`："DEP takes a stage for the vision tower out of the text budget"），之后一直沿用。
- **这个偏差以前记过，但没有带进现在的分支。**
  - 08-20 的撤回文档（`phase13_k3like_48b_posttrain/DEP_MEASUREMENT_RETRACTION_2026-08-19.md`）和 09-05 的审计（`phase13_k3like_48b_posttrain/DEP_AUDIT_2026-09-05.md` §1）都写了"实现的机制 ≠ 报告的机制"。
  - 09-10 重建时没有带过去。v4 body 的 Summary 又写成 "Report sec 5.2.3: the vision tower takes a pipeline stage of its own"。
- **气泡掩盖确实有效果，但只用到了 rank 0 自己的空闲槽，而且被时序卡住。**
  - 按逻辑槽计数（§3），pp8 × vp4、M16 时只有 2/16 个前向进了气泡，6 个只能 inline。塔的开销比取 0.25、0.5、1 都是 2。
  - 如果每个 rank 上都有塔，同一个计数里除了 upfront 的 8 个，其余 8 个前向都能进气泡，反向能进 13/16。
  - 08-20 那条"前向藏不住是机制本身的形状"，是塔只在一个 rank 上这个前提造成的，要收回。

## 1. 原文要点

**K2.5（DEP 本身），一个训练步分三个阶段：**
- **Balanced Vision Forward**
  - 先对全局 batch 的全部视觉数据做前向。
  - 塔小，所以不管用了什么并行，每个 GPU 上都放一份。
  - 前向的工作量按负载指标（图像数或 patch 数）平均分到所有 GPU，消除 PP 和视觉 token 数造成的不均衡。
  - 丢掉全部中间激活，只留输出，结果汇集到 PP stage 0。
- **Backbone Training**
  - 骨干网络的前向和反向。因为上一阶段没留中间激活，这里可以直接用纯文本训练里验证过的任何并行方式。
  - 这一阶段结束时，梯度累积在视觉输出上。
- **Vision Recomputation & Backward**：重算塔的前向，再做反向，得到塔的参数梯度。
- 目的有两个：负载均衡；把塔的优化策略和骨干的优化策略解耦。

**K3（在 DEP 上的细化）：**
- DEP 把 ViT 和文本训练分成不同阶段，并把视觉的前向、反向均衡到各个 PP stage。
- 在 interleaved 1F1B 下，前几个 PP micro-batch 的文本前向全都排在最开头，最后几个的文本反向到最末尾才结束。所以：
  - 前几个 micro-batch 的 ViT 前向同步地放在最前面；
  - 其余的前向排进流水线气泡；
  - 反向同理。
- 这样大部分 ViT 计算藏在气泡里。

## 2. 逐条对照

| DEP 的要求 | 原文 | 分支 `31f372593` |
|---|---|---|
| 塔放在哪 | 每个 GPU 一份，与其他并行方式无关；不在 PP 切分里 | 单独占一个 PP stage，`vit_dep_split` 把 `[tok_embeddings, vision_encoder]` 放在第一个 stage，只在 PP rank 0 上；参数在第一个 stage 的根 FSDP 单元里 |
| 骨干的切分 | 用纯文本训练验证过的并行方式，不因为塔而改变 | 文本层只分到 `num_stages - 1` 个 stage（`_generate_llm_fqn_per_model_part(num_stages - 1, ...)`），塔 stage 占掉一个文本 stage 的名额 |
| 视觉前向的负载均衡 | 按图像或 patch 数在所有 GPU 间平均分配 | 没有。每个 DP 副本的 rank 0 编码本副本的全部图像，其余 PP rank 不参与；视觉 token 数在 DP rank 间的差异原样保留 |
| 第一阶段的显存 | 只留输出，丢掉全部中间激活 | `VisionFeatureCache.encode` 带梯度跑塔，计算图一直保留到塔的反向；`GradQueue` 的 docstring 自己写着 "Each waiting entry keeps one micro-batch's tower graph alive" |
| 结果去哪 | 汇集到 PP stage 0 | 本来就在 rank 0 上算，不涉及汇集 |
| 视觉输出上的梯度 | 骨干阶段结束时累积在视觉输出上 | 只在 bubble 模式下，`cut_for_deferred_backward` 在拼接处用 detached 替身接住梯度；prefetch 模式下梯度直接流进 stage 的反向 |
| 第三阶段 | 重算塔的前向再反向，均衡到各 PP stage | 不重算，直接对保留的计算图做反向；全部在 rank 0 上，每个反向动作之后跑一个（`run_next`），剩下的在步末 drain |
| 前几个 micro-batch 的前向同步在前（K3） | 有 | 有：`upfront=min(pp_size, n)`（第 1 步全部 inline） |
| 其余前向排进气泡（K3） | 各 PP stage 的气泡 | 只用 `pipeline_order[0]` 里的 `None` 槽 |
| 反向同理（K3） | 各 PP stage 的气泡 | 没有规划：每个反向动作之后跑最早的一个待处理项，不看那里是不是空闲 |
| 塔与骨干优化策略解耦 | 是目的之一 | 相反：塔在 PP 切分里，和第一个 stage 一起分片 |

## 3. 气泡：塔只在 rank 0，还是每个 rank 都有

**模型和前提：**
- 用 torch 的 `ScheduleInterleaved1F1B` 算出每个 rank 的 `pipeline_order`（本机 torch 2.13，`_calculate_single_rank_operations`）。
- 每个动作占 1 个逻辑槽，`None` 是空闲槽。
- 塔前向 1 槽，塔反向 2 槽；送到 stage 0 或从 stage 0 送回，各算 1 槽。
- 这是计数，不是实测：没算真实耗时、PCIe 或 NVLink 传输、第一阶段的 all-to-all，也没算塔的参数 gather。

**分支那一列**用分支自己的 `plan_for_rank`（`dep_bubble_plan.py`，来自 `31f372593`）。**每个 rank 那一列**是一个贪心计数：
- 前向：只要任何一个 rank 在"rank 0 消费它的时刻减 1"之前有空闲槽，就算能进气泡；
- 反向：只要任何一个 rank 在"B0.m 之后加 1"有两个空闲槽，就算能进气泡。

| 形状 | 分支：进气泡的前向 / upfront / inline | 每个 rank 都有塔：进气泡的前向 / 反向 |
|---|---|---|
| pp4 × vp2，M8 | 2/8，4，2 | 4/8，5/8 |
| pp4 × vp2，M16 | 4/16，4，8 | 12/16，13/16 |
| pp8 × vp4，M16 | 2/16，8，6 | 8/16，13/16 |
| pp8 × vp4，M32 | 8/32，8，16 | 24/32，29/32 |

- 分支那一列不受预算限制。pp8 × vp4、M16 时开销比取 0.25、0.5、1，都是只能放进 2 个，其余全都 `exhausted`。
- 原因是 rank 0 的空闲槽几乎都在 cooldown 段，前向的消费点却在前面。这正是 08-20 看到的"气泡在尾、需求在头"。
- 其余 rank 在开头的 warmup 段有 1 到 P−1 个空闲槽，中段也有，时间上正好赶在 rank 0 消费之前。分支的塔只在 rank 0 上，这些槽用不上。
- 脚本和输出：`kit_dep_mixed_2026-09-27/report_check/`（`order_probe.py`、`allrank_probe.py`、`allrank_probe.out.txt`）。运行时需要把分支里的 `dep_bubble_plan.py` 放在同一目录。

## 4. diff 里其他的问题

1. **计算图保留在显存压力最大的 rank 上。** upfront 的 P 个 micro-batch 在步首带梯度编码，塔的计算图一直活到各自的反向。这些都压在 rank 0 上，而 1F1B 下 rank 0 在飞的 micro-batch 本来就最多。DEP 只留输出。
2. **`install_vision_dep` 把 `pp_schedule.step` 换成了包装函数。** 这是 monkeypatch（#4577 规则第 3 条），body 没有写明缺的是哪个 seam。
   - DEP 的第一和第三阶段是训练步层面的事。
   - `TrainingEngine` 在 PP 路径上经 `forward_backward_body_fn`（`_pp_forward_backward_body`）调 `pp_schedule.step`，每个 PP rank 手里已经有本 DP 副本全部 micro-batch 的 `kwarg_mbs`，其中包括 `pixel_values`、`grid_thw`（`preprocess_inputs` 在每个 rank 上都跑）。
   - 模型这一侧目前没有 hook 能包这个 body，这才是该提的 seam。
3. **optimizer 那个提交（`ae7e08d1e`）的动机会跟着塔 stage 一起消失。** 它是因为只有塔的 stage 匹配不到 Muon 的矩阵才加的。提交里提到的"只有 head 的 stage"是不是真实存在的配置，要单独确认；如果不存在，这个提交就没有理由留下。
4. **bubble 开和关的数值差，解释不成立。**
   - 实测（`kit_dep_mixed_2026-09-27/bubble_onoff/cmp_grads.txt`）：第 2 步，塔的 22 个参数梯度全不同（相对差 1.2e-3 到 3.1e-3），文本的 443 个逐位相同。
   - 提交 `70ffbaf9d` 的说明把它归因于 "Gradients accumulate, so the replay is exact but not bitwise against the inline order"。
   - 但按代码，两种模式下塔反向的顺序相同，都是按 micro-batch 升序：
     - 延后的那些，在 `B0.m` 结束后由 `after_backward` 马上 `run_next`，跑的正是刚存进去的 m；
     - inline 的那些，在 `B0.m` 里面跑。
   - 所以累加顺序没变，这个解释站不住，差异的来源还没定位。按数值验收规则，定位之前不能写 "exact"。
   - 下一步的探针：第 2 步每次塔反向前后各取一次塔的梯度，算出每个 micro-batch 自己的那一份，在两种模式之间比。
     - 如果单个 micro-batch 的就不同，问题在重放的反向本身（例如 AC 重算时所处的上下文）；
     - 如果单个的都相同、累加后才不同，再回头查累加。

## 5. body（v4）

- **Summary 的出处写错了。** 原文没有让塔占 PP stage，K2.5 写的恰恰是每个 GPU 一份、不受其他并行方式影响。"Report sec 5.2.3" 不能挂在这个机制上。
- **Design 第一条和 DEP 的目的相反。** "the tower stage comes out of the text stages' budget" 改变了骨干的切分，而 DEP 的目的之一就是让骨干保持纯文本的并行方式。
- **标题里的 "schedule ViT stages into text LLM PP bubbles"** 用的也是 PP stage 的说法。
- **负载均衡一个字都没有**，而这是 DEP 的第一个主张。
- **Test plan 里三条 pending (H100)** 测的都是这个机制本身。机制换了，这几条就作废。

## 6. 建议（一个立场）

#4381 保持 draft，标题前缀不去掉，v4 body 不贴。按原文重做，分两步：

1. **第一个 PR：DEP 本身（K2.5 的三个阶段）。**
   - 塔不进 PP 切分，每个 PP rank 一份（参数可以按 ZeRO 分片）。
   - 第一阶段：
     - 按 patch 数把本 DP 副本全部 micro-batch 的图像分给这个副本的 P 个 rank。图像已经在每个 rank 的 `kwarg_mbs` 里，所以不用搬图像，只把特征送回 stage 0。
     - 各 rank 在 `no_grad` 下编码，只留输出。
     - 跨 DP 的均衡（K2.5 的 "all GPUs"）作为后续的开关。
   - 第二阶段：文本的切分不变。stage 0 通过分支现有的 `vision_embeds` 参数读特征，在拼接处接住梯度（`cut_for_deferred_backward` 的做法可以复用，但只存梯度，不存计算图）。
   - 第三阶段：
     - 梯度送回做编码的 rank，重算，再反向；
     - 塔的梯度在持有副本的全部 rank 上规约，grad norm 里只算一次。
2. **第二个 PR：K3 的细化。** 前 P 个 micro-batch 的编码同步放在最前面；其余的编码，以及重算加反向，排进各 rank 的空闲槽。`plan_for_rank` 改成对每个 rank 规划，消费点取 stage 0 的时刻减去传输时间。

**丢掉：** `vit_dep_split` 和塔 stage，`pp_schedule.step` 的 monkeypatch。optimizer 那个提交也丢掉，除非确认确实存在只有 head 的 stage。

**可以复用：** `encode_images` 和 `forward(vision_embeds=...)`；在拼接处接梯度的写法；空闲槽规划的逻辑（改成每个 rank 都规划）。

**写代码之前先查清楚的：**
1. 塔放在 PP 切分之外，每个 rank 怎么留一份。`pipeline_module_split` 会删掉不在 FQN 列表里的模块。
2. 塔在 pp 维上重复，checkpoint 的保存和加载怎么去重，grad norm 怎么只算一次。
3. 第一和第三阶段放在哪个 seam：`TrainingEngine.forward_backward_body_fn`，模型侧目前没有 hook。
4. `kimi_k2_7` 的 MoonViT 分片（`sharding.py`）哪些能直接用。

## 7. 要用户定的

1. 同不同意按 §6 重做。一旦重做，#4381 现在的 4 个提交基本作废，只保留 §6 列出的可复用部分。
2. 第一个 PR 的均衡范围：先只在一个 DP 副本的 PP rank 之间做（不搬图像），还是一开始就跨 DP（需要 all-to-all）。
3. 在重做之前，线上的 #4381 要不要先发一句说明。它现在的标题和旧 body 都在说 DEP。

## 8. 重写（2026-09-28，用户："DEP原文+K3说的气泡掩藏都纳入考虑了吗？你本地直接在k3 pp mm重写"）

**§6 的两步合成一次重写，两段原文都做进去了。** 本地 `k3_pp_mm` = `637ddb20f`（main `f35966713` 上 3 个提交，worktree 在 `C:/Users/78532/AppData/Local/Temp/claude/dep`），**没有推送**。fork 上的 `k3_pp_mm` 和 `dep_review1` 仍是旧实现 `31f372593`。正文草稿：`PR_BODY_PP_MM_v5_2026-09-28.md`。

**对应关系：**

| 原文 | 实现 |
|---|---|
| K2.5：塔在每个 GPU 上都有，不受其他并行方式影响 | 每个 PP rank 一份计算用的副本（不属于 model part），每步开头从 stage 0 那份的 `full_tensor()` 广播过来 |
| K2.5：骨干用纯文本的并行方式 | 文本切分就是 core 的切分，不变；塔留在 stage 0，照常归优化器、checkpoint、FSDP 和 grad norm 管，但训练时不再运行 |
| K2.5：按图像或 patch 数均衡 | 以 micro-batch 为单位，按 `grid_thw` 算 patch 数，在一个 DP 副本的 P 个 rank 之间用 LPT 均衡。图像本来就在每个 rank 的 `kwarg_mbs` 里，不用搬 |
| K2.5：只留输出，结果汇集到 stage 0 | 编码在 `no_grad` 下进行，特征经独立的通信组发到 stage 0；stage 0 通过模型新增的 `vision_embeds` 参数拼进去 |
| K2.5：梯度累积在视觉输出上，然后重算加反向 | stage 0 在特征叶子上挂 hook 接梯度，发给 plan 指定的 rank；那个 rank 重算塔再做反向，梯度按 fp32 累加；步末 reduce 到 stage 0，再在 dp 上 all-reduce，最后切回原始塔的本地分片 |
| K3：前 PP 个 micro-batch 的前向同步放在最前面 | `bubble` 开时，stage 0 最先消费的 P 个 micro-batch 在调度开始之前编码，并均衡到各 rank |
| K3：其余前向排进气泡 | 其余的编码放进任意 rank 在消费截止时间之前的空闲槽（留 1 个槽给传输），放不下的归入前面那批 |
| K3：反向同理 | 重算加反向放进 B0.m 加传输时间之后任意 rank 的空闲槽，放不下的在步末均衡地跑 |

**和旧实现比，额外修掉的：**
- 视觉计算放在独立的 CUDA 流上。旧实现在主计算流上、紧跟着锚点动作发出，runtime 随后为这个动作的输出发出的 send 会先等视觉计算跑完，视觉计算就挡在了下游 rank 的关键路径上。
- 特征和梯度各用一个独立的通信组（每个 pp 组建两个，`_connect` 预先建好点对点连接），它们的 P2P 顺序和调度自己的 send/recv 互不影响。
- `pp_schedule.step` 的 monkeypatch 换成返回一个 `_PipelineSchedule` 代理；正文里写明 engine 的 PP step 缺一个模型 hook。
- 删掉了 optimizer 那个提交：不再存在只有塔的 stage。
- 名字的去包装改用 core 的 `canonical_fqn`。

**本机验证**（Windows，torch 2.13，scratchpad 的 harness：对模型包打桩，补 nightly 的 `step(arg_mbs=...)`；harness 不进提交）：
- 20 个测试通过：`test_kimi_k3_dep_plan.py` 8 个，`test_kimi_k3_vision_dep.py` 3 个（4 进程 gloo），加上 K3 PP 的旧测试 `test_kimi_k3_pp_stage.py`、`test_kimi_k3_pp_block_grads.py`。
- 端到端结果：K2.5 模式、`bubble` 模式、冻结塔三种情况，第 1 步的 loss 和全部梯度都和单设备逐位一致；第 2 步的 loss 和文本梯度也逐位一致，塔梯度差在 fp32 求和顺序以内；eval 与单设备一致。副本初始权重清零，所以同步是真的在起作用。
- 规划器在 torch 真实的 Interleaved1F1B 动作表上（开销比 1）：pp8 × vp4 时，M16 下最前面 8 个之外的前向全部进气泡，M32 下 24/32。反向能进多少取决于开销比：开销比 1 时，pp4 × vp2、M16 只有 3/16；开销比 0.5 时，非均匀负载的配置能进 10/14。原因是一次重算加反向要占 3 个槽，cooldown 段 1 个槽的小空隙放不下。

**GPU 上要验证的**（在这之前，任何数字都不写进正文）：
1. B200 格子 `kimi_k3_fsdp2_pp4_vpp2_vision_dep`（8 卡，dp2 × pp4 × vp2，偶数 DP rank 只有文本）能跑通。这是第一次覆盖到这些路径：FSDP 分片的原始塔的 `full_tensor()` 和 `distribute_tensor`、dp all-reduce、独立 CUDA 流、NCCL 下的 `_connect`。
2. 数值，在同一份暖缓存上：DEP 关 vs 开（K2.5 模式）第 1 步的 loss 逐位一致、文本梯度逐位一致；塔梯度和 DEP 关时比较，并配一行噪声底（同一格跑两次）。`bubble` 关 vs 开：第 1 步的 loss 和文本梯度应当逐位一致；塔梯度会因反向分到的 rank 不同而改变 fp32 的求和分组，只差在求和顺序的量级，同样配一行噪声底。
3. profiler：看视觉计算是否真的落在各 rank 的空闲段里，send 是否没有被推迟；再对比 DEP 关、DEP 开、`bubble` 开三者的步时。
4. 显存：rank 0 多出一份 bf16 副本，每个 rank 多一个 fp32 累加器（塔大小乘以 4 字节）。另外，塔还在 stage 0 的根 FSDP 单元里，每步会被 all-gather 一次但用不上；如果显存吃紧，可以把塔单独包成一个 FSDP 单元。
5. 大图：重算加反向一次只处理一个 micro-batch，副本上套了和原始塔相同的 AC 策略。

**这次没做的：**
- **跨 DP 的均衡**（K2.5 的 "all GPUs"）：需要在 DP rank 之间用 all-to-all 搬图像，留作后续的开关。
- **TP > 1 时的分工：** 每个 TP 坐标的 pp 组各自完整编码一遍，也就是在 TP 上重复计算，没有切分。
