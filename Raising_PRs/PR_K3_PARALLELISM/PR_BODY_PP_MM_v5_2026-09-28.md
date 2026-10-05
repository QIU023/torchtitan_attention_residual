# PR 4381 body v5（draft PR，`k3_pp_mm`），2026-09-28：按 K2.5 DEP 原文和 K3 §5.2.3 重写

## 状态（不粘贴）

- **10-04 CPU 会话核对 PR head `8518f473f`（用户："拉取 核对DEP最新分支和body和结果，目前哪个测试是还原K3论文的气泡排布的？"）：**
  - 三个分支一致：`dep_review1` = `k3_pp_mm` = #4381 head = `8518f473f`。本机 CPU 20 passed、46 subtests；规划器在 pp3 × vp4、6 个 micro-batch、代价比 0.01 到 1/3 时和图 11 逐项相同；真跑运行时（3 个 gloo rank、12 个 stage，代价比 0.1 和 0.3）每个 rank 的执行顺序和图 11 完全一样，文本动作之间没有 ViT 工作，第 1 步梯度和 loss 与单卡逐位相同。
  - 线上 body（08:31 UTC）是 H100 之前的版本：覆盖率和 `bubble=True` 的数值还写着 Pending，比这里的粘贴区旧。线上多了一段用户加的 "Illustration Figure"（图 11 截图），粘贴区原来没有，已经合进来，放在 Summary 之后、Design 之前；图片的 alt 是截图默认文件名，在线上显示成乱码（"屏幕截图" 的编码坏了），改成英文 "Figure 11 of the Kimi K3 report"。整段贴的时候用这里的版本，图不会丢。
  - Results 的 "one image per sample" 改成 "at most one image per sample"：这组数据每步 16 个 micro-batch 里 13 个带图（`DEP_K3RANGE_H100_2026-10-02.md`），规划日志也是 13 个。
  - #4381 和 main 在 `kimi_k3/model.py` 有冲突（#5026 在 `vision_dep` 旁边加了 `local_compile_regions`），rebase 等用户的话。

- **10-05 H100：** NCCL 单测在 `8518f473f`（干净的工作树）上 2 passed，粘贴区那行从 Pending (H100) 改成 "2 passed on 4 H100s"。粘贴区现在没有 Pending 了。
- **10-04 H100（`8518f473f`，`kit_dep_bands_2026-10-04/h100_dep_bands.sh`，结果在本机 scratchpad 的 `h100_1004/dep_fig11/`）：**
  - 数值（seq 2048、224 px、M16、代价比 0.116、deterministic、动态形状关，100 步，同一份缓存的副本）：DEP 关两遍、K2.5、bubble 四次运行每一步都相同；DEP 关的第 1 / 10 / 50 / 100 步和 10-02 那次也完全一样（8.15085 / 34.0000 … 2.05389 / 4.9375），表里那一行不变。
  - 覆盖率（标注 kernel 时间）：代价比 0.046 / 0.136 / 0.169 时，K3 形式 71% / 71% / 69%，K2.5 形式 1% / 6% / 0%。旧放置规则（10-02）是 73% / 81% / 84%。差别来自报告的固定结构：后 PP 个 micro-batch 的反向现在在调度后做，这组数据里是 13 个带图的 micro-batch 中的 3 个（以前只有 1 个）；编码进气泡的是 10 个（以前 9 个）。规划日志："encodes 3 before the schedule, 10 in idle slots; backwards 10 in idle slots, 3 after it"。
  - 粘贴区改了：覆盖率表填回（表头和说明和 10-02 那版一样），数值那句加上 `bubble=True`。GPU 单测那行仍是 Pending (H100)：这一轮 H100 没跑它，已排在 MoonEP 那一轮后面。
- **10-04 PR 分支同步（用户："DEP排布的检查有加到unit test吗？review分支检查过了吗？有的话推pr分支"）：** `k3_pp_mm` 从 `6f5312fab` 快进到 `8518f473f`（= `dep_review1`），普通推送，不是强推；#4381 现在 7 个提交、11 个文件、+2198 / −11。推之前在 `8518f473f` 上核对过：CPU 20 passed（46 个 subtest）；图 11 的单测和首尾固定的单测在 `6f5312fab`、`6b580e438` 两个旧规划器上各失败 7 个 subtest；运行时执行顺序检查在旧规划器 `6f5312fab` 上失败；NCCL 单测在 4 张 5060 上 2 passed（含新的执行顺序检查，代价比 0.25）。GitHub 显示和 main 冲突：只有 `kimi_k3/model.py` 一处，main 的 #5026（`e76e810c7`）在 `KimiK3Model.Config` 同一位置加了 `local_compile_regions`，DEP 在那里加 `vision_dep`，两个字段都留即可；推之前的 `6f5312fab` 也是同样的冲突。rebase 要等你的话。粘贴区可以贴，覆盖率表和 `bubble=True` 的数值仍是 Pending (H100)，H100 正在跑 `8518f473f`。
- **10-04 按审查意见改（用户转来的审查 1 到 3 条，并说"别drift了，排布还得是这样的"）：** `dep_review1` = `8518f473f`（替换 `5b01a6932`，备份 `backup/dep_review1_pre_20261004b`）。排布不变：每个 ViT 工作的 rank、所在气泡和先后顺序仍是图 11。去掉了规划器把结尾气泡里的反向报成"贴着段尾"的那段（运行时从来是在 rank 最后一个文本动作之后就做）；图 11 的单测改成断言 rank、所在气泡和顺序；运行时测试新增逐 rank 的执行顺序检查；docstring 压短。粘贴区改了两处措辞（"ahead of the final ones"、表下面那句测试说明），通过数不变（20 passed）。
- **10-04 按图 11 重排（用户："为什么？不要随便否认图里面的正确性，并且现在的DEP body还得想办法在PR head代码里面yield和图片完全一致的stage排布"）：** review 分支 `dep_review1` = `5b01a6932`（替换 `6b580e438`，备份 `backup/dep_review1_pre_20261004`），PR 分支 `k3_pp_mm` 仍是 `6f5312fab`。细节在 `DEP_K3_FIG11_REVIEW_2026-10-04.md` 的"第二轮"。
  - 粘贴区改了：Design 的 Placement 换成新规则，Fit 补上 ViT 反向按 3 个前向算；Design 末尾加图 11 的排布表和检查它的测试名；Results 里隐藏率表和 `bubble=True` 的 100 步数值改成 Pending (H100)（都是旧放置规则下测的），DEP 关和 `bubble=False` 那句保留（K2.5 形式的规划结果不变，随机 592 组逐项相同）；Test plan 的 CPU 通过数 18 → 20，GPU 那行改成 Pending (H100)（本机 5060 上 2 passed，只记在 logbook）。
  - 贴之前要先把 PR 分支同步到 `5b01a6932`（等你的话）。

- **10-02 Design 改成两级 bullet（用户："Design里面又是大段段落，改成bullet points嵌套"）：** 原来三段的内容拆成六组（ViT copies、One micro-batch's ViT work、Placement、Fit、Transfers、Wiring），每条一行，内容没有增删。

- **10-02 CPU 会话在上一条之上补了三处（同一句用户要求，两个会话同时改，合并时以 GPU 会话那版为底，没有加长）：**
  - 粘贴区里 tower、encoded 统一改成 ViT 的说法（ViT forward、ViT backward、debug ViT）。
  - 第一次出现 pipeline bubbles 的地方注明是"idle slots in a rank's action order"。
  - Design 补一句代价模型：格子按最长动作计时（前向 1、反向 2），`bubble_cost_ratio` 决定放不放得下，两份报告都没给代价模型，这些数是 plan 自己的估计。
  - 数值表的数字从 `kit_h100_2026-10-02/results/dep_new_numerics/*/run.log` 重新核对过：五次运行（DEP 关两次、`bubble=False`、`bubble=True` 新旧代码）100 步全部逐位相同。

- **10-02 写法改清楚（用户："DEP body里面写的太confuse了，`bubble`是什么意思 表格里面应该写ViT computation covered in bubble之类的意思"）：**
  - 表名改成 "ViT computation covered by pipeline bubbles"。
  - 两列改成 `bubble=False`（K2.5 形式）和 `bubble=True`（K3 形式）。
  - 比例那列写成"ViT forward / text stage forward"。
  - Summary 和 Design 里第一次出现 `bubble` 时写明含义；数值那句也改成 `bubble=False` / `bubble=True`。
  - B200 那行从 pending 改成本机 5060 上 10 步跑完（用户："这个在本机5060跑就行，10步"）。PR head `6f5312fab`，8 张卡，rc 0，日志在 scratchpad 的 `dep_b200cell_5060/`。

- **10-02 大幅精简（用户："DEP body太spamming了，大幅度精简简洁"）：** 粘贴区整段重写，从约 1600 词压到约 500 词。
  - Summary：一句话加三条。
  - Design：三段，分别讲机制、放置、为什么放在这里；传输只留一句。
  - Results：隐藏率表只留百分比；数值表只放 DEP 关一行，保留第 1、10、50、100 步四列，加一句"其余三格 100 步全部相同"。
  - Test plan：只留命令和通过数。
  - 去掉了 Optimus 的那句引用。

- **10-02 B200 格子删掉，现有格子打开 DEP（用户："为什么又加b200 recipe？规则没写清楚吗？不能乱加CI cell！删了……kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4 里面的config直接默认打开DEP就行了"）：**
  - `dep_review1` 和 `k3_pp_mm` 都从 `b2a57dff7` force-with-lease 推到 `6f5312fab`，旧 head 备份在 `backup/dep_review1_pre_20261002b` 和 `backup/k3_pp_mm_pre_20261002b`。
  - 加 B200 格子的提交 `6b489c94e` 直接从历史里去掉（`rebase --onto`），不留"加了又删"。后面三个提交内容不变，SHA 变成 `e9fce9193`、`2afbf3f54`、`8ba759831`（重构）。
  - 新提交 `6f5312fab`：现有 recipe `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4` 加一行 `config.model.vision_dep.enabled = True`，是 K2.5 形式，bubble 没开。被删的 recipe 还开了 bubble、让偶数 DP rank 只有文本，这两项没搬。
  - PR 现在 6 个提交、11 个文件、+2092 / −11。DEP 的代码和测试与 `b2a57dff7` 逐字节相同，所以 H100 上在 `b2a57dff7` 测的隐藏率、数值和 GPU 单测对 `6f5312fab` 同样成立。
  - 本机：CPU 27 passed、28 subtests；ufmt、flake8 干净。改过的 B200 格子（8 卡）在当前树上还没在 GPU 上跑过。
  - 粘贴区改了：Summary 和 Test plan 里 B200 那两条。

- **10-02 PR 分支已同步（用户："拉取，然后DEP直接推draft PR分支"）：**
  - `k3_pp_mm` 用 force-with-lease 从 `d27839459` 推到 `b2a57dff7`，旧 head 备份在 `backup/k3_pp_mm_pre_20261002`。#4381 是 draft，6 个提交、13 个文件、+2106 / −11，可合并（CI 待跑）。
  - 内容是四个 DEP 提交 rebase 到 main `db050eb3f`，加上类型和 docstring 的修正 `a93cd48ea`，再加上 CPU 会话的重构 `b2a57dff7`。
  - 重构核对：AST 有 64 个定义相同，规划器等价检查 0 处不同；H100 上同一个 bubble 格新旧两棵树 20 步逐位相同。100 步那组和新代码的 GPU 单测还在跑。
  - 标题前缀 "[DO NOT review, pending K3 text PP merging]" 已经过时（#4312 在 09-26 合了），要不要改由你定。
  - 粘贴区改了：Summary 里的 recipe 路径；Design 末尾加了一句 Optimus 的引用（K3 报告 v2 的 [34]），并写明这里拆的单位是整个 micro-batch；新增 Results，先放隐藏率表；去掉 Test plan 里"DEP on and off ... pending"那条。之后又补了 100 步数值表。
  - **已填完，可以贴（09:08 UTC）。** 数值表：DEP 关、DEP 关再跑一次、K2.5、bubble 四格，100 步全部逐位相同。
  - 新代码的 GPU 单测 3 passed（4 张 H100），Test plan 里那行的数不变，现在对应的是 `b2a57dff7`。
  - B200 格要 8 张卡，仍写 pending。

- **10-02 结构重构（用户："按照tianyu在4312 comment针对cache/hook方案重构为attnrespipelinestage的方式重构……不能影响数值"）：** review 分支 `dep_review1` 从 `a93cd48ea` 快进到 `b2a57dff7`，PR 分支 `k3_pp_mm` 没动（仍是 `d27839459`）。DEP 代码搬进 `pipeline_parallel/vision_dep/`，按 4312 的拆法分成 `plan.py` / `runtime.py` / `stage.py` / `schedule.py` / `__init__.py`；数值不变（逐位 A/B、规划器等价检查），细节和审计在 `DEP_VISION_DEP_PACKAGE_2026-10-02.md`。粘贴区改了 Summary 的文件列表、Design 末句、Test plan 里的测试文件名（通过数仍是 18）。GPU 那边 H100 的 K3 区间测量跑在 `a93cd48ea` 上，结果对 `b2a57dff7` 同样成立。
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

Implements the decoupled encoder process (DEP) of Kimi K2.5 for the Kimi K3 pipeline, and, as in the K3 report (sec 5.2.3), an option that schedules the ViT computation into pipeline bubbles.

- `kimi_k3/pipeline_parallel/vision_dep/`: `VisionDepPlan` (`plan.py`), `VisionDep` (`runtime.py`), `VisionDepPipelineStage` (`stage.py`), `VisionDepSchedule` (`schedule.py`), wired into `pipeline_kimi_k3` by `__init__.py`.
- `kimi_k3/model.py`: a `vision_dep` config (`enabled` turns DEP on, `bubble` schedules the ViT computation into pipeline bubbles, `bubble_cost_ratio` is the cost the scheduler assumes for it) and a `vision_embeds` argument on `forward`.
- `torchtitan_recipes/tests/suites/b200.py`: the existing pp2 x vpp4 B200 recipe enables `vision_dep`.

## Illustration Figure

The green part of ViT forward and backward shown from the Kimi K3 tech report and implemented in this PR:

<img width="1546" height="488" alt="Figure 11 of the Kimi K3 report" src="https://github.com/user-attachments/assets/b236e986-af6a-49c0-8bf9-baa03906e16a" />

## Design

- ViT copies:
  - Every pipeline rank holds a tensor-parallel copy of the ViT, refreshed from stage 0 each step.
  - Stage 0 keeps the parameters, so optimizer, checkpoint and FSDP are unchanged.
- One micro-batch's ViT work:
  - The ViT forward runs under `no_grad` on one rank, and its features go to stage 0 as `vision_embeds`.
  - The gradient at the features goes back to one rank, which recomputes the ViT forward and runs the ViT backward.
  - The copies' gradients are summed in fp32 and reduced into the ViT's gradients at the end of the step.
- Placement:
  - `bubble=False` (the K2.5 form): every ViT forward runs before the pipeline schedule and every ViT backward after it, balanced by patch count.
  - `bubble=True` (the K3 form): the ViT forwards of the schedule's first pipeline-degree micro-batches run before it and the ViT backwards of its last pipeline-degree micro-batches after it, balanced by patch count.
  - The other ViT forwards run in the pipeline bubble that opens each rank's schedule (its idle slots before its first action), right after the upfront ones, and the other ViT backwards in the bubble that closes it, ahead of the final ones.
  - Each goes to the least filled of those bubbles for its length, the lower rank on a tie, so a rank's share grows with its bubble.
  - Work that fits no bubble falls back to before or after the schedule.
- Fit:
  - A slot lasts as long as its longest action (forward 1, backward 2), `bubble_cost_ratio` is one ViT forward in units of one text-stage forward, and a ViT backward costs three ViT forwards (the recompute and the backward).
  - Neither report gives a cost model; these numbers are the plan's own estimates.
  - Every rank derives the same plan from `pipeline_order`.
- Transfers:
  - Features and gradients use a process group per pipeline group.
  - Both ends of a transfer post it at the same slot boundary, so no posted send or receive waits on its own rank's later work.
- Wiring:
  - `pipeline_kimi_k3` wraps the schedule's `step`, since the engine's pipeline step has no model hook.
  - The package is split like the AttnRes pipeline: the plan like the block layout tables, the runtime like the rank store, and the stage subclasses the AttnRes stage.

With pp 3 x vpp 4 and 6 micro-batches the plan is the layout of Figure 11 in the K3 report (micro-batches numbered from 1); `test_three_ranks_and_six_microbatches_lay_out_as_in_the_k3_report` checks the rank, bubble and order of every item for cost ratios 0.05 to 0.3, and the runtime tests check that every rank runs exactly its planned ViT work, in order, before its first text action and after its last:

| rank | ViT forward before the schedule | ViT forward in the opening bubble | ViT backward in the closing bubble | ViT backward after the schedule |
|---|---|---|---|---|
| 0 | 1 | | | 4 |
| 1 | 2 | 4 | 1 | 5 |
| 2 | 3 | 5, 6 | 2, 3 | 6 |

## Results

4 H100s, pp4 x vpp4, Interleaved1F1B with 16 micro-batches, full activation checkpointing; the Kimi K3 debug model widened to dim 6144 with its 2-layer debug ViT, at most one image per sample.

ViT computation covered by pipeline bubbles: the share of the ViT forward and backward kernel time that runs inside pipeline bubbles, one traced step (the ViT work wrapped in profiler ranges locally for the measurement):

| seq | image side | ViT forward / text stage forward | covered, `bubble=False` (K2.5 form) | covered, `bubble=True` (K3 form) |
|---:|---:|---:|---:|---:|
| 2048 | 224 px | 0.046 | 1% | 71% |
| 2048 | 1008 px | 0.136 | 6% | 71% |
| 1536 | 1008 px | 0.169 | 0% | 69% |

Loss / grad norm, seq 2048 with 224 px images, deterministic, one warm compile cache, automatic dynamic shapes off (the ViT and the text share the compiled flex attention):

| cell | step 1 | step 10 | step 50 | step 100 |
|---|---:|---:|---:|---:|
| DEP off | 8.15085 / 34.0000 | 7.50268 / 27.6250 | 2.57208 / 9.8750 | 2.05389 / 4.9375 |

DEP off again, DEP with `bubble=False` and DEP with `bubble=True` are identical to DEP off on all 100 steps.

## Relation to earlier revisions of this PR

Earlier revisions gave the ViT its own pipeline stage on the first rank; this revision replaces that.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_vision_dep_plan.py tests/unit_tests/cpu/test_kimi_k3_vision_dep.py -q` (20 passed).
- `pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q` (2 passed on 4 H100s).
- The B200 cell `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4` with DEP on: 10 steps on 8 RTX 5060 Ti (no B200 at hand).

--- PASTE END ---
