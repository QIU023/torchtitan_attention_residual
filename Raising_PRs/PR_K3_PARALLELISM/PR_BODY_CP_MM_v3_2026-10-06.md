# PR 4380 body v3（10-09 rebase 后：PR 分支 `k3_cp_mm` = `cpmm_review1` = `097068e0f`，在 main `01b1b69e3` 上，九个提交），2026-10-06

## 状态（不粘贴）

- **10-09 rebase 后复查 body（用户："rebase完之后...本身diff已经没有任何需要改动了吧？body也更新了吗？"）：**
  - 粘贴区改了三处 rebase 后过时的地方：
    1. Results 说明补一句 "These runs predate the rebase onto `01b1b69e3`, which leaves this PR's diff unchanged."；
    2. Test plan 删掉 rebase 前的 SHA `6317c5538`，改成 "matches the revision the tables measure"；
    3. "#5191" 那条改成 "Both cells train since #5191, which this branch includes."。
  - GitHub 上的 body（10-09 07:48 UTC 更新的）是更早的一版：没有 #5191 的说明，H100 那条还是 "On 4 H100"，没有 h100 格子那两条，也没有 Relation 一节。整份粘贴区要重新贴。
  - 标题仍是 "[DO NOT review, finalizing] ..."，由用户决定什么时候去掉。

- **10-09 rebase（用户："现在先rebase push，然后其他暂时不改，通用性的问题可以我单独问maintainers"）：**
  - 9 个提交原样搬到 main `01b1b69e3`（含 #5191），没有冲突。逐个提交的 patch-id 和整体 diff 的 patch-id 都和 rebase 前相同（整体 `0255879abbc1`）。
  - 推送：先把旧 head `5701f2fbc` 备份到 `backup/k3_cp_mm_pre_20261009`，再用 force-with-lease 把 `cpmm_review1` 和 `k3_cp_mm` 都推到 `097068e0f`。
  - 本机检查：
    - CPU 规划测试 7 passed，pyflakes 干净；
    - gloo 上 4 rank 跑 rebase 后的切图路径，和 rebase 前（`6317c5538`，在 `948d65c86` 上）存的输出与梯度比，368 个张量逐位相同（kit `local/vision_cp_equiv/`，CPU，两边都用稠密注意力）。
  - 下一步：打 `ciflow/h100.8`，两个 K3 CP 格子现在应能跑完。body 和代码其他地方没有改；通用性的问题由用户单独问 maintainer。

- **10-09 CPU 会话复查 diff（用户："拉取，再次检查dynamic cp diff"）：**
  - 10-08 以后没有新提交：PR head = `k3_cp_mm` = `cpmm_review1` = `5701f2fbc`，基于 `948d65c86`，9 个提交。
  - 和 main `01b1b69e3`（多了 35 个提交）合并没有冲突。PR 用到的接口在新 main 上都还在：`spmd_mesh_group`、`compiled_create_block_mask`、`VisionAttention._qkv`、`MoonViTEncoder` 的 `compute_position_embeddings` / `merge_kernel_size`、`_tpool_patch_merger`、`ComplexRoPE.apply_rotary_emb`、`FlexInnerAttention.inductor_configs` / `_compiled_flex_attn`。
  - main 上碰到 K3 的提交，和本 PR 语义上都不重叠：#4838 删了 `parallelize` 的 `skip_dp`，删了 `spmd_types` 的两个 state dict 转换函数；#5184 把 `KDAAttentionMetadata` 改名为 `LinearAttentionMetadata`。
  - **要 rebase 才能让 h100 CI 变绿：** `integration_test_h100.yaml` 由推送 `ciflow/h100.8/*` tag 触发，跑的是 PR head 本身，不是和 main 合并后的结果。PR head 的历史里没有 #5191，所以两个 K3 CP 格子在 PR 上仍会在建 DistMuon 时失败。rebase 到 main（≥ `31b503c89`）之后再打 `ciflow/h100.8`。预计没有冲突。
  - 新情况：#5184（acisseJZhong，10-09）给 Qwen3.5 加了多模态 CP，vision bank 同样在每个 CP rank 上整份编码。Qwen3.5 的 tower 用的也是 common 的 `VisionAttention`，所以 `VisionCPAttention` 对它适用；`MoonViTCPEncoder` 不适用，它的 merger 和位置表不一样，切分的前向要另写。Jessica 是 #4639 和 #5184 的作者，可能会问到这一点。

- **10-09（晚）：** #5145 被 maintainer 的 #5191 取代（acisseJZhong，10-09 06:54 UTC 合入，main `31b503c89`，做法相同）。粘贴区里两处 #5145 改成 #5191：Results 的说明改成 "did not build under CP on that main (#5191 has fixed it since)"，Test plan 改成 "Both cells train on main since #5191."。#4380（`5701f2fbc`）和 main `31b503c89` 合并没有冲突。
- **10-09：** CI 修复 PR 已由用户开成 #5145；Results 的表注和 Test plan 各补了一处 #5145。
- **10-08 CPU 会话复核粘贴区：** Test plan 第三条 "On 4 H100, the CP=2 all-gather run" 改成 "On 2 H100s"。CP=2、dp1 的格子用 2 张卡：`run_optin_h100.sh` 的端到端调用的是和 muonfix 同一个格子脚本、同一份 cache，muonfix 的日志里是 `--nproc_per_node=2`。GPU 单测那条 "on 4 H100" 本来就对，没改。其余已核对：Design 是嵌套 bullet；表只到 step 20，并在说明里交代了原因；有噪声底那一行；没有 we/our/us、破折号或非 ASCII 字符。

- **10-08 CPU 会话检查 diff 和 body（用户："拉取，检查Dynamic CP diff和body"）：**
  - **diff（`6317c5538`）：** 和 main `2154d68a9`（多了 18 个提交，只有 #5111 碰 K3）合并没有冲突。新增注释 2 行、docstring 23 行，都是一行说明。私有 def 做过复用检查：`_split_mask` 对照了 common 的 `create_block_diagonal_mask`（`models/common/vision_encoder.py:106`），后者要求 Q 和 K 等长、段号连续，表达不了 gather 之后 K 比 Q 长、补齐行编号 −1 的情况，所以不能直接用。
  - **还没在 head 上确认的：** `pytest tests/unit_tests/gpu/test_kimi_k3_vision_cp.py` 在 `6317c5538` 上没有记录（Test plan 写的 "2 passed" 是 `9b03b4af1` 时的），pre-commit 的 pyrefly 也没有。H100 上的 fp32 大图检查（12/12）和端到端都是在 `6317c5538` 上跑的，本机 gloo 的新旧对比是 368/368。
  - **body 违反两条规则：**
    1. Design 在压缩时变成了 9 条平铺的 bullet，没有标签和子 bullet，有几条用分号连着两件事，不符合 10-07 的 "Design sections are nested bullets"。压缩后的措辞可以保留，只需恢复成带标签的嵌套结构。
    2. 第一张表报了 step 50 和 100。main 的 loss 从第 50 步的 2.47760 回升到第 100 步的 3.04643，已经进入记忆数据集的阶段，按 numerics-table 规则不能报。应改成 step 1 / 10 / 20（先看 main 的轨迹第一次非单调在哪一步）。"100 步逐位相同"这类说法可以留在文字里。100 步的日志在 H100 box 的 `/workspace/h100_cpmm`，logbook 里没有第 20 步的数，要 GPU 会话从日志里取。
  - 第二张表下面的说明也有两条用分号连着两件事（"Figures include ...; flex compiles ..."），按同一条规则可以拆开。
  - **线上：** #4380 标题是 "[DO NOT review, pending check and specific scenario]"，body 是压缩前的 v3；CI 修复 PR（`k3_cp_muon_layout`）还没开。

- **10-08 改成 opt-in（用户的建议："改成 opt-in，阈值在打开时仍用 256 ... 开关要从模型 config 读 ... 在现有的两个 h100 CP 格子的 recipe 里打开它"）：**
  - **提交 `5701f2fbc` "kimi_k3: dynamic CP of the vision tower is opt-in"**（叠在 `96233d5c8` 上，单独一个提交，4 个文件 +22/−9）：`dynamic_cp_min_patches: int | None = None`；`install_vision_cp(tower, config, parallelism_context)` 在 CP 关、config 为 None 或阈值为 None 时直接返回；`KimiK3Model.parallelize` 传 `cast(KimiK3Model.Config, self.config).vision_encoder`（PP 的 `_split_module` deepcopy 整个模型，再把非本 stage 的模块设成 None，所以没有 tower 的 stage 也有完整的 config；`cast` 照 muse_glimmer 的写法，否则 pyrefly 报 missing-attribute）；两个 h100 CP recipe 各加一行 `config.model.vision_encoder.dynamic_cp_min_patches = 256`（和 DEP 在 B200 格子里加 `vision_dep.enabled = True` 一样）。
  - 先推的是 `c5fdbcca5`（没有 cast，pyrefly 多 1 个错误），amend 成 `5701f2fbc` 后 force-with-lease 推 `cpmm_review1`，`c5fdbcca5` 备份在 `backup/cpmm_review1_pre_20261008b`；用户说"推"后，PR 分支 `k3_cp_mm` 快进到 `5701f2fbc`（不是 force），`96233d5c8` 备份在 `backup/k3_cp_mm_pre_20261008b`。
  - **H100 验证（kit `run_optin_h100.sh`，在 `c5fdbcca5` 上跑，`5701f2fbc` 只多一个运行时不起作用的 `cast`）：** PP 探针（gloo，8 rank，pp2 × cp4，只有 stage 0 有 tower）打开时每个 rank 都建组、不卡，关闭时谁都不建；CPU 规划测试 7 passed；GPU 单测 2 passed in 81 s；端到端（CP2 all-gather，一份 cache，10 步）默认关闭和 main 逐位相同，打开 256 和 `6317c5538` 逐位相同（第 5 步起和 main 不同）；带 DistMuon 修复的两个 h100 recipe 新旧 10 步逐位相同，all-gather 和 Ulysses 相同。
  - **body：** Summary 的阈值那条改成开关的语义；Design 的 Sub-CP groups 改一句（开关从每个 stage 都有的模型 config 读）；两张表的表注各加一句"本 PR 的数据都打开 dynamic CP"；Test plan 加一句默认关闭和打开的逐位对比。

- **10-08 GPU 会话按检查结论修复（用户："修复，确认一下GPU单测是H100 ... 先跑完H100"）：**
  - **第一张表改成 step 1 / 10 / 20：** 数取自 10-07 H100 的 100 步日志（本机 scratchpad `h100_cpmm_results/`，就是 box `/workspace/h100_cpmm` 拉回来的那份）。main 第一次非单调在第 10 步（3.48143 → 3.68411），之后第 12、14、15、17、19、20 步也有回升，所以"第 20 步前单调"这个前提不成立。但每步 2048 token、梯度累积 1，每步只用一个样本，cc12m-test 32 个样本第 33 步才第一次重复；前 32 步的起伏是不同样本的难度，不是记住数据。表注改成说明这一点（规则允许在会被记住的小数据集上提前截止并写明，也不因参考轨迹不单调而删列）。第 20 步：main 两次都是 3.27230 / 3.1562，本 PR 3.26093 / 3.1094，所有图都切 3.25868 / 3.1250；Ulysses 和 all-gather 相同、bank 替换和 main 相同、CP=1 相同都在 100 步里核对过。
  - **Design 恢复成带标签的嵌套 bullet**（Scope、Which images split、How an image splits、One tower call per micro-batch、Attention、Vision bank、Sub-CP groups、Placement），保留压缩后的措辞。
  - **用分号连着两件事的条目全部拆开**（Summary、两张表的说明、Relation）；粘贴区文字 893 词（含表格 1151）。
  - **head 上的检查（H100 .240，`6317c5538`，`run_unit_6317.sh`）：** CPU 规划测试 7 passed；pre-commit 的 pyrefly（0.45.1，`--remove-unused-ignores --summarize-errors`，在两棵树的拷贝上跑）PR head 和 main `948d65c86` 是同一组 20 个错误，都是环境问题（没有 `torch_checkpointing`；torch 0906 缺 `param_dtype_override_fn`、pipelining 的两个名字、`scaled_addmm_` 等），没有一个落在 PR 的文件里；hook 在两边都删了同一处多余的 ignore（`video.py` 的 `import av`，因为这个 venv 装了 av）。GPU 单测在 `6317c5538` 原样（CI 路径，flex 带 max-autotune）：第一个用例（两个 DP 组、FSDP）PASSED；第二个用例（4 种情形，fp32）卡在 autotune 上，4 个 rank 各有一个 ptxas 单线程编译候选 kernel（每个 2.5 GB），25 分钟没结束，停掉了。
  - **上游 CI 会跑这个测试，而且原样会超时：** `.github/workflows/unit_test_gpu.yaml` 的 "Multi-GPU Unit Tests" 每个 PR 都跑 `pytest tests/unit_tests/gpu -m multi_gpu`，整个 job `timeout: 30` 分钟；本测试标了 `multi_gpu`。
  - **修复：新提交 `96233d5c8` "tests: compile flex without max-autotune in the dynamic CP GPU test"**（只改测试，+13 行）：私有 helper `_compile_flex_without_autotune` 照 `set_determinism`（`torchtitan/distributed/utils.py`）关掉 `inductor_configs` 的 `max_autotune` 和 `coordinate_descent_tuning` 并重新编译 `_compiled_flex_attn`，两个测试方法开头各调用一次（测试在 spawn 出来的子进程里跑，`setUp` 不在子进程执行）；一行注释说明原因。上游 `tests/unit_tests/cpu/test_fused_mla_override.py` 也这样关 autotune。不直接调 `set_determinism`：它还会打开 `use_deterministic_algorithms`（默认不只警告），tower 的双线性插值反向在 CUDA 上没有确定性实现。black 22.12、usort、pyflakes 干净。
  - **H100 上（空的 inductor 和 Triton cache，和 CI 冷启动一样）：2 passed in 80.9 s**（第二个用例 47.9 s，第一个 30.2 s）。跑的文件和提交的逐字节相同。
  - `cpmm_review1` 快进到 `96233d5c8`（fork）。用户说"推"后，PR 分支 `k3_cp_mm` 也快进到 `96233d5c8`（不是 force），旧 head `6317c5538` 备份在 `backup/k3_cp_mm_pre_20261008`。Test plan 已改成 "on 4 H100 (2 passed in 81 s from a cold cache)"，描述的是 `96233d5c8`。

- **10-07 按 DEP 方式重构（用户："直接改，review和pr分支都做，然后改body，及时推送diff"）：** `k3_cp_mm` = `cpmm_review1` = `6317c5538`，在 `9b03b4af1` 之上快进两个提交：
  - `e094d6b9a` "kimi_k3: dynamic CP as a vision_cp package"：新包 `kimi_k3/vision_cp/`，包括 `plan.py`（原 `vit_cp_plan.py`，内容不变）、`attention.py`（`VisionCPAttention`、`VisionCPLayout`、gather）、`encoder.py`（`MoonViTCPEncoder`，`_forward_split` 拆成 `_pack_inputs`、`_encode`、`_assemble_bank`）、`__init__.py`（`build_cp_subgroups`、`install_vision_cp`）。K3 目录只剩三处改动：`KimiK3VisionEncoder(MoonViTCPEncoder)`、flavor 换用 `VisionCPAttention`、`parallelize` 里一行 `install_vision_cp`。
  - `6317c5538`：GPU 测试开头的两行注释缩成一行。
  - **数值不变的证据（本机）：**
    - 原样搬过去的 9 个函数和方法，AST 和旧 head 一致，只差改名；`plan.py` 逐字节相同。
    - 拆开的 `_forward_split` 做了新旧对比：4 个 gloo 进程（CPU、fp32，两边都换成同一个稠密注意力），跑 GPU 测试的 4 种情形，新旧两棵树的输出和全部参数梯度 368 个张量逐位相同，并确认每种情形都走了切图路径（kit `kit_cpmm_2026-10-06/local/vision_cp_equiv/`；本机用桩代替 CuTeDSL，不进任何提交）。
    - CPU 规划测试 7 passed，black 22.12 和 pyflakes 干净。
  - **GPU 会话待跑：** `test_kimi_k3_vision_cp.py`（4 卡，2 passed）；任选一个 CP=2、阈值 128 的端到端格子，在同一份 cache 上和 `9b03b4af1` 比 20 步轨迹（应逐位相同）；pre-commit 的 pyrefly 只看改动文件。
  - 粘贴区改了：Summary 的文件和类名、Design 的 Sub-CP groups 一条、新增 Placement 一条、Test plan 的测试文件名。

- **为什么重写：** #4639 已以 `3f087cf15` 合入 main。v2 描述的是叠在 #4639 旧版上的 `923f8bd47`；新版是在 main 上重写的一个提交，设计改了两处（见 `CPMM_4380_PREP_2026-10-06.md`）：
  - tower 每个 micro-batch 只调用一次（旧版按图逐张调用，不开 PP 时各 rank 的 FSDP all-gather 次数会对不上，2 卡复现会卡死）；
  - 拆分的大图特征在整个 CP 组做一次 all-gather（旧版在别的子组上把这些大图整张重编一遍）。
- **PR 分支（10-06 晚，用户：审核没问题就推）：** `k3_cp_mm` = `9c648c4e2`，force push 前的 `923f8bd47` 备份在 `backup/k3_cp_mm_pre_20261006`。在 `7202a40e7` 上有四个单独的提交：`fcaaeb25f`（gather 放到 SPMD 类型检查外）、`f53f65a18`（删私有 helper 的 docstring）、`2d93f8016`（子组改为 unflatten CP mesh）、`9c648c4e2`（删规划模块没用到的部分，修报错信息，删复述断言的测试注释）。要粘贴本 body，并把标题里的 "[DO NOT review, pending rebase]" 去掉。
- **5060 上的结果**（不进 body）：在 `CPMM_4380_PREP_2026-10-06.md` 里；body 的 Results 等 H100。
- 粘贴区已检查：没有 we/our/us，没有破折号。
- **Results（10-07 H100）：** box 115.124.123.239，4 卡 H100，驱动 560，所以 torch 是 cu126 的 0906 nightly，加 kit 的 shim（`param_dtype_override_fn`、DistMoE 的两个 pipelining 名字）。测的是 PR head `6317c5538` 和 main `948d65c86`，和 GitHub 上的 PR 一致。kit `kit_cpmm_2026-10-06`（`run_matrix2.sh` 预热 100 步，`table_4380.py` 出表），结果在 box 的 `/workspace/h100_cpmm`。bank 替换：swaplog 里 200 次调用，预热和正式各 100 次。
- **10-07 Design 改成嵌套 bullet**（用户："design又是大段长段落，改成嵌套bullet points的格式"）：内容和原来三段相同，另外按代码补了三条：行带尾部的 rank 补齐、子组数取 CP 大小不超过大图张数的最大因子并按长度优先填、补齐的 query 在 projector 之后丢掉。每条一行。
- **10-07 rebase（用户："现在rebase 4380 把除了h100数值之外的其他的事情都做了"）：** 五个提交原样搬到 main `948d65c86`（中间 21 个上游提交，6 个碰 K3 文件，没有冲突），PR 自身 diff 的 patch-id 不变。旧 head `9c648c4e2` 备份在 `backup/k3_cp_mm_pre_20261007`。
- **CI 覆盖：** main 的两个 h100 K3 CP 格子在 main 上建 optimizer 就失败（DistMuon layout，10-07 nightly 证实）。修复在单独的分支 `k3_cp_muon_layout`，body 草稿 `PR_BODY_K3_CP_MUON_LAYOUT_2026-10-07.md`。那个 PR 开出来以后，可以在本 body 的 Test plan 加一句："The h100 cells `kimi_k3_mm_allgather_kv_cp` and `kimi_k3_mm_ulysses_cp` split their first 256-patch image at step 5; they train once #<修复 PR 号> lands."

- **10-08 H100 新增（用户："把昨天dynamic cp剩下的H100实验还有特殊的高分辨率测试都跑了，得看到CP维度在超清大图下"）：** Results 末尾加了 tower 的显存和时间表，六行，外加五条说明。数据出自 box 115.124.123.240 -p 30797，PR head `6317c5538`，main `948d65c86`；完整三种 AC 的表和端到端的数在 `CPMM_4380_PREP_2026-10-06.md` 的 10-08 一节。粘贴区约 1290 词（按 wc 计，含表格）。
- **10-08 压缩（用户："1确定压缩 只压缩词语 表格数据不压缩"）：** 两张表逐字节不变；文字从 1024 词压到 783 词（含表格从 1301 到 1060），其中已经算上新加的两条阈值说明（packing 下快 11% 到 23%、显存少 38% 到 58%；单张 256 到约 9000 patch 的图慢约 5%，见 logbook 10-08 阈值一节，默认 256 不改）。

--- PR 4380 body v3: PASTE BEGIN ---

## Summary

Adds dynamic context parallelism for the Kimi K3 vision tower (report sec 5.2.3): under CP a large image is split by rows over a sub-CP group that gathers its keys and values, and several large images spread over equal sub-groups.

- `vision_cp/plan.py`: which images split, over which sub-groups, and each rank's rows.
- `vision_cp/attention.py`: `VisionCPAttention`, `VisionAttention` plus the sub-group key and value gather.
- `vision_cp/encoder.py`: `MoonViTCPEncoder`, the base of Kimi K3's encoder.
- `MoonViTCPEncoder.Config.dynamic_cp_min_patches` turns dynamic CP on: `None`, the default, keeps main's path, and the two h100 CP cells set 256.
- `vision_cp/__init__.py`: `install_vision_cp` builds the sub-CP groups from `KimiK3Model.parallelize`.

## Design

- Scope: main's CP path encodes every image on every CP rank, and this PR changes only how that vision bank is computed.
- Which images split:
  - An image with at least `dynamic_cp_min_patches` patches whose height takes whole merge blocks.
  - Every other image stays whole on every rank.
- How an image splits:
  - Bands are whole merge blocks of rows, the same rows in every frame.
  - Trailing ranks pad to the first rank's band.
  - Large images fill equal sub-groups longest first, as many as the largest divisor of the CP size not above their number.
- One tower call per micro-batch:
  - The tower runs once per micro-batch, as on main, so FSDP issues the same collectives on every rank.
  - The packed stream holds the whole images, then this rank's bands.
- Attention:
  - Only bands gather keys and values, over their sub-group.
  - Padding keys match nothing.
  - Padded queries are dropped after the projector.
- Vision bank:
  - One all-gather over the CP group returns the split images' merged tokens, so the bank is main's.
  - With no large image, the tower runs main's path.
- Sub-CP groups:
  - `install_vision_cp` unflattens the CP mesh once per divisor of the CP size.
  - Building is collective, so every rank builds them and reads the switch from the model config, which pipeline stages without the tower also hold.
- Placement:
  - The classes extend only `VisionAttention` and `MoonViTEncoder`, so other MoonViT models can reuse them.
  - Kimi K3 changes in three places: its encoder's base class, its flavors' vision attention and `parallelize`.

## Results

Kimi K3 debug model on 4x H100 (torch 2.15.0.dev20260906+cu126), main `948d65c86`, CP=2, 100 steps, seed 42, deterministic, typechecking on as in #4639's cells, one warm cache per block of cells. Training uses AdamW (lr 8e-4), since DistMuon did not build under CP on that main (#5191 has fixed it since). These runs predate the rebase onto `01b1b69e3`, which leaves this PR's diff unchanged. This PR's rows run with dynamic CP on at 256, the last one at 128 so every image splits. cc12m-test holds 32 samples and each step takes one, so the columns stop at step 20, before any sample repeats. Cells are loss / grad norm.

| Configuration | Step 1 | Step 10 | Step 20 |
|---|---|---|---|
| main, all-gather or Ulysses | 8.19922 / 2.5781 | 3.68411 / 2.5312 | 3.27230 / 3.1562 |
| main, all-gather, second run | 8.19922 / 2.5781 | 3.68411 / 2.5312 | 3.27230 / 3.1562 |
| this PR, all-gather or Ulysses | 8.19922 / 2.5781 | 3.68293 / 2.5625 | 3.26093 / 3.1094 |
| this PR, every image split (threshold 128) | 8.19896 / 2.5781 | 3.67482 / 2.5469 | 3.25868 / 3.1250 |

- All-gather and Ulysses are identical for 100 steps, on main and on this PR.
- Main's second run is identical to its first for 100 steps.
- This PR matches main bitwise through step 4.
- Step 5's micro-batch holds the first 256-patch image, the first one this PR splits.
- Splitting every image moves step 1 by 2.6e-4, the split tower's bf16 rounding: with main's vision bank swapped in, the run matches main bitwise for 100 steps.
- In fp32 the split tower matches the whole tower within 2e-6 forward and 1e-5 in gradients, in the GPU test and at CP 2 and 4 on 2016 and 4032 px images and 16 video frames.
- With CP=1 this PR matches main bitwise for 100 steps.

Vision tower alone (Kimi K3's: 27 layers, dim 1024, random bf16 weights), forward and backward over one micro-batch's images on 4x H100, with the recipe's selective checkpointing unless noted. Every CP rank holds all the images, so main measures the same at CP 1, 2 and 4. This PR's columns and the training figures below run with dynamic CP on at 256. Cells are peak memory per GPU / time.

| Images (patches) | main | this PR, CP=2 | this PR, CP=4 |
|---|---|---|---|
| one 2016 px image (20736) | 15.1 GiB / 0.81 s | 10.5 GiB / 0.44 s | 7.5 GiB / 0.27 s |
| one 4032 px image (82944) | 57.6 GiB / 11.1 s | 39.3 GiB / 5.6 s | 27.1 GiB / 2.9 s |
| one 4032 px image, no checkpointing | out of memory | 47.8 GiB / 5.5 s | 31.4 GiB / 2.8 s |
| one 4032 px image, full checkpointing | 9.4 GiB / 14.0 s | 5.8 GiB / 7.0 s | 3.9 GiB / 3.6 s |
| four 2016 px images (82944) | 57.6 GiB / 3.2 s | 29.7 GiB / 1.6 s | 15.8 GiB / 0.81 s |
| 16 frames of 448 px in 4-frame items (16384) | 12.0 GiB / 0.22 s | 6.5 GiB / 0.15 s | 3.7 GiB / 0.15 s |

- Figures include 1.7 GiB of weights and gradients.
- Flex attention compiles without max-autotune on both sides.
- Several images or video items go whole to one-rank sub-groups, so memory falls as 1/CP.
- A split image keeps its whole keys and values on each rank for the attention backward, so its memory falls as 1/CP only under full checkpointing.
- Training the debug text model with this tower (recipe settings, AdamW) on 2016 px images peaks at 19.1 and 17.6 GiB on main at CP=2 and 4, and 15.1 and 10.8 GiB with this PR.
- With samples packed into 8192-token micro-batches, this PR trains 11 to 23% faster than main with 38 to 58% less memory (cc12m-test at native size and at 1008 px, CP 2 and 4).
- A micro-batch holding a single image of 256 to about 9000 patches trains about 5% slower, since its per-layer gathers cost more CPU time than the split saves.
- On cc12m-test resized to 1008 px, this PR differs from main from step 1 (8.20263 against 8.20187).
- With main's vision bank swapped into that run, it matches main bitwise for 20 steps.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_vision_cp_plan.py -q` (7 passed).
- `pytest tests/unit_tests/gpu/test_kimi_k3_vision_cp.py -q` on 4 H100 (2 passed in 81 s from a cold cache): split against whole tower in fp32 for one image over the CP group, two images over two sub-groups, one image per rank and a video whose last rank holds only padding, and two data-parallel groups splitting different numbers of images under FSDP.
- On 2 H100s, the CP=2 all-gather run (seed 42, 10 steps) with dynamic CP off, the default, matches main bitwise, and with it on at 256 matches the revision the tables measure, bitwise.
- The h100 cells `kimi_k3_mm_allgather_kv_cp` and `kimi_k3_mm_ulysses_cp` turn dynamic CP on at 256 and split their first 256-patch image at step 5.
- Both cells train since #5191, which this branch includes.

## Relation to earlier revisions of this PR

The earlier revision sat on the first #4639, called the tower once per image and re-encoded other sub-groups' large images. This one calls it once per micro-batch.

--- PASTE END ---
