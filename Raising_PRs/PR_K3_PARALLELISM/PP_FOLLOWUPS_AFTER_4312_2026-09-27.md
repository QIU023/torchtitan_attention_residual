# 4312 合并之后：PR A、DEP rebase 与审核，#4656 / #4780 的处理（2026-09-27）

用户："rebase，4312合并了，下一步是pp review optimize和DEP，审核这两个分支和body当前情况，还有 #4656 r4099737626、#4780 review 5299962900（只剩tianyu comment里面第二点，第一点拆到4881并且应该快merge了，第三点留给Jessica）这两个PR如果要合并考虑的话，如何处理？"

## 4312 怎么合的

- squash 成 `e033f7517`（09-26 21:55 UTC），合并时的 head 就是 `ffdd169ef`，两者树完全相同。
- main 之后又进了 #4617（CP 输入与元数据分片解耦），现在是 `f35966713`。#4617 改了 92 个文件，包括把 `ParallelismConfig` 挪到 `torchtitan.config.parallelism`、把 K3 的 `attention_masks` 类型改成 `HybridAttentionMetadata`。

## PR A（pp cache optimize）

- `pp_review_optimize` = `d445b2f7f`，main 上 4 个提交（V4 三个加 PR A），已推到 fork。还没有 PR 分支。
- **rebase：** 一处冲突在 `model.py` 的 `forward` 签名：#4617 的 `HybridAttentionMetadata` 与 PR A 的 `blocks_TD: list[torch.Tensor]`，两边都保留。PR 自己的 diff（+/- 行）rebase 前后完全相同。
- **检查：** pyflakes、flake8、ufmt 干净；4 个测试文件 68 个通过；5060 组合格 `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4` 10 步 rc=0，loss 8.10820 → 3.48003（本地 torch 兼容补丁）。
- **审核：**
  - 新增注释 8 行，都是代码表达不了的约束（顺序、就地相加的前提、谁是最后一个读者），不用删。
  - `sharding.py` 删掉了 layer 0 边界上对 stack 的布局转换：列表一开始是空的，第一个 block 就是 layer 0 转换后的输入。
  - 仍在重写 torch 的私有方法（接收缓冲、send 的 wait），body 写明"torch 提供之后删"；torch issue 还没提（计划书 §6 第 5 条）。评审可能会问这一点。
- **body：** `PR_BODY_PP_CACHE_OPT_2026-09-27.md`（新写，从 #4765 的 body 里拆出来）。
  - 加了研究分支的 CPU 计数：不开 PP 时，stack 写法让 24 层、6 个 block 的模型有 21 个 [T, D] 活到反向，列表写法是 6 个，FullAC、selective、region AC 都一样。这就是对 shuhuayu r4099737626 的回答。
  - 5060 同 base：main 12.00 / 11.09 → PR A 7.11 / 6.23 GiB（最大 / 均值），10 步逐位一致。
- **#4765、#4764 的 review 分支**（`7c0f5cd3c`、`e6241b78b`）还叠在旧的 PR A（`7ae870508`）上，之后要重叠。

## DEP（#4381）

- `dep_review1` = `k3_pp_mm` = `31f372593`，main 上 4 个提交。
- **rebase：**
  - `model.py` 冲突：main 的 #4777（修多模态 FSDP 卡死）让没有图像的 micro-batch 在 forward 里造一张假图跑视觉塔，再用零依赖接回；DEP 让 forward 接收预先编码好的 `vision_embeds`。解法：先用 `vision_embeds`，没有才就地编码，然后原样走 #4777 的假图分支。main 的 #4777 代码一行没改。
  - #4617 挪走了 `ParallelismConfig`，DEP 的测试改了导入（折进测试那个提交）。
- **#4777 是否要求 DEP 另外处理（用户问"4777不是修复了hang吗？为什么需要我们再加假图"）：**
  - 我先以为要：DEP 的缓存跳过没图的 micro-batch，有图、没图的 DP rank 的视觉塔 all-gather 一个在放置点、一个在 forward 里，会错开。于是加过"在放置点编码假图"。
  - 实测推翻了这个判断：用 #4777 自带的 `set_rank_conditional_image_presence`（偶数 DP rank 只有文本），在 dp2 × pp4 的 vit_dep 格子上跑 4 步（本地 recipe `scratchpad/dep_mixed/dep_mixed.py`，不进分支）：
    - 不加假图：rc=0，4 步都跑完；
    - 我加的版本：第 2 步崩了。缓存构造时记下的设备是 meta（模型那时还没实体化），假图造在 meta 上。CPU 单测里的假视觉塔参数一开始就在 CPU 上，没测出来。
  - 原因（代码核实）：开 PP 时 `MultimodalModel._apply_fsdp` 不单独包视觉塔，它的参数在第一个 stage 的根 FSDP 单元里，`tok_embeddings` 是另一个单元。PP 下默认 `reshard_after_forward=False`，两个单元每步各 all-gather 一次。缓存提前编码时按 `modules()` 顺序 unshard（先根、后 `tok_embeddings`），和没图的 rank 在 forward 里的顺序相同。
  - 所以假图那个改动整个撤掉了（模型、缓存、测试、提交说明都恢复），DEP 只保留 rebase 必需的部分。
  - `fsdp_reshard_after_forward=always`（每个 micro-batch 都 all-gather）：同样的混合数据 4 步跑完，rc=0，没有超时。
- **注释修剪：** 按"默认不加注释"规则，14 处多行注释改成一行约束或删掉（讲设计怎么选的那些，比如"包一层而不是继承"、recipe 里"这个格子就是为了跑这段代码"）；starved/exhausted 的含义挪进字段 docstring。
- **检查：** pyflakes、flake8、ufmt 干净；`-k "kimi_k3 or pipeline_parallel or cli or integration_test"` 120 passed、1 skipped；optimizer 的测试 24 passed；5060 vit_dep 格 10 步 rc=0，loss 7.96843 → 6.13097，DEP 的计数和 09-26 相同。
- **body：** `PR_BODY_PP_MM_v4_2026-09-27.md`：去掉"叠在 #4312 上"；Design 写明没图的 micro-batch 走 #4777 的就地路径以及为什么不错开；测试数更新；标题前缀 `[DO NOT review, pending K3 text PP merging]` 可以去掉（用户改）；bubble 开关对比那条还是 09-22 旧 base 的结论，要重测。

## #4656 / #4780

研究分支的方案在 `PR32_torchtitan_kimi_k3_attnres_recompute/PLAN_4656_4780_2026-09-27.md`：

- **保留 #4656、关闭 #4780：** #4780 在 tianyu 三点里只剩第 2 点（torch_remat 代替 autograd Function），这正是 #4656 的内容；shuhuayu 的 review 在 #4656 上；用户 09-25 已在 #4780 上公开说过重算由 #4656 解决。这推翻了 09-24 "留 #4780" 的选项（那是在用户回复和 shuhuayu review 之前写的）。
- **顺序：** #4881（已批准；合并前把 body 里那个不存在的测试文件删掉）→ PR A → #4656 rebase 到 PR A 之上，只保留调用处 checkpoint 和 `KimiK3Model.parallelize` 里的标记，去掉 Configurable 那层（第 3 点归 Jessica），签名改成列表，body 的数字在新树上重测。
- **shuhuayu r4099737626 对不对：** 对，而且问题在 stack 本身，不是 #4656 的 checkpoint（CPU 计数 21 对 6，所有 AC 模式都一样）；PR A 的列表修掉它。
- **回复草稿**（note 里）：回 shuhuayu 的那段要等 PR A 开出来、把 `#PR_A` 换成编号；回 tianyu 并关闭 #4780 的那段等 #4881 合并后贴。
- **退路：** 如果 PR A 的 review 拖太久，#4656 可以先合，回复里说明这份开销 FullAC 下本来就有、由 PR A 修；需要用户点头。

## 推送状态

- `pp_review_optimize` `d445b2f7f`：已推 fork。
- `dep_review1` `31f372593`：已推 fork（原 `232834a4d`）。
- PR 分支 `k3_pp_mm`：按用户的话同步到 `31f372593`（draft），旧 head 备份为 `backup/k3_pp_mm_pre_20260927`；#4381 现在 4 个提交、12 个文件，mergeable。body v4 等用户粘贴。

## 更正（用户指出 stack 是 eager AttnRes 的）

- shuhuayu 在 #4656 r4099737626 指的是 eager AttnRes 的载体 `block_residual_TND`（`KimiK3TransformerBlock.forward` 每开一个 block 就 `torch.cat`），不是 PP cache 的 stack；21 对 6 是不开 PP 的 eager 计数。PR A 的 PP 内容和这条评论无关，上面"这就是对 shuhuayu 的回答"的说法收回。
- 列表载体可以放进 #4656，已合并的 stage 在调模型处小改即可（`unbind` 视图传入，按列表下标提交和路由；读代码得出，未实现）。方案改为 #4881 → #4656（checkpoint 加 eager 列表载体）→ PR A（只做 PP）。详见 `PR32_torchtitan_kimi_k3_attnres_recompute/PLAN_4656_4780_2026-09-27.md` 的"更正"一节。
- 如果按这个方案走，`PR_BODY_PP_CACHE_OPT_2026-09-27.md` 的 Summary 第一条和 Design 第二段（列表、21 对 6）要从 PR A 的 body 里删掉。

## 按更正重排：#4656 带列表载体，PR A 叠在它上面（09-27 晚）

- `attnres_review1` = `f14d681f4`（#4656 的新 review 分支，main 上两个提交：`aa6d9fedc` 列表载体加 stage 适配，`f14d681f4` 调用处的 torch_remat 重算），已推；旧 head `38fcdde4a` 备份为 `backup/attnres_review1_pre_20260927`。PR 分支 `k3_ac_reuse_attention` 没动。
- `pp_review_optimize` = `e8d0a4aec`，由用户压缩后的 `be9e2fa69` 重放到 `f14d681f4` 之上，已推。PP 的 5 个文件与 `be9e2fa69` 完全相同（`be9e2fa69` 删掉的 docstring、`_placeholder` 和改回的 `cache.py` 都保留）；`model.py`、`sharding.py` 取 #4656 的，PR A 不再改它们。
- body：#4656 新写 `PR_BODY_4656_2026-09-27.md`；PR A 改好 `PR_BODY_PP_CACHE_OPT_2026-09-27.md`（去掉列表载体，注明叠在 #4656 上）。两份的 Results 和测试数都等 5060 重测：#4656 对 main 的一致性和显存（`scratchpad/acr/`），PR A 对 #4656 的生产切分显存和计时（`scratchpad/lb6/`），两条分支的组合格冒烟。

## 两份 body 的数字（09-27 晚，8 × 5060 实测）

- **#4656**（`kit_4656_2026-09-27/`）：debug model、一卡一格，与 main `f35966713` 相比，none、selective、full 三种 AC 下 loss 和 grad norm 10/10 相同；AC 关时显存 0.66 → 0.55 GiB，tps 1523 → 1362（cache 本身的噪声在 4% 左右），selective 和 full 显存不变。预热 cache 的 token 扫描（AC 关、一个 micro-batch、3 步）：2048 / 4096 / 8192 token 分别 1.89 → 1.42、3.51 → 2.62、6.71 → 4.95 GiB，loss 3/3 相同。组合格 10 步 rc=0。干净导出树上 4 passed、71 passed。
- **PR A**（`kit_pp_lowerbound_2026-09-26/results/s6_*`）：生产切分（93 层、block 12、pp8 × vp4、dim 2048、seq 2048、M16、FullAC），#4656 对 #4656 加 PR A，同一条 cache 血缘：每个 rank 峰值的最大 / 平均 11.47 / 10.80 → 7.12 / 6.23 GiB，10 步逐位一致；第 8 步窗口 25.10 → 24.94 s，计算 7.45 → 7.38 s。组合格 10 步 rc=0。干净导出树上 68 passed（加上 #4656 的测试文件是 72）。
- 带本地 torch 兼容补丁的 worktree 里，core 的 `test_pipeline_parallel.py` 有 5 个 `KeyError: 'max_active_stages'`，是补丁过滤参数造成的；两棵干净树上都不出现。
- 补丁已从 `wt_attnres`、`wt_ppopt` 撤掉，临时的 `wt_main_f359` 已删除。
