# Overnight 结果（2026-09-29 夜，计划见 `OVERNIGHT_PLAN_2026-09-29.md`）

用户确认计划（"4个都可以"，"把goal全部完成 开始"）。本机 8 × RTX 5060 Ti，H100 冻结。套件在 `kit_overnight_2026-09-29/`。5060 上的数字只进 logbook。

## T1a 新 venv

- `/workspace/venv_0928`：torch `2.15.0.dev20260928+cu130`、torchvision 同日、Triton `3.8.0+gitc01b6774`、cutlass DSL 全家 4.6.2、attn-gym 0.0.13、mooncake 0.3.13.post1、transformers（`--no-deps`）、ufmt 2.3.0 / black 22.12.0 / usort 1.0.5（仓库 pre-commit 钉的版本；新版 black 会误报）。脚本 `setup_venv_0928.sh`。
- 核对：`pipeline_per_edge_p2p` 在，`_recv_buffers.py` 在（torch#196463），`unshard_lookahead` 在，BFX9 能开。main `5dc97a3e7` 的 PP 单测 67 passed，不用兼容补丁。
- mooncake 的 wheel 链的是 `libcudart.so.12`，cu130 的 venv 里没有，要补 `nvidia-cuda-runtime-cu12==12.8.*`，否则 #4764 的两个用例会跳过。

## T1b PR A 重建

- `pp_review_optimize` = `4ddf8e912`，#4656 `5d469fdf3` 之上一个提交，6 个文件 +493/−274。内容 = 本地 dev `1777ad806` 去掉接收覆盖（H100 上验证过的 `dev_send_only_probe.patch`），再把 `_collect_into` 的两行注释压成一行。旧 head `85eefa54b` 在 `backup/pp_review_optimize_pre_20260929`。
- 逐行审过 diff：去掉四个接收覆盖和两处缩缓冲的循环后，`_make_tensor_from_meta` 的导入一起删掉，没有别的死代码；新增 12 行注释和 docstring 都是约束或一行"做什么"。
- 检查：pyflakes 干净，ufmt（钉的版本）干净，新 venv 上五个测试文件 71 passed。提交信息没有 trailer，没有跨仓库引用。

## T1d 5060 上 #4656 对新 PR A

设置：debug model 放宽到 dim 2048，16 个 micro-batch × 2048 token，c4 文本（每行最多 2047 个 token，一个 micro-batch 一行），FullAC，AdamW，seed 42，确定性，每格 20 步，第 5 步逐动作记账。每个布局一份预热缓存，两棵树在不相交的 GPU 上同时跑。探针 `pra_bound/probe_bound.py`，脚本 `pra_bound/run_bound.sh`，表 `pra_bound/tab_bound.py`，日志 `results_bound/`。下界在探针里按实际执行的调度算："紧界"是每个 block 在带它进来的 stage 反向时释放，"论文界"是一个 micro-batch 的 block 留到它在这个 rank 上最后一次反向。

**逐位和峰值（每个 rank 第 10 步的峰值 allocated，GiB）：**

| 布局 | 20 步逐位相同 | #4656 各 rank | PR A 各 rank | 省 |
|---|---|---|---|---|
| pp4 × vpp2 | 20/20 | 7.80 / 7.31 / 7.13 / 5.70 | 7.16 / 6.30 / 6.04 / 4.76 | 0.64 到 1.09 |
| pp4 × vpp4 | 20/20 | 9.51 / 6.87 / 7.94 / 7.68 | 7.98 / 5.59 / 6.35 / 5.80 | 1.27 到 1.88 |
| pp2 × vpp4 | 20/20 | 12.60 / 10.56 | 11.37 / 9.26 | 1.23 到 1.31 |
| dp2 × pp2 × vpp2 | 20/20 | 9.92 / 9.92 / 8.12 / 8.12 | 9.17 / 9.17 / 7.29 / 7.29 | 0.75 到 0.83 |
| pp2 × vpp2（expandable segments） | 20/20 | 12.39 / 9.80 | 11.63 / 8.97 | 0.76 到 0.83 |

- pp2 × vpp2 第一次跑（默认分配器）时，#4656 在第 8 步 OOM（前 7 步和 PR A 相同）：已分配 10.05 GiB，另有 3.87 GiB 保留未用（碎片），16 GB 的卡放不下；PR A 同一格 20 步跑完。两棵树都加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 重跑（`results_bound_pp2vp2_es/`），上表是重跑的数。expandable segments 只影响 reserved，不影响 allocated。

**各 rank 峰值那个动作上的 block 占用（第 5 步，GiB）：**

| 布局 | rank | #4656 block 合计（其中只被 send 扣住） | PR A block 合计（其中 store） | 紧界 | 论文界 |
|---|---:|---|---|---:|---:|
| pp4 × vpp2 | 0 | 0.53（0.39） | 0.17（0.09） | 0.09 | 0.16 |
| pp4 × vpp2 | 1 | 0.85（0.64） | 0.19（0.12） | 0.12 | 0.19 |
| pp4 × vpp2 | 2 | 0.75（0.55） | 0.14（0.09） | 0.09 | 0.16 |
| pp4 × vpp2 | 3 | 0.44（0.31） | 0.13（0.09） | 0.09 | 0.16 |
| pp4 × vpp4 | 0 | 1.22（0.89） | 0.28（0.14） | 0.14 | 0.17 |
| pp4 × vpp4 | 1 | 0.91（0.59） | 0.25（0.12） | 0.12 | 0.16 |
| pp4 × vpp4 | 2 | 1.44（0.97） | 0.27（0.16） | 0.16 | 0.20 |
| pp4 × vpp4 | 3 | 1.00（0.66） | 0.20（0.12） | 0.12 | 0.16 |
| pp2 × vpp4 | 0 | 0.73（0.61） | 0.13（0.06） | 0.06 | 0.08 |
| pp2 × vpp4 | 1 | 0.80（0.64） | 0.13（0.08） | 0.08 | 0.11 |
| dp2 × pp2 × vpp2 | 0、1 | 0.52（0.48） | 0.06（0.03） | 0.03 | 0.08 |
| dp2 × pp2 × vpp2 | 2、3 | 0.39（0.34） | 0.06（0.05） | 0.05 | 0.08 |
| pp2 × vpp2 | 0 | 0.52（0.48） | 0.06（0.03） | 0.03 | 0.06 |
| pp2 × vpp2 | 1 | 0.39（0.34） | 0.06（0.05） | 0.05 | 0.08 |

- **PR A 的 store 在每个布局、每个 rank 的峰值时刻都正好等于紧界**，也就是论文 §4.1 说的每个 block 在 rank 上只存一份，而且比论文的释放点更早。
- **对照的一方是 #4656 的树，但多出来的显存不是 #4656 的**：#4656 只改模型内部（列表载体和 attention residual 的 checkpoint 重算），PP stage 还是 main 里 #4312 那一版。main 的 PP stage 在峰值时刻占着紧界的 5 到 13 倍，大头是整步都被 send 扣住的张量，其次是每个 stage 进门拼出来的 stack 和模型输出的 stack。用 #4656 作对照，只是因为 PR A 叠在它上面。
- **PR A 高出紧界的部分是还没 wait 的前向 send。** PR A 在本 stage 反向那个 micro-batch 时才 wait。探针补丁 `pra_bound/early_fwd_wait_probe.patch` 改成在同一个 rank 上下一个虚拟 stage 对同一个 micro-batch 做前向时 wait：那个前向的输入要经过接收方对这个 micro-batch 的前向，所以接收方一定已经用完。最后一个虚拟 stage 仍在反向时 wait。它在 71 个单测上全过；GPU 上的对比见 T1g。
## T1g 前向 send 提前 wait 的探针（不并入 PR A）

- 补丁 `pra_bound/early_fwd_wait_probe.patch`：前向 send 在同一个 rank 上下一个虚拟 stage 对同一个 micro-batch 做前向时 wait（那个前向的输入要经过接收方对这个 micro-batch 的前向，所以接收方一定已经用完），最后一个虚拟 stage 仍在反向时 wait。71 个单测全过。
- 5060 上对 PR A（pp4 × vpp2、pp4 × vpp4，同样设置，20 步，`results_bound_early/`）：20/20 逐位相同；**各 rank 的整卡峰值完全不变**（7.16 / 6.30 / 6.04 / 4.76，7.98 / 5.59 / 6.35 / 5.80 GiB）；峰值动作结束时的 block 占用从 0.13 到 0.28 降到 0.09 到 0.19 GiB，基本等于紧界加在途的接收缓冲。
- 原因：rank 的峰值落在它最后一个虚拟 stage 的反向里，那个 stage 的前向 send 两种写法都在反向时 wait；更早 stage 的 send 在峰值时刻已经放掉，或者和前向缓存里本来就要留到反向的输出是同一块存储。
- 结论：PR A 不改。block 占用在峰值时刻已经是紧界加在途的一次传输。

## T1e body 和 torch issue

- PR A body v3：`PR_BODY_PP_CACHE_OPT_v3_2026-09-29.md`。Summary 引论文 §4.1 "each block is stored exactly once across all V virtual stages"；Design 三段：一个 block 为什么在 4312 的 stage 里存了 1 + k 份、为什么在带进来的 stage 反向释放、send 早等为什么是释放生效的前提（接收缓冲交给 torch）。Results：Pending (H100)。
- torch issue 草稿 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md` 改成只提 send；接收缓冲写明已由 #196463 解决；原来的 5060 测量按规则拿掉，只留两卡复现。

## T1f #4765、#4764 重叠

- 重放到新 PR A 上没有冲突：#4765 = `019462171`，#4764 = `2a719e512`、`c4afb61f4`。pyflakes、ufmt 干净。新 venv 上 #4765 76 passed，#4764 90 passed（和之前的数一样）。
- body 的 Relation 改成叠在 #4656 和 PR A 上，不再提 `1777ad806`。
- **5060 冒烟**（pp4 × vpp2，dim 2048，16 × 2048，FullAC，6 步，一份缓存，`smoke_4765_4764/smoke.sh`）：五格全部 rc=0，第 6 步 loss、grad norm 都是 5.08641 / 11.3750。各 rank 峰值（第 2 到 5 步，GiB）：什么都不开 6.85 / 6.31 / 6.05 / 4.78；#4765 offload all 6.78 / 6.26 / 5.97 / 4.76；#4764 planned 6.78 / 6.26 / 5.99 / 4.78；balance（tcp）6.78 / 6.26 / 6.02 / 4.78；planned 加 balance 6.78 / 6.26 / 6.01 / 4.78。这个宽度下峰值主要是静态显存，它们能挪的东西少，和 09-26 的结论一致。
- **推送（用户确认）：** `k3_pp_offload` = `pp_offload_review1` = `019462171`，`k3_pp_balance` = `pp_balance_review1` = `c4afb61f4`，force-with-lease；旧 head 在 `backup/pp_offload_review1_pre_20260929` = `56f4cd3b0`、`backup/pp_balance_review1_pre_20260929` = `0a9034257`。upstream 的 `refs/pull/4765/head`、`refs/pull/4764/head` 已经是新 head。body 的 Relation 改了，需要你在 GitHub 上替换。

## T2 DEP 的文本 / 视觉计算比例

**T2a、T2b（子任务，logbook `000edaf`，文档 `DEP_RATIO_2026-09-29.md`）：**
- 口径：一次编码前向对一个文本 stage 前向的比例 R_f，加上各 micro-batch 之间图数的差别。
- K3 发布配置的推算：PP16 × VP2、每个 micro-batch 4096 到 8192 个 token、一张 448 到 1024 px 的图，R_f 是 0.02 到 0.32，图多时按张数线性增加。放宽到 6144 的 debug model 塔和文本同宽，同一张图比 K3 重 5 到 8 倍（对 PP16 × VP2），用同样的 pp2 × vpp4 比是 19 到 31 倍。
- 数据集：HF `jackyhate/text-to-image-2M` 的 `data_1024_10K`（MIT，一万张 1024 × 1024），打成 `/workspace/dep_data/t2i1024_k4/`（2500 个样本，每个 0 到 4 张图，4.7 GB，重新打包逐字节相同）；本地 recipe 固定正方形 resize（会放大），`max_patches` 至少 5184。
- 三档：L1 224 px（R_f 约 0.21，K3 区间）、L2 448 px（约 0.87）、L3 1024 px（约 2.6）。

**T2c 5060 实测（`kit_overnight_2026-09-29/dep_ratio/`，结果 `results_dep_ratio/`）：**

单卡微基准，编码前向和一个文本 stage 前向的时间比（括号里是"编码前向加重算加反向"对"stage 前向加反向"的比）：

| dim，seq | stage 前向 ms | 224 px | 448 px | 768 px | 1024 px |
|---|---:|---|---|---|---|
| 1024，1024 | 14.46 | 0.26（0.59） | 0.34（0.62） | 0.67（1.29） | 1.45（3.05） |
| 1024，2048 | 16.08 | 0.23（0.39） | 0.22（0.41） | 0.60（0.92） | 1.30（2.17） |
| 2048，1024 | 17.32 | 0.23（0.34） | 0.46（0.65） | 1.53（2.23） | 3.11（4.86） |
| 2048，2048 | 单卡 OOM（微基准不开激活检查点） | | | | |

- 小图（224、448）在 dim 1024 时编码时间几乎一样，是启动开销主导；时间比比 FLOP 比高，和 H100 上 cc12m-test 的观察一致。
- 第一次跑微基准时把 rope 的 buffer 也转成了 bf16，`torch.polar` 报错；改成只转参数（FSDP 混合精度就是这样）后重跑。

4 卡三档冒烟（dim 1024，pp2 × vpp4 × tp2 × ep2，M4，10 步）：九格全部 rc=0。计划日志三档都一样：K2.5 是"3 次编码都在调度前，反向 3 次都在调度后"；bubble 是"2 次在调度前，1 次进空闲槽；反向 0 次进空闲槽，3 次在调度后"。

M16 核对（L2，3 步）：K2.5 13 次编码都在调度前；bubble 8 次在调度前、5 次进空闲槽；反向仍然 0 次进空闲槽、13 次在调度后。

- **M4 时空闲槽太少，**bubble 只能挪一次编码，和视觉占比无关；M16 时能挪 5/13。所以 H100 的格子要加 M16（已写进 `H100_NEXT_2026-09-30.md`）。
- **反向一次都没进空闲槽。** K3 报告说反向也排进气泡；09-27 的审计按塔在每个 rank 上的设计推算，pp8 × vp4、M16 时反向能进 13/16。我们的实现在测过的所有布局上都是 0。这是 DEP 规划器要查的问题，今晚照计划不改 DEP 代码，先记下。

## T3 MoonEP 审计（子任务完成，logbook `8fb189e`）

- **一个真 bug，已修：** `reduce_grad` 前少一道跨 rank 栅栏。MoonEP 的规约 kernel 直接远程读各 rank 的槽梯度，自己不带 barrier（`grad_reduce.py:467-470`），要调用方保证各 rank 都写完；重写版写完槽梯度马上规约，中间没有同步，别的 rank 可能读到没写完的梯度。H100 上 GPU 单测 2 passed 只说明那两次时序没撞上。修法：规约前在 EP 组上做一次单元素 all-reduce，排在同一条 stream 上。
- **一个稳健性问题，已修：** `Buffer` 改成 `explicitly_destroy=True`（和 DeepEP 一样），GC 时不再跑带设备同步和 barrier 的 `destroy()`。
- 其余调用逐个对过 MoonEP `33327eb` 的源码：dtype、`zero_copy`、复用 plan 时的 rank 同步、补零行、池的对齐、梯度都是新张量，没问题。
- **rebase 和推送（用户确认）：** `moonep_review1` = `1633dcd79`，upstream main `a182e530a` 上三个提交（两个原提交重放，加修复），没有冲突；旧 head `a505f74a8` 在 `backup/moonep_review1_pre_20260929`。CPU 单测 34 passed、13 subtests passed，ufmt 干净。PR 分支 `k3_moonep_seam` 没动。新 head 还没在真实 MoonEP 上跑过。
- **负载均衡的证据方案**（`kit_overnight_2026-09-29/moonep/`）：128 个专家、top-8，FSDP4 × EP4，standard 对 MoonEP，自然路由和偏向 rank 0 专家的路由各一格；记每层每个 rank 的负载 max / mean，以及 MoonEP 每次 dispatch 是否正好 S × K。假包版本排在本机 GPU 队列里；真机版本约 15 分钟，另加新 head 的 GPU 单测和 CI 格约 10 分钟。
- 审计文档 `DIFF_AUDIT_MOONEP_2026-09-28.md` §7；body `PR_BODY_MOONEP_v2_2026-09-29.md` 的 Test plan 和 Results 改成负载统计加步时，Requirements 加了栅栏和"LoRA 与 MoonEP 不能同时开"。
- **假包负载测试（本机 4 × 5060，只验证记账和流程）：** 第一次跑六格都在建模型时失败：探针包装 `RoutedExperts.forward` 时改了参数名，titan 的 local_spmd 按参数名找输入布局，报 `in_dst_shardings is missing entries for: ['counts_E', '_original']`。子任务只在 CPU 上核对过 recipe 和导入，没跑过前向。改成和真实 forward 同名的参数、用闭包传原函数后重跑，四格全部 rc=0：

| 格子 | 第 1 / 10 / 20 步 loss | 静态放置时每层"最忙的 rank / 平均"（均值，最坏） | MoonEP 每次 dispatch 是否正好 S × K |
|---|---|---|---|
| standard，自然路由 | 8.24211 / 3.73188 / 3.07722 | 1.55，2.25 | |
| MoonEP（假包），自然路由 | 8.24213 / 3.72677 / 3.06986 | 1.54，2.33 | 0 / 1280（假包不做规划，预期内） |
| standard，偏置 | 8.25292 / 3.69951 / 3.06190 | 3.14，3.91 | |
| MoonEP（假包），偏置 | 8.25329 / 3.70047 / 3.06361 | 3.14，3.91 | 0 / 1280 |

  - 128 个专家、top-8、EP4 时，刚初始化的自然路由就已经有 1.55 倍的失衡；偏置格接近 4 个 rank 的上限 4。真实 MoonEP 应当把每次 dispatch 都做成正好 S × K，这要到 H100 上看。

## 总结

- **目标一（PR A 回到 cache 下界）：** 完成。`pp_review_optimize` = `4ddf8e912`，#4656 之上一个提交；5060 上五个布局 20 步都和 #4656 逐位相同，各 rank 峰值省 0.64 到 1.88 GiB；PR A 的 store 在每个峰值时刻正好等于紧界（比论文 §4.1 的释放点更早），#4656 是紧界的 5 到 13 倍。#4765、#4764 重叠到它上面并推了 draft 分支。body v3、torch issue（只剩 send）已改。前向 send 提前 wait 的探针不降峰值，不并入。
- **目标二（DEP 比例）：** 完成探索。K3 的区间 R_f 0.02 到 0.32（推算）；数据集和三档 recipe 就绪；5060 上测了时间比；发现 M4 时空闲槽太少、反向从不进空闲槽两个问题。H100 的格子见 `H100_NEXT_2026-09-30.md`。
- **目标三（MoonEP 审计）：** 完成。找到并修了 `reduce_grad` 前缺的跨 rank 栅栏；`moonep_review1` = `1633dcd79` 已 rebase 到最新 main 并推送；负载均衡的证据方案在假包上跑通，真实数字等 H100。
- **下次 H100：** `H100_NEXT_2026-09-30.md`，三条线约 4 到 4.5 小时。

## 09-29 晚：整条线 rebase 到含 #4914 的 main（用户："b"，"后续PR A整条线是建立在4656 rebase main之后"）

- #4656 的 PR 分支不动（用户选 b）。rebase 后的 #4656 两个提交（`9c6904dea`、`4ae9422db`）只作为 PR A 分支底下的两个提交存在；和原来的 diff 一样，只解了 #4905 改名那处 import 冲突。
- PR A `ec8bb420a`：重放时唯一的冲突是 #4914 改过 docstring 的 `_pack_outgoing_delta`（本 PR 删掉了它），取本 PR 这边。同一个提交里改了 #4914 的页面：`PP_ATTN_RES_CACHE.md` 的表、1F1B 段，加一段 send 早等；stage 图只改图例里用"stack / 列"说的几行和 cache 关那栏的副标题（本地用 cairosvg 渲染核对过，图例都在框内）。
- #4765 = `e852c5ae5`，#4764 = `a6c0aef16`、`8545ea009`，重放没有冲突。
- 检查：四个 head 的 pyflakes、ufmt 干净，新 torch 上单测 70 / 71 / 76 / 90 passed，和换底前一样；加的行里没有 `parallel_dims` 这类旧名字。
- 5060 复核（pp4 × vpp2，20 步，`results_bound_rb/`）：rebase 后的 #4656 对 rebase 后的 PR A 20/20 逐位相同，loss、各 rank 峰值、block 记账和换底前一个数不差。
- 推送：`pp_review_optimize` = `ec8bb420a`，`pp_offload_review1` = `k3_pp_offload` = `e852c5ae5`，`pp_balance_review1` = `k3_pp_balance` = `8545ea009`；旧 head 都在 `backup/*_pre_20260929b`。upstream 的 `refs/pull/4765/head`、`refs/pull/4764/head` 已是新 head；`refs/pull/4656/head` 还是 `5d469fdf3`。

## 09-29 夜：PR A raise 前检查（用户："改"）

- **CI 格（`kit_overnight_2026-09-29/ci_cells/`，B200 那套里 K3 的三格，8 × 5060，torch 2.15.0.dev20260928，recipe 原样）：**

  | 格 | PR A `ec8bb420a` | PR A `5a8163a58` | rebase 后的 #4656 `4ae9422db` | main `d6810e4d6` |
  |---|---|---|---|---|
  | `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4`（8 卡） | rc=0，10 步 | rc=0，10 步 | rc=0，10 步 | 没跑 |
  | `kimi_k3_debugmodel_mm_muon`（2 卡） | rc=0 | 没跑 | rc=0 | 没跑 |
  | `kimi_k3_debugmodel_mm`（4 卡） | rc=1 | 没跑 | rc=1 | rc=1 |

  - `mm` 三处报的都是 `SPMD type mismatch on axis mesh_ep: tensor has P, expected I`。main 本身在同一环境同样失败，不是这条线引入的。
  - 这套 recipe 不设 seed，每格用自己的 cache：三次 PP 格的 step 1 loss 就不一样（8.17270 / 8.23765 / 8.21091），所以这些格只看 rc，不比 loss。PR A 对 #4656 的逐位对比仍是 T1d 和 rebase 复核那组（同 seed、同一份 cache）。
- **pyrefly 0.45.1（只读，`ci_cells/pyrefly/`）：** main 16 个错误（15 个是本机缺 `torch_checkpointing`、`rich` 的 import，1 个是 main 自己的 `int * None`）。PR A `ec8bb420a` 18 个，多出的 2 个都在 `stage.py`：`list(flatten_args(...))` 标成 `list[torch.Tensor]`，以及 Optional 上的 `.contiguous()`。修成 `5a8163a58` 后是 16，和 main 一样；只改类型，71 passed。
- **pre-commit（分范围，跳过 pyrefly 和 no-commit-to-branch）：** PR A 的文件全部通过，没有改文件。
- **#4765、#4764 重叠到 `5a8163a58`：**
  - #4765 = `6c9943a91`。`stage.py` 前向有一处冲突，取 #4765 那边，只去掉 `kwarg_tensors` 的 `list[torch.Tensor]` 标注。76 passed。
  - #4764 = `d248662a8`、`43ccdcc06`，没有冲突。90 passed。
  - pyrefly 比 PR A 多：#4765 自己 2 个（`ActivationStorage._device` 只读描述符被赋值；`register_stage` 收到 `Module | Tensor`），#4764 再多 3 个（TypedDict 键 `StorageBackend` 对 `HostBackend`；`group` 是 Optional；`PPMemoryController._device`）。两个 draft 转正式前要修。
- **推送：**
  - `pp_review_optimize` = `5a8163a58`；
  - `pp_offload_review1` = `k3_pp_offload` = `6c9943a91`；
  - `pp_balance_review1` = `k3_pp_balance` = `43ccdcc06`；
  - 旧 head 在 `backup/*_pre_20260929c`；
  - upstream 的 `refs/pull/4765/head`、`refs/pull/4764/head` 已是新 head；
  - #4656 的分支没动。
- **body：**
  - PR A v3 的 Relation 加了一句：前两个提交是 rebase 后的 #4656，#4656 合并前会显示在这里。
  - 三份 body 的状态区已更新。
  - PR A raise 前只差 `<torch issue link>`；Results 等 H100。

## 09-29 深夜：#4765、#4764 的 pyrefly，以及 balance 在 5060 上"tcp 传过去再放进显存"的冒烟（用户："现在修 并且balance 5060smoke结果的预期是八卡pp rank cache的显存接近平衡（走tcp传输 然后重新注入显存，这是5060 test only 并且有可能遇到PCIe瓶颈）"）

- **pyrefly：** #4765 = `2140058c5`，#4764 = `76b644a2a`、`224bdbaf4`，已推（旧 head 在 `backup/*_pre_20260929d`，GitHub 上两个 draft 的 head 已更新）。在 venv_0928 激活的环境里是 16 个错误，和 main 一样；76 / 90 passed；pre-commit（分范围）干净。两个 `_device` 错误来自 torch 0928 nightly 的类型存根（`torch.device` 声明了 `__get__`），按 main 的写法加 `# pyrefly: ignore [read-only]`。细节在两份 body 的状态区。
- **kit：** `kit_overnight_2026-09-29/balance_tcp_device/`（测试补丁、probe、三版启动脚本、结果）。都只在本地，不进 PR。

### 先踩到的坑：mooncake 往显存写走默认流

- **单独测是通的**（`probe_tcp_device_pool.py`，2 卡）：mooncake 0.3.13.post1 的 tcp 能读写对方注册的显存，同步和按流排队的接口都对，64 MiB 约 1.1 到 1.5 GB/s。所以 #4764 里 "The TCP transport serves host memory only" 这句注释不对。
- **但拷贝排在对方卡的默认流后面**（`probe_tcp_device_pool_busy.py`）：对方在默认流上空转 4 秒，写 64 MiB 用了 3.96 秒；空转放在旁路流上 0.05 秒；池在 pinned host 上 0.04 秒。mooncake 在池所在的卡上用 legacy 默认流拷贝，titan 的计算也在这条流上。
- **所以在训练进程里直接把池放进显存会卡死**（第一次，`run_bal8.sh`）：建池时目的 rank 在 NCCL barrier 上等源 rank，源 rank 的第一次写等目的卡的默认流，mooncake 报 `ClientSession: no status frame within 30s`。停掉了。
- **让计算改走旁路流也不行**（第二次，`run_bal8b.sh`）：只换流、不开 balance 的对照格在第 1 步的优化器 pre-hook（MoE quantile balancer）里就报 device-side assert，默认流上同样的配置没问题。栈里有组件在非默认流下不安全，没有继续查。
- **可行的办法：池放进同卡上的一个辅助进程**（第三次，`run_bal8c.sh`，`pool_holder.py`，`K3_TEST_TCP_POOL_PROCESS=1`）：辅助进程有自己的 CUDA context，mooncake 往它的池里拷贝用的是它自己的默认流，不和训练互等。源端走的还是 `RemoteBackend` 原来的调用；数据经 tcp 到目的机器，再由辅助进程拷进目的卡的显存。每张目的卡上的辅助进程另外占约 0.13 GiB context。

### 结果（8 × 5060，#4764 `224bdbaf4`；09-28 D 组的布局：93 层、block 12、每 stage 3 层、pp8 × vp4 Interleaved1F1B，16 个 micro-batch × 512 token，dim 1024，关 AC，seed 42，6 步，同一份预热缓存）

第 5 步的峰值，GiB（训练进程的 `max_memory_allocated`；"按卡"= 训练进程峰值加上放在这张卡上的池）：

| rank | 关 balance | balance，池在 host（PR 现状） | balance，池在目的卡（辅助进程） | 其中这张卡上的池 |
|---:|---:|---:|---:|---:|
| 0 | 6.93 | 6.42 | 6.43 | |
| 1 | 6.89 | 6.72 | 6.75 | |
| 2 | 6.61 | 6.59 | 6.59 | |
| 3 | 7.58 | 7.01 | 7.02 | |
| 4 | 6.12 | 6.12 | 6.35 | 0.23 |
| 5 | 5.83 | 5.83 | 6.41 | 0.58 |
| 6 | 5.57 | 5.57 | 6.30 | 0.73 |
| 7 | 6.04 | 6.04 | 6.38 | 0.34 |
| 最大 / 最小 | 7.58 / 5.57 | 7.01 / 5.57 | 7.02 / 6.30 | |
| rank 间的差 | 2.01 | 1.44 | 0.72 | |
| 平均 | 6.45 | 6.29 | 6.53 | |

- **数值：** 四格（关、关的重复、池在 host、池在目的卡）6 步的 loss 和 grad norm 全部逐位相同。
- **nvidia-smi 看到的每卡最高占用**（含 allocator 缓存和 context）：关 balance 时 9.73 到 7.37 GiB，差 2.36；池在目的卡时 9.29 到 8.23 GiB，差 1.06；目的卡 4 到 7 比关 balance 时多 0.36 / 0.71 / 0.85 / 0.47 GiB，也就是池加上辅助进程的 context。GPU 0 上一直压着另外 7 个 rank 各自的 132 MiB context，共约 0.9 GiB，关 balance 时也一样，训练进程自己的统计看不到。
- **步时**（rank 7 日志时间戳，第 3 到 6 步平均）：关 4.09 s，重复 4.10 s，池在 host 5.81 s（+42%），池在目的卡 35.79 s（8.7 倍）。每步源 rank 共停放约 1.9 GiB，反向再读回同样多（rank 0 到 3 五步累计 2.90 / 2.22 / 1.14 / 3.30 GiB）。慢的不是 PCIe 带宽：探针里单次 64 MiB 能到 1.1 到 1.5 GB/s，按这个速度每步只该多 3 秒左右。慢在 mooncake tcp 进显存的逐次开销：每个保存的张量单独一次传输，每次在对方同步拷贝一次，源端存储流上的传输一个接一个。
- **为什么没有完全拉平：**
  - 计划用的是第 1 步的 profile（计划里 profiled 那一列正好等于各 rank 第 1 步的峰值）。关 balance 时，每个 rank 从第 2 步起的峰值都比第 1 步高约 0.53 GiB（rank 3 高 0.62），大小和 AdamW 的两份状态一致：第 1 步之后常驻约 1.06 GiB，按参数、梯度、两份状态各约 0.27 GiB（bf16）推算。AdamW 在第一次 step 时才分配状态。所以计划的绝对数都偏低约 0.55 GiB（源 rank 计划 5.87 / 6.16 / 5.97 / 6.43，实测 6.42 / 6.72 / 6.59 / 7.01）。
  - 池会满：rank 0 到 2 每步各有 25 到 54 个保存张量因为自己那段池满了，只好留在本卡（统计里的 `remote_full`），rank 1、2 降得比计划少。
  - 池一共 1.88 GiB，源 rank 的峰值只一共降了 1.27 GiB：池要装下同一时刻停着的全部张量，比源 rank 峰值时刻停着的多。目的 rank 先填到均值，源 rank 就降不下去了。
  - 最重的 rank 3 只配了一个目的地（rank 6），计划本身就把它留在均值以上 0.5 GiB。
- **结论：** 把池真正放进目的卡的显存后，8 张卡的峰值差从 2.01 降到 0.72 GiB（按卡），loss 逐位不变，但在 5060 上每步慢 8.7 倍；PR 现在的 host 池只把源 rank 压下来（差 1.44），目的卡不涨，实际是 host offload。
- **对 #4764 的影响（没改，等用户定）：**
  - tcp 那句注释不对，真正的限制是"mooncake 往显存拷贝走默认流，会等本卡的计算"；
  - profile 步应该在优化器状态分配之后，或者把优化器状态计进去；
  - 池的大小和计划对不上，出现 `remote_full`；
  - H100 上的 `nvlink_intra` 要先用同一个忙流探针确认不会和默认流互等（已写进 `H100_NEXT_2026-09-30.md` §3b）。
