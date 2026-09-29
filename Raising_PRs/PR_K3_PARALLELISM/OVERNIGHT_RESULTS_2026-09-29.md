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
- **#4656 在峰值时刻占着紧界的 5 到 13 倍**，大头是整步都被 send 扣住的张量，其次是每个 stage 进门拼出来的 stack 和模型输出的 stack。
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

## T3 MoonEP 审计（子任务完成，logbook `8fb189e`）

- **一个真 bug，已修：** `reduce_grad` 前少一道跨 rank 栅栏。MoonEP 的规约 kernel 直接远程读各 rank 的槽梯度，自己不带 barrier（`grad_reduce.py:467-470`），要调用方保证各 rank 都写完；重写版写完槽梯度马上规约，中间没有同步，别的 rank 可能读到没写完的梯度。H100 上 GPU 单测 2 passed 只说明那两次时序没撞上。修法：规约前在 EP 组上做一次单元素 all-reduce，排在同一条 stream 上。
- **一个稳健性问题，已修：** `Buffer` 改成 `explicitly_destroy=True`（和 DeepEP 一样），GC 时不再跑带设备同步和 barrier 的 `destroy()`。
- 其余调用逐个对过 MoonEP `33327eb` 的源码：dtype、`zero_copy`、复用 plan 时的 rank 同步、补零行、池的对齐、梯度都是新张量，没问题。
- **rebase 和推送（用户确认）：** `moonep_review1` = `1633dcd79`，upstream main `a182e530a` 上三个提交（两个原提交重放，加修复），没有冲突；旧 head `a505f74a8` 在 `backup/moonep_review1_pre_20260929`。CPU 单测 34 passed、13 subtests passed，ufmt 干净。PR 分支 `k3_moonep_seam` 没动。新 head 还没在真实 MoonEP 上跑过。
- **负载均衡的证据方案**（`kit_overnight_2026-09-29/moonep/`）：128 个专家、top-8，FSDP4 × EP4，standard 对 MoonEP，自然路由和偏向 rank 0 专家的路由各一格；记每层每个 rank 的负载 max / mean，以及 MoonEP 每次 dispatch 是否正好 S × K。假包版本排在本机 GPU 队列里；真机版本约 15 分钟，另加新 head 的 GPU 单测和 CI 格约 10 分钟。
- 审计文档 `DIFF_AUDIT_MOONEP_2026-09-28.md` §7；body `PR_BODY_MOONEP_v2_2026-09-29.md` 的 Test plan 和 Results 改成负载统计加步时，Requirements 加了栅栏和"LoRA 与 MoonEP 不能同时开"。
