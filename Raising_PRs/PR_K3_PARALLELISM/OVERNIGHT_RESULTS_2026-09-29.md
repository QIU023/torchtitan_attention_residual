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

## T1d 5060 上 #4656 对新 PR A（进行中）

设置：debug model 放宽到 dim 2048，16 个 micro-batch × 2048 token，c4 文本（每行最多 2047 个 token，一个 micro-batch 一行），FullAC，AdamW，seed 42，确定性，每格 20 步，第 5 步逐动作记账。每个布局一份预热缓存，两棵树在不相交的 GPU 上同时跑。探针 `pra_bound/probe_bound.py`，脚本 `pra_bound/run_bound.sh`，表 `pra_bound/tab_bound.py`。下界在探针里按实际执行的调度算："紧界"是每个 block 在带它进来的 stage 反向时释放，"论文界"是一个 micro-batch 的 block 留到它在这个 rank 上最后一次反向。

**pp4 × vpp2：** 20/20 步 loss 和 grad norm 相同。

| rank | 第 10 步峰值 #4656 | PR A | 省 |
|---:|---:|---:|---:|
| 0 | 7.80 | 7.16 | 0.64 |
| 1 | 7.31 | 6.30 | 1.02 |
| 2 | 7.13 | 6.04 | 1.09 |
| 3 | 5.70 | 4.76 | 0.94 |

第 5 步，各 rank 峰值那个动作上的 block 占用（GiB）：

| rank | 树 | block 合计 | store | stage 输入 | stage 输出 | 只被 send 扣住 | 接收缓冲 | 紧界 | 论文界 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | #4656 | 0.53 | 0.02 | 0.05 | 0.08 | 0.39 | 0.02 | 0.09 | 0.16 |
| 0 | PR A | 0.17 | 0.09 | 0.00 | 0.00 | 0.08 | 0.02 | 0.09 | 0.16 |
| 1 | #4656 | 0.85 | 0.00 | 0.10 | 0.11 | 0.64 | 0.00 | 0.11 | 0.12 |
| 1 | PR A | 0.19 | 0.12 | 0.00 | 0.00 | 0.06 | 0.02 | 0.12 | 0.19 |
| 2 | #4656 | 0.75 | 0.00 | 0.13 | 0.08 | 0.55 | 0.00 | 0.09 | 0.12 |
| 2 | PR A | 0.14 | 0.09 | 0.00 | 0.00 | 0.05 | 0.02 | 0.09 | 0.16 |
| 3 | #4656 | 0.44 | 0.00 | 0.06 | 0.06 | 0.31 | 0.05 | 0.09 | 0.16 |
| 3 | PR A | 0.13 | 0.09 | 0.00 | 0.00 | 0.03 | 0.05 | 0.09 | 0.16 |

- #4656 在峰值时刻占着紧界的 5 到 7 倍，大头是整步都被 send 扣住的张量。
- PR A 的 store 正好等于紧界，整段 block 占用和论文界持平。高出紧界的是还没 wait 的前向 send：PR A 在本 stage 反向那个 micro-batch 时才 wait，接收方其实早就用完了。如果照梯度 send 的办法，从 `pipeline_order` 找出接收方用完之后的第一个动作再 wait，还能再贴近紧界。这是下一步可做的一项，今晚不改。

## T1e body 和 torch issue

- PR A body v3：`PR_BODY_PP_CACHE_OPT_v3_2026-09-29.md`。Summary 引论文 §4.1 "each block is stored exactly once across all V virtual stages"；Design 三段：一个 block 为什么在 4312 的 stage 里存了 1 + k 份、为什么在带进来的 stage 反向释放、send 早等为什么是释放生效的前提（接收缓冲交给 torch）。Results：Pending (H100)。
- torch issue 草稿 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md` 改成只提 send；接收缓冲写明已由 #196463 解决；原来的 5060 测量按规则拿掉，只留两卡复现。

## T1f #4765、#4764 重叠

- 重放到新 PR A 上没有冲突：#4765 = `019462171`，#4764 = `2a719e512`、`c4afb61f4`。pyflakes、ufmt 干净。新 venv 上 #4765 76 passed，#4764 90 passed（和之前的数一样）。
- body 的 Relation 改成叠在 #4656 和 PR A 上，不再提 `1777ad806`。
- 5060 冒烟和推送：待做（下界验证跑完以后）。
