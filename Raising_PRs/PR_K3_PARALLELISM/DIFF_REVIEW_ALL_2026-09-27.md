# PP 线所有 PR 分支的 diff 重新审核（2026-09-27 夜）

用户："先暂停h100了 你这么多pr分支都重新审核diff并且确保5060smoke完成，然后制定好新尺寸，记录下"。

审核只列问题，**没有改任何分支**；要改的等用户点头。所有分支的提交信息都没有 Co-Authored-By / Claude-Session，也没有跨仓库引用（`owner/repo#N`、github 链接、三位以上的 `#N`）。upstream main 仍是 `f35966713`。

| PR | 分支 = head（base） | 自身 diff | 结论 |
|---|---|---|---|
| #4881 零初始化 | `k3_attnres_zero_init` = `ae3a7881b`（`56f04c702`） | 1 文件 +9/−3 | 就绪（tianyu 已批准） |
| #4656 列表载体 | review `attnres_review1` 的 `aa6d9fedc`（main） | 6 文件 +173/−78 | 代码就绪，测试文件要按拆分收一下 |
| #4780 torch_remat checkpoint | `attnres_review1` = `f14d681f4`（#4656 之上） | 2 文件 +138/−4 | 就绪 |
| PR A | `pp_review_optimize` = `439bd2088`（#4656 之上） | 5 文件 +516/−268 | 代码就绪；依赖 torch 私有接口，torch issue 未开 |
| #4765 | `k3_pp_offload` = `61734f376`（PR A 之上） | 9 文件 +928/−16 | 功能完整，有两处可收 |
| #4764 | `k3_pp_balance` = `71e8bfaf2`（#4765 之上，两个提交） | 7 文件 +1071/−37 | 不算就绪 |
| DEP #4381 | `k3_pp_mm` = `31f372593`（main） | 12 文件 +1262/−13 | 基本就绪，几处小问题 |
| MoonEP #4751 | `k3_moonep_seam` = `f556ab4fd`（09-18 的 `6c2dadbb3`） | 13 文件 +1072/−12 | 不算就绪 |

## #4881
- diff 干净。新加的一行注释（"Zero init gives the uniform initial depth weights the report requires."）讲设计理由，按"默认不加注释"可以删；已批准，可不动。
- body 还列着 09-25 删掉的 `test_kimi_k3_attention_residual_init.py`，合并前要删。

## #4656（`aa6d9fedc`）
- `model.py`、`sharding.py`、`stage.py` 的改动逐行对：模型拿到的是 stack leaf 的 `unbind` 视图，core 仍从 leaf 读输入梯度；提交和 payload 按列表下标取；store 存 block 本身而不是 stack 的视图。`stage.py` 那行注释讲的是约束（core 从 leaf 读梯度），留。
- **测试文件是拆分留下的样子：** `test_kimi_k3_attention_residual.py` 的模块 docstring 写 "its block list and its recompute in backward"；`_TwoBlocks.forward` 和 MLA 的 CPU 替身只有 #4780 的 recompute 测试用到，#4656 自己的列表测试只用 `layers["0"]`。应该挪到 #4780 的提交里。
- 测试辅助类上的 docstring（`_CpuKernel`、`_TwoBlocks`）和一行注释，按用户修 PR A 时删私有 docstring 的做法应该删。

## #4780（`f14d681f4`）
- 调用处的 torch_remat checkpoint、block 的 `checkpoint_residual` 旗标、`KimiK3Model.parallelize` 在 selective / full / region AC 时清旗标、output aggregation 总是自己 checkpoint，都和 body 一致。两行注释都是约束，留。
- 测试里 `_unwrapped_residual` 带一个辅助函数 docstring，同上可删。

## PR A（`439bd2088`）
- 逐行看过 `stage.py`、`cache.py`、`__init__.py`：收到的 block 直接作为输入，store 的 block 做成新 leaf，输入梯度按传输顺序在前；store 里 block 的梯度 deposit 汇到带它来的 stage 再发回上游；每个 block 在带它来的 stage 的反向时释放；rank 最后一次反向检查没有残留；LoRA 冻结 embedding 没有梯度的情况也处理了。没有发现逻辑错误。
- 测试辅助类没有 docstring；新增注释 8 行都是约束。
- **风险：** 覆盖 torch 的 `_setup_forward_recv_info`、`_setup_backward_recv_info`，用 `_batch_p2p`、`_make_tensor_from_meta`、`_PipelineScheduleRuntime`、`args_recv_info` / `grad_recv_info` 的内部字段。签名变了就会坏：5060 的 0906 torch 上测过，**H100 上用的 0926 nightly 还没跑过 PR A 的单测**，下次上机第一件事先跑单测。torch issue（`TORCH_ISSUE_PP_BUFFERS_2026-09-27.md`）还没开。

## #4765（`61734f376`）
- `activation_storage.py` 逐行看过：复制出去的源张量持有到拷贝完成；读回在计算流分配、存储流拷贝、event 同步；pin 住的 block 不参与 offload；RECOMPUTE 直接报错。
- 可收的两处：`BackwardPrefetch.end()` 在 #4765 里是空方法（给 #4764 的控制器留的接口，在这里是死代码）；按层打标签靠每个 block 的 forward pre-hook，#4577 的标准不鼓励 hook，body 要一句话说明缺的是哪个 seam。
- `model.py` 的配置从 `pipeline_parallel.activations` 引入 `PPMemoryConfig`（模型配置依赖 PP 包），这是 09-26 定的"开关都放在模型配置里"，评审可能会问。

## #4764（`400add9f9` + `71e8bfaf2`）
- `_load_transfer_engine()` 在导入 mooncake 失败时用 ctypes 预加载 CUDA 12 的 runtime：在库代码里绕环境问题，评审大概率会挑。
- 两个实测问题没定位：profile 出来的峰值比实测高 1.1 到 1.8 GiB；balance 和 offload 一起开时 rank 间差比只开 balance 大。
- `nvlink_intra` 没在 NVLink 机器上跑过（可能要用 mooncake 的 `NVLinkAllocator` 分配池）；5060 没有 P2P 测不了。
- 源码 600 多行偏大；`balance` 单独开的那个新提交（`71e8bfaf2`）+55/−21，测试里一行注释解释目标值，是约束，留。

## DEP（`31f372593`）
- `install_vision_dep` 的报错还用旧名字 `vit_bubble` / `vit_prefetch`，配置已经是 `vision_dep.bubble` / `vision_dep.prefetch`。
- 命名混用：函数、常量、日志是 `vit_dep`，配置是 `vision_dep`。
- 把 `pp_schedule.step` 包了一层做每步的开始和结束，body 没写原因（core 的 schedule 没有每步的 hook；#4764 的 body 写了）。
- 改了 core 的 `OptimizersContainer`（某个 stage 上 pattern 匹配不到时不报错）：评审可能要求拆成单独的 core PR。
- 新增一个 B200 CI 格和 recipe（`kimi_k3_debugmodel_pp4_vp2_vit_dep`）：按规则加之前要问用户，请确认保留。
- 注释和 docstring 比 #4577 的标准多，可以再修一遍。

## MoonEP #4751（`f556ab4fd`）
- 基于 09-18 的 main，落后 59 个提交；和现在的 main 有 5 个文件冲突（`models/common/moe.py`、`kimi_k3/__init__.py`、`kimi_k3/config_registry.py`、`tests/integration_tests/h100.py`、`torchtitan_recipes/tests/h100.py`），验证前要 rebase。
- 带了一个选后端的 flavor `kimi_k3_debugmodel_moonep` 和 h100 格 `kimi_k3_moonep_fsdp4_ep4`：和 CLAUDE.md 的规则冲突（选后端的 flavor 不加，评审想换后端自己改 registry 参数），要用户定。
- 换到 MoonEP 公开版后，on-device test 和 h100 格还没在公开版上跑过；缺 DeepEP 对比（DeepEP v2 已装在 H100 机器的 `venv_k3`）。
- 注释比例 9.9%（09-19 审核的数），偏高。diff 里没有遗留的调试代码或 logbook 路径。

## 5060 冒烟（`scratchpad/s5060/`，结果拷进 `kit_5060_smoke_2026-09-27/`）

- torch 2.15.0.dev20260906（`venv_bfx9`），现在 main 的树都打本地兼容补丁（跑完撤掉，`worktrees_after.txt` 记了撤掉后的状态）。设置都是已有记录的：A 是 body 的 recipe，B 是 B200 组合格，C 是 PR A 表的 PP 布局，D 是 b1 的 balance 布局。
- **A. 单卡，body 的 recipe**（debug model，2048 token/步、512 一个 micro-batch，10 步，seed 42，deterministic，一份预热 cache）：main 对 #4656 在三种 AC 下 10 步 loss 和 grad norm 全部逐位相同。

| AC | main 第 1 步 | main 第 10 步 | #4656 与 main 相同的步数 | 显存 main / #4656（GiB） |
|---|---|---|---:|---|
| none | 8.03468 / 2.3594 | 5.90405 / 8.5000 | 10 / 10 | 0.66 / 0.66 |
| selective | 8.03468 / 2.3594 | 5.90405 / 8.5000 | 10 / 10 | 0.34 / 0.34 |
| full | 8.03468 / 2.3594 | 5.90405 / 8.5000 | 10 / 10 | 0.30 / 0.30 |

- #4881（`ae3a7881b`，base 是 09-25 的 main `56f04c702`，那时的 debug model 还是 dim 1024、vocab 163840、24 层，所以第 1 步 12.53217 和上表不可比）：关 AC 10 步 rc=0。
- **B. B200 组合格 `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4`（8 卡，10 步，recipe 不固定 seed）：** #4656 和 PR A 都 rc=0（第 10 步 3.45575、3.62889，两次初始化不同，不比）。
- **C. PR A 表的 PP 布局**（93 层、block 12、每 stage 3 层、dim 2048、seq 2048、M16、FullAC、pp8 × vp4，6 步，一份预热 cache）：四格 rc=0，6 步 loss 和 grad norm 四格逐位相同；第 5 步每个 rank 的峰值（GiB，实测 5060）：

| | #4656 | PR A | #4765 `all` | #4764 `planned` |
|---|---:|---:|---:|---:|
| 最大 / 平均 | 11.47 / 10.80 | 7.11 / 6.23 | 6.74 / 5.91 | 6.75 / 6.15 |

#4656 和 PR A 两列和 09-27 的 s6（11.47 / 10.80 → 7.12 / 6.23）一致。

- **D. b1 的 balance 布局**（同上但关 AC、dim 1024、seq 512，6 步，一份预热 cache；5060 没有 P2P，池走 tcp 放在对端的 host 内存里，这里没有模拟设备上的池）：三格 rc=0，6 步逐位相同。

| | #4764 都不开 | 只开 balance | planned 加 balance |
|---|---:|---:|---:|
| 最大 / 最小（GiB） | 11.19 / 8.88 | 10.76 / 8.88 | 10.20 / 8.87 |
| rank 间的差 | 2.31 | 1.87 | 1.32 |

  - 这里的差只反映源 rank 搬走了多少：池在 host 内存里，目的 rank 的 GPU 峰值不涨。09-27 模拟设备池的 b1 是 2.31 → 0.70（`kit_pp_lowerbound_2026-09-26/results/balance_table_b1.md`）。H100 上用 `nvlink_intra` 才是池真正在目的 GPU 上的情况。
  - profile 和实测对不上，两个方向都有，都没定位：planned 加 balance 的 profile 比都不开的实测高 1.1 到 1.8 GiB（和上面 #4764 审核里记的是同一个问题）；只开 balance 的 profile 反过来比都不开的实测低 0.52 到 0.63 GiB，八个 rank 差不多一样，所以只开 balance 时源 rank 的实测比 plan 高约 0.55 GiB。
- 表和原始记录：`kit_5060_smoke_2026-09-27/results/`（`table_a.md`、`table_cd.md`，每格的 `steps.txt`、`rank*.json`、`plan*.json`）。
