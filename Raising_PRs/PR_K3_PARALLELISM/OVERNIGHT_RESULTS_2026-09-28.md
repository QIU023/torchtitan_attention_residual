# Overnight 结果（2026-09-28 夜）

计划：`OVERNIGHT_PLAN_2026-09-28.md`。用户的目标："全部完成，不要动任何已经发布pr的分支"，任务 T2 到 T6。

## 约束的执行

- 已发布 PR 的分支一个都没动：#4656 `k3_ac_reuse_attention`、#4780、#4881、DEP #4381 `k3_pp_mm`、#4765 `k3_pp_offload`、#4764 `k3_pp_balance`。
- T4 里"推两个 draft 分支"按这条约束改成只推它们的 review 分支（`pp_offload_review1`、`pp_balance_review1`）；draft 也是已发布的 PR。
- 计划里的 T1（H100）不在目标里，H100 没动，机器还开着（按小时计费）。

## T6 pre-commit（含 pyrefly）

- **#4656**（`5d469fdf3`，`model.py`、`sharding.py`、`test_kimi_k3_attention_residual_recompute.py`）：trailing whitespace、ast、merge conflict、large files、license、flake8、µfmt、pydoclint、codespell 全部通过。pyrefly 失败，但和 main（同一环境、同一个 hook，干净的 `f35966713`）逐文件比较完全相同：22 个文件、42 个错（torch 0906 缺的接口、`torch_checkpointing` 等环境问题和 main 自己的错）。#4656 没有新增。
- **PR A**（`pp_review_optimize`）：同样只有 pyrefly 失败，但比 main 多 1 个：`stage.py` 的 `get_fwd_recv_ops` 把 `info.tensor_meta`（`_TensorMeta | None`）直接传给 `_make_tensor_from_meta`。补上 `is not None` 的判断（反向那边本来就有），amend 成 `d40bd628d`，推到 review 分支；之后 pyrefly 和 main 完全相同，PP 单测 67 passed。前向要接收的张量一定有 meta，所以这个判断不改变行为。
- hook 会在整个仓库删掉 pyrefly 认为多余的 suppression（25 个文件 40 处），这些附带改动都在临时 worktree 里撤掉了，没有进任何提交。
- 日志：`kit_overnight_2026-09-28/t6_*.log`。

## T2：PR A（只含第 3 类）对 main，8 × 5060

- 树：main `f35966713`，PR A `d37fb90f1`（T6 之前的版本；`d40bd628d` 只多一个运行时不会走到的 `is not None` 判断）。
- 布局：s6（93 层、block 12、每 stage 3 层、dim 2048、seq 2048、M16、FullAC、pp8 × vp4，seed 42，deterministic），`campaign2.sh` 的协议：一份预热 cache，10 步看显存和一致性，另跑一格在第 8 步抓 trace。

| 第 5 步，GiB | main | PR A |
|---|---:|---:|
| 最重的 rank | 12.00（rank 4） | 8.49（rank 4） |
| 平均 | 11.09 | 7.44 |

- 10 步的 loss 和 grad norm 两边逐位相同。main 这一列和 09-27 在 4312 上测的基线（s5：12.00 / 11.09）相同。
- 第 8 步（8 个 rank 的平均，ms）：窗口 25052 → 24962，计算 7431 → 7428，暴露的通信 14540 → 14543。显存省了，时间没有增加。
- 原始记录：`kit_overnight_2026-09-28/t2/`（每个 rank 的 json、每步的 loss、表）。

## T3：dev 分支重建，第 1、2 类在新叠法上的作用

- 本地 `pp_review_optimize_dev` = `1777ad806`：#4656 `2516926f3`、`5d469fdf3` → PR A `c87a5e102`（`d40bd628d` 挑过来）→ 第 1、2 类 `1777ad806`（5 个文件 +383/−271；`model.py` 对外改回收发列表，按 block 传输、bringer 释放，PP 文件取自旧的完整版 `439bd2088`，`get_fwd_recv_ops` 同样补了 `is not None`）。旧 head 留成本地 tag `pp_review_optimize_dev_pre_20260928`。没有推。
- 代码树和旧的完整版 `439bd2088` 只差 #4656：`model.py` 里的 checkpoint、原来那个 recompute 测试文件，以及 #4656 不再带的列表测试。
- 单测：recompute、三个 PP 测试和 `test_pipeline_parallel.py` 共 71 passed。
- 5060，s6 布局，一份预热 cache，10 步：

| 第 5 步，GiB | PR A 叠在 main 上（T2） | #4656 加 PR A（`c87a5e102`） | dev（加第 1、2 类，`1777ad806`） |
|---|---:|---:|---:|
| 最重的 rank | 8.49 | 8.09 | 7.12 |
| 平均 | 7.44 | 7.18 | 6.23 |

- 后两格 10 步逐位相同。第 1、2 类在 #4656 加 PR A 之上每个 rank 再省 0.73 到 1.22 GiB（最重的 rank 省 0.97），和之前在旧叠法上测的 0.9 GiB 左右一致。
- T2 那一列和这里的第二列不是同一份 cache，只作参考：#4656 的列表载体在 FullAC 下本身也省了一些（8.49 → 8.09）。
- 原始记录：`kit_overnight_2026-09-28/t3/`。

## T4：#4765、#4764 重叠到 dev 分支

- **重叠：** #4765 `61734f376` 挑到 dev `1777ad806` 上成为 `56f4cd3b0`，#4764 的两个提交 `400add9f9`、`71e8bfaf2` 再挑上去成为 `df8aeb930`、`0a9034257`。三个提交都没有冲突，各自的 diff 和原来逐行相同（去掉 diff 头比较 +/- 行），功能没改。
- **单测**（两份 body 的 Test plan 命令）：#4765 76 passed，#4764 90 passed，和原来一样。
- **5060 冒烟，C 组**（s6 布局，FullAC，一份预热 cache，6 步）：三格和 dev 参照格 6 步逐位相同；第 5 步峰值（最重的 rank / 平均，GiB）dev 7.12 / 6.23，#4765 `cpu_offload=all` 6.75 / 5.91，#4764 `planned` 6.76 / 6.15，和 09-27 在旧叠法上的 7.11 / 6.23、6.74 / 5.91、6.75 / 6.15 相差不超过 0.01。
- **5060 冒烟，D 组**（b1 布局：关 AC，dim 1024，seq 512；池走 tcp，不模拟设备池）：三格 6 步逐位相同，loss 和 09-27 的 D 组也逐位相同。

| 第 5 步，GiB | 都不开 | 只开 balance | planned 加 balance |
|---|---:|---:|---:|
| 最大 / 最小 | 7.58 / 5.57 | 7.01 / 5.57 | 6.48 / 5.57 |
| rank 间的差 | 2.01 | 1.44 | 0.91 |

  峰值比 09-27（11.19 / 10.76 / 10.20）低，是因为现在的底子里有 #4656：关 AC 时它的 recompute 生效；loss 不变，说明 recompute 不改变数值。
- **推送：** 只推了 review 分支，`pp_offload_review1` = `56f4cd3b0`、`pp_balance_review1` = `0a9034257`（旧 head 备份为 `backup/pp_offload_review1_pre_20260928`、`backup/pp_balance_review1_pre_20260928`）。PR 分支 `k3_pp_offload`（`61734f376`）、`k3_pp_balance`（`71e8bfaf2`）是已发布的 draft，按目标里"不要动任何已经发布pr的分支"没有动，同步等你说。
- 原始记录：`kit_overnight_2026-09-28/t4/`。

## T5：新 DEP `bb3e38d4a` 的 GPU 验证（8 × 5060）

**结论：B200 格子第 1 步就死锁，§8 清单后面几项（数值、trace、显存）都测不了。** 原因在 DEP 的传输：步首就把特征和梯度的 receive 全部挂出去，对面的 send 却要等本 rank 之后的计算。只验证，DEP 分支没有改。worktree 用完已恢复干净（`bb3e38d4a`，dirty=0）。

| §8 清单 | 结果 |
|---|---|
| CPU 单测（本机 torch 2.15.0.dev20260906） | 45 passed, 6 subtests passed（dep_plan、vision_dep、pp_stage、pp_block_grads、pp_layout、integration definitions） |
| 1. B200 格子 `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep` 跑 10 步 | 第 1 步死锁，600 s 后 NCCL 超时（rc=1）；加 `CUDA_MODULE_LOADING=EAGER` 重跑，照样死锁 |
| 2. DEP 关对开、bubble 关对开的数值 | 测不了：DEP 开的格子（K2.5 模式、bubble 模式）都过不了第 1 步。DEP 关的预热格正常（rc=0） |
| 3. 三种配置的 trace 和步时 | 同上 |
| 4. 每个 rank 的显存 | 同上 |
| 5. 大图 | 没测 |

**死锁的现场**（默认设置，`kit_overnight_2026-09-28/t5/lazy_default/`）：
- 有图像的 dp 副本（rank 2、3、6、7）：主线程全都停在第一个预编码里，塔前向的第一行 `grid_thw.tolist()`（`kimi_k2_7/vision_encoder.py:419`，调用链 `begin_step` → `_encode` → `_tower_forward`）。
- 超时的操作都在 DEP 自己建的两个组上：
  - stage 0 所在的 rank 2、3，挂在特征组上（name 45、46）；
  - pp1 的 rank 6、7，挂在梯度组上（name 49、50）。
  - 这两处都是 `_post_receives` 在步首挂出的 irecv。特征要等 pp1 自己编完才发，梯度要等 stage 0 的反向才发。
- 纯文本的副本（rank 0、1、4、5）已经进了调度，停在第 1 步形状推断里的 FSDP all-gather 和 `recv_object_list` 上。它们在等上面那四个 rank，是被连带卡住的。
- K2.5 模式（`bubble=False`）的预热格也卡在同一个地方。计划日志：K2.5 模式 "encodes 4 before the schedule … backwards 4 after it"；bubble 模式 "encodes 2 before the schedule, 2 in idle slots; backwards 0 in idle slots, 4 after it"（这个格子 pp2 × vp4、M4，没有一个反向放进空闲槽）。

**机制，用两卡最小复现确认**（`t5/repro/irecv_repro.py`，30 行，输出在 `t5/repro/`）：
- rank 1 先挂一个 irecv，再做一次计算加读回；rank 0 要等 rank 1 读回完成才发送。这就是 DEP 里"特征要等对面编码完才发"的依赖关系。
- `irecv` 在 host 上立刻返回（0.000 s）。可之后的计算：
  - 放在当前流上，读回卡住，40 s 超时；
  - 放在 torch 新建的旁路流上，同样卡住；
  - 不做读回、只轮询一个 GPU 事件，host 连轮询都走不到，也卡住。
- 能跑通的三种情况：
  - 先把同样的计算跑一次再挂 irecv（kernel 已经加载），读回 0.000 s；
  - `CUDA_MODULE_LOADING=EAGER`，读回 0.001 s；
  - 计算做完再挂 irecv，读回 0.04 s。
- 这正是 CUDA Programming Guide §4.8.5.1（Lazy Loading, "Impact on Concurrent Kernel Execution"）写的情况："A deadlock can occur if cross-kernel synchronization is required, but kernel execution has been serialized."
  - NCCL 的 receive kernel 一直在等对面；
  - 本 rank 上一个 kernel 第一次加载，要等设备安静下来；
  - 而对面发送之前，要等本 rank 把这个 kernel 跑完。
  - torch 在没有设置时默认用 `CUDA_MODULE_LOADING=LAZY`。
- **EAGER 不是规避办法：** 真实格子在 EAGER 下卡住的位置从第 419 行挪到了第 429 行 `grid_thw.prod(dim=-1)`（`t5/eager/`）。它只消掉了第一个要等设备的点。此外，运行时编译的 Triton kernel（塔里的 flex attention），以及新图像尺寸下挑到的新 kernel，每次都要重新加载。

**归属和影响：**
- 这是 DEP 传输（#4381）的问题，不是 torch 或 NCCL 的：按 CUDA 文档，receive 挂着不能跨过对面要等的计算。
- 按代码推理（没测）：torch 自己的调度也是先挂 receive、后算，但它的 receive 一定在对面同一步的 send 就位时才等，所以不会绕成环。DEP 的 receive 在步首挂出，对面的 send 要等本 rank 之后的计算（编码时 TP 组的集合通信、stage 0 的反向）。
- 推断（没在 B200 上跑过）：CI 的 B200 格子同样会在第 1 步卡住。那边默认同样是懒加载，而且这个环不看机器快慢。
- gloo 单测看不到这个问题：没有 CUDA kernel，也就没有加载。需要一个 NCCL 下的 GPU 测试。

**给 CPU 那边的约束**（怎么改由 DEP 的作者定）：
- 一个 P2P 操作（receive 或 send）挂出去以后，在对面的配对操作就位之前，本 rank 不能再做对面在等的工作。
- 已知安全的写法，是 torch 调度自己的写法：双方在同一个点挂出配对的 send 和 receive（`batch_isend_irecv`），然后马上等。
  - K2.5 模式：预编码全部做完以后，特征一次交换；调度结束以后、尾段反向之前，梯度一次交换。
  - bubble 模式：传输要跟锚点动作两侧都会到达的点配对。
- 推断（没测）：只是把 receive 推迟到用之前才挂，可能不够。对面的 isend 同样会挂着，而对面后面还要跑自己的 stage，也可能绕成环。

**这一项花掉的时间：** 队列里 DEP 开的格子每格白等约 11 分钟。确认之后我手动停了队列（13:41），跑完复现，又用 EAGER 试了一格。DEP 关的三个测量格没有跑：没有 DEP 开的格子配对，它们单独没有用。等修好以后，按 `t5b.sh` 的顺序一起跑。
