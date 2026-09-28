# Overnight 计划（2026-09-28 夜，等用户点头再开始）

用户："那回到PR a 除了第三点目前保留 其他两点改在哪里 为什么严格说那两点加入之后才是cache的理论下界？怎么又涉及到torch运行时机制了？和后续的统一内存管理器还有balance的改动有联带做好吗？ cpu claude重构了dep，读取k3 pp mm 制定完整的overnight计划"

## 0. 资源和边界

- **5060：** 本机 8 × RTX 5060 Ti 16 GB，盘还剩 158 GB。现在 main 上的树要打本地 torch 兼容补丁（跑前打、跑后撤）。夜里这台只归我用。
- **H100：** `135.135.24.114:40191`，1 × 80 GB，vast 按小时计费，torch 2.15.0.dev20260926+cu130（`~/venv_pp`）。只有一张卡，PP、DEP、多卡格子都跑不了。
- **不碰：** 已发布的 PR 分支（#4656 已同步；#4780 等你贴完回复后关；#4881）；DEP 的代码（CPU 那边在改，我只跑验证，发现问题只记录、不改）；MoonEP（等你定 flavor）。
- **可以推：** review 分支、draft PR 分支（按"base 过时的 draft 直接 rebase 然后推"）、logbook。本地 dev 分支不推。

## 1. 现状

| PR | 分支 = head | 在谁之上 | 状态 |
|---|---|---|---|
| #4656 | `k3_ac_reuse_attention` = `attnres_review1` = `5d469fdf3` | main `f35966713` | 已同步；body v4 和给 shuhuayu 的回复待你贴 |
| #4780 | `k3_attnres_recompute` = `9f6bae06f` | 旧 main | 待你回 tianyu 后关闭 |
| PR A（未开） | `pp_review_optimize` = `d37fb90f1`（只含第 3 类） | main | body `PR_BODY_PP_CACHE_OPT_v2_2026-09-28.md`；5060 上对 main 还没测 |
| PR A 第 1、2 类 | 本地 `pp_review_optimize_dev` = `439bd2088`（fork 上的备份 `backup/pp_review_optimize_pre_20260928`） | 旧的列表提交 `aa6d9fedc` | 要重建到新的 #4656 和 PR A 之上 |
| #4765 draft | `k3_pp_offload` = `61734f376` | 旧 PR A `439bd2088` | 要跟着重叠 |
| #4764 draft | `k3_pp_balance` = `71e8bfaf2` | #4765 | 要跟着重叠 |
| DEP #4381 draft | `k3_pp_mm` = `dep_review1` = `bb3e38d4a`（CPU 那边重写） | main | GPU 上一项都还没验证 |

## 2. 用户的四个问题

1. **第 1、2 类在哪：** 只在本地 `pp_review_optimize_dev`（`439bd2088`），fork 上另有备份 ref。它们还叠在旧的列表提交 `aa6d9fedc` 上，不在任何 PR 里。
2. **为什么加上第 1、2 类才是 cache 的理论下界：** 下界要求两件事：每个 block 在一个 rank 上只存一份；从它到达（或产生）的那一刻，只活到它在这个 rank 上最后一次被读。
   - **第 1 类解决"只存一份"：** 4312 的 stage 在每个 stage、每个 micro-batch 进门时，都用 `torch.stack` 把 store 里的 block 和收到的 delta 拼成一条新的 [T, N, D] leaf，交给模型；这条 leaf 就是 stage 的输入，要留到反向（core 从它上面取输入梯度）。所以同一个 block 除了 store 里那份，每个读到它的 stage 还各有一份拷贝。第 1 类让 block 逐个传：收到的 block 就是它自己的接收缓冲，本 stage 产生的 block 就是模型自己的张量，store 和各层只存引用。
   - **第 2 类解决"只活到最后一次读"：** 按反向顺序，一个 block 在这个 rank 上最后一个读者，就是把它带进来的那个 stage，所以在那个 stage 的反向时释放。
   - **实测**（5060，C 组布局）：只加第 3 类是 8.07 GiB（最重的 rank），再加第 1、2 类是 7.11 GiB，每个 rank 都再少 0.9 到 1.1 GiB。
3. **为什么又牵涉 torch 的运行时：** block 能不能真的释放，要看还有谁引用它的存储。有两个引用是 torch 的，不归 cache 管：
   - torch 给每个 micro-batch 的每个输入都常驻一份接收缓冲，整步不放；按第 1 类的做法，收到的 block 就是这份缓冲；
   - action-list runtime 等到一步结束才 wait 所有 send，没 wait 的 send 会钉住发出去的张量。
   这两份不放，cache 释放了也没用。第 3 类就是改这两处。它本身省得最多（11.47 → 8.07 GiB），因为这些常驻缓冲和被钉住的 send 是最大的一块多余内存。
4. **#4765、#4764 有没有跟着改好：** 没有。它们还叠在旧 PR A `439bd2088` 上，而且依赖第 1、2 类：#4765 在 `cache.put` 时 pin block、按 block 释放时逐个 unpin（第 2 类的接口），前向时把第 1 类那组按 block 拆开的输入交给存储层。PR A 缩成第 3 类以后，它们只能叠在"第 3 类加第 1、2 类"的 dev 分支上。下面 T3、T4 就做这件事。

## 3. 夜里的任务（按顺序）

**T1 H100，约 20 分钟，然后建议停机：** 在 torch 0926 nightly 上跑 CPU 单测，这个 nightly 此前没跑过这些代码：
- #4656 `5d469fdf3`、PR A `d37fb90f1`（覆盖了 torch 的私有方法，最需要在新 nightly 上跑）、DEP `bb3e38d4a`（CPU 那边只在 Windows 上用打桩的 harness 跑过，这是第一次在真环境里跑），以及 T3、T4 重建出来的分支；
- 再跑单卡的 `tests/unit_tests/gpu/test_kimi_k3.py`。
- 之后 H100 上就没有能跑的了（PR A、DEP、#4765、#4764 都要多卡），建议停机：你来停，或者授权我在机器里用 `vastai stop instance`。

**T2 5060，约 30 分钟：** PR A `d37fb90f1` 对 main，用 PR A 一直以来的 5060 布局（s6：93 层、block 12、每 stage 3 层、dim 2048、seq 2048、M16、FullAC、pp8 × vp4）。10 步看每个 rank 的峰值和逐位一致，第 8 步 trace 计时（`campaign2.sh` 的协议）。数字只进 logbook，body 保持 Pending (H100)。

**T3 本地 CPU 加 5060，约 1 小时：** 重建 dev 分支：main → #4656 `5d469fdf3` → PR A `d37fb90f1` → 一个"第 1、2 类"提交。
- 这个提交的内容取自 `439bd2088`：模型对外改成收发列表（按 block 传输需要）、stage 按 block 传输、bringer 释放、相应测试。
- 跑 CPU 单测；再在 5060 上用 s6 布局对比 dev 和 PR A，确认 0.9 GiB 左右的差在新的叠法上还在。
- `pp_review_optimize_dev` 指向新 head，旧 head 留成本地 tag。

**T4 本地 CPU 加 5060，约 1.5 小时：** 把 #4765（`61734f376`）重叠到新的 dev 分支上，#4764（`71e8bfaf2`）再叠在 #4765 上。
- 只解冲突，不改功能。跑 CPU 单测，此前分别是 76 和 90 passed。
- 5060 冒烟：#4765 `cpu_offload=all`、#4764 `planned`、只开 balance（tcp）、planned 加 balance，布局同 09-27 夜 C、D 组。
- 通过后推两个 draft 分支（`k3_pp_offload` = `pp_offload_review1`，`k3_pp_balance` = `pp_balance_review1`），旧 head 先备份。body 里写明新的叠法（dev 分支不是 PR）。

**T5 5060，约 2 到 2.5 小时：** DEP `bb3e38d4a` 的 GPU 验证，照审计文档 §8 那张清单（`DEP_VS_REPORT_AUDIT_2026-09-27.md`）：
1. B200 格子 `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4_vision_dep`（8 卡）10 步能跑通；
2. 数值，同一份预热 cache：DEP 关对开（K2.5 模式），第 1 步 loss 逐位相同、文本梯度逐位相同，塔梯度配一行噪声底（同一格跑两次）；bubble 关对开，做同样的比较。梯度用 09-27 那套 dump 探针（`dep_bubble_probe.py`、`cmp_grads.py`）改到新实现上；如果挂不上新的路径，就只比 loss 和 grad norm，并记下原因；
3. DEP 关、DEP 开、bubble 开三格，各抓一步 trace 看步时；5060 是 PCIe，只作参考；
4. 三格每个 rank 的显存。
- 只验证不改代码；发现问题记进 logbook，早上报给你（或转给 CPU 那边）。结果只进 logbook，DEP 的 body 保持 pending。

**T6 本地，约 30 分钟：** #4656 和 PR A 改过的文件跑 scoped pre-commit（含 pyrefly；上一版 #4656 的 body 列过这一项）。

**T7：** 每做完一项就把 logbook 推一次（另一个 session 也在推，推之前先 merge），并更新记忆。

总共约 6 小时。

## 4. 早上给你看的，以及要你定的

- #4656：贴 body v4，回复 shuhuayu；#4780：回复 tianyu 后关闭（草稿都在 `PLAN_4656_4780_2026-09-27.md` 的"更正 3"）。
- PR A：用 `PR_BODY_PP_CACHE_OPT_v2_2026-09-28.md` 以 draft 开；torch issue（草稿 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md`）由你开。
- #4765、#4764：重叠后的 body。
- DEP：T5 的结果和发现的问题。
- H100 是否停机；MoonEP 的 flavor 怎么处理。

## 5. 停止条件

- 任何格子数值对不上：先按规矩在同一份 cache 上重跑定位；定位之前不写结论、不进 body，早上报给你。
- 冲突解不干净、单测数目对不上：停在本地，不推。
- 盘剩不到 60 GB：先删这次的 cache 目录再继续。
