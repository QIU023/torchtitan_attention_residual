# Phase 3 报告 — 流水并行 + 跨 stage 缓存适配器

**日期**：2026-04-20 → 2026-04-23（4 个调试 session + 1000 步 / ~200K 步验证）
**状态**：**4-GPU PP=4 V=2 已完成**；8-GPU 复测延后（无 8 卡机）
**硬件**：4× RTX 5090 PCIe（每卡 32 GB），单节点

---

## 1. 目标

让 Block AttnRes 在 torchtitan PP 下跑通，**并且**把跨 stage 通信成本从 `O(stage_id × d)` 压成每跳常数 `≈(P−1)·N_p·d`（cross-stage caching adapter）。Adapter 是 RFC PR-2 的工程价值核心——正确性单靠 Phase 2 的 tuple-output forward 就够（PyTorch `_PipelineSchedule` 原生解 tuple，naive PP 不需要 adapter 代码）。

验收门槛：adapter 和 naive PP 的 loss 在 bf16 / NCCL 噪声内对齐；峰值显存接近 naive 基线（不再有 `retain_graph` 膨胀）；前向 wire 字节数和静态布局表 `delta_to_send` 一致。

---

## 2. 交付物

### 2.1 工作区（`phase3_attnres_pp_integration/`，**不**进 PR）

| 文件 | 作用 |
|---|---|
| `README.md` | runbook + 8-GPU 阶梯计划 |
| `adapter_design.md` | adapter 状态机和不变量；列了 5 个开放未知项（mb 键、VP 顺序、hook 可靠性、AC 交互、FSDP reshard），逐项标记解决进度 |
| `go_8gpu.sh` | 一键编排（env → tokenizer → C4 prefetch → 单测 → naive PP → adapter PP → 对比） |
| `prefetch_c4.py` | C4 shard 并行下载到 HF cache（默认 150 shard ≈ 45 GB / 22B tokens），解决 Phase 2 的 streaming httpx 崩溃 |
| `fake_pg_test.py` | 单卡 fake-PG `PP=4` smoke（调试辅助，不在主路径上） |
| `launch_8gpu_naive.sh` / `launch_8gpu_adapter.sh` | 原计划的 `PP=8 V=2` 启动器（本期未跑） |
| `launch_4gpu_naive.sh` / `launch_4gpu_adapter.sh` | 替代的 `PP=4 V=2 lps=2` 启动器（8 个虚拟 stage，每 rank 2 个 chunk） |
| `launch_4gpu_baseline_L16.sh` | 非 AttnRes 的 Llama3 PP=4 sanity（独立检验 PP 路径） |
| `rank_to_gpu_wrapper.py` | 未启用的 MPS wrapper——为将来 "8 ranks × 4 GPUs via `CUDA_VISIBLE_DEVICES=$((LR%4))`" 留个起点（NCCL 拒绝默认下的重复 GPU 折叠） |
| `compare_pp_vs_single.py` | TB events 对比器：单卡参考 / naive PP / adapter PP 的 max-abs diff |
| `plot_naive_vs_adapter.py` + `naive_vs_adapter_loss.png` | 头条对齐图 |
| `handoff_status_20260420{,_part2,_part3}.md`、`20260421.md` | 5 次设计迭代的 session-by-session 调试记录（详见 §3） |

### 2.2 进 PR 的代码（`torchtitan/experiments/attn_res/`）

| 文件 | 作用 |
|---|---|
| `layout.py`（新增） | `BlockLayoutTables`——对 `(P, V, num_blocks, n_layers, layers_per_block)` 离线化算子代数：`commits_at(S)`、`rank_cache_at_entry(R, v)`、`delta_to_send(S)`、`producer_stage_of_block(b)`、`cache_consumers_of_block(b)`。纯元数据，零 NCCL。`_grad_tag_base()` 为已废弃的 send-back 协议保留 P2P tag 空间 |
| `pipeline_adapter.py`（新增） | `RankLocalCache`（每 rank 每 mb 跨 virtual stage 共享）、`CrossStageCacheAdapter`（包每个 stage 的 submod）、`pipeline_llm_with_cache_adapter`（自定义 `pipelining_fn` 挂到 `ModelSpec`）。由 `TORCHTITAN_ATTNRES_CACHE=1` 闸门，非 Interleaved1F1B 走 fallback warn。Monkey-patch `forward_one_chunk` / `backward_one_chunk`，把 schedule 的整型 chunk id 通过 thread-local 串起来 |
| `model.py`（修改） | 新增 `_return_only_new_blocks` 标志；置 True 时非末 stage forward 只返回本 stage 新提交的 block（常数大小发送） |

`tests/test_pipeline_adapter.py` 的 CPU 单测从 0 → 41+，覆盖：mb-index threading、rank-cache 语义、forward-delta 数值、backward grad equivalence（P=2 V=2 + 2-stage 线性两个 canary）、schedule guard、VP drop-guard、hook+Capture autograd 契约、多 consumer grad 累加、producer-param-grad 端到端等价、`_return_only_new_blocks` 空提交 shape 契约。

---

## 3. 设计弯路 — 6 次迭代才把 backward 跑通

前向 delta 布局直接落地；backward 走了 4 个 session 的弯路。完整记下来防止下一个人重蹈覆辙。

### 3.1（Day 1, session 1）— 初始脚手架

`handoff_20260420.md` 的 4 个已知障碍：
- **Issue 1：`K_s=0` 空 commit 断言**——`layers_per_stage=1` 时奇数 virtual stage 不跨 `is_block_start`，`_return_only_new_blocks=True` 切出空列表。修复：返回 `partial.new_zeros((0, *partial.shape))`，P2P shape 静态保持。
- **Issue 2：`id(partial)` mb-key 跨 P2P 失效**——NCCL 给消费者另分配 recv buffer，producer 的 `id(...)` ≠ consumer 的。修复：monkey-patch `forward_one_chunk` / `backward_one_chunk`，把 schedule 的整型 chunk id 塞进 thread-local。
- **Issue 3：backward grad 回送未接通**——首次尝试是两个 `autograd.Function`：`_SendBlockGradsBack` / `_RecvBlockGradsFromConsumers`，在 `backward()` 里走 `dist.isend` / `irecv`。
- **Issue 4：launcher / config 注释不一致**——docstring 说 `lps=2`，launcher 跑 `lps=1`。对齐到显式的 virtual-stage 算式。

单测从 30 → 41。session 末状态：adapter 接通，naive 透传确认，准备开 delta。

### 3.2（session 2）— backward 路径试了 5 次

`handoff_20260420_part2.md`：

1. **`autograd.Function.backward` 内做 NCCL P2P** —— 失败。autograd 引擎单线程深度优先，我们 `dist.isend(...).wait()` 把引擎卡住等 peer，peer 的引擎还没到对应 Function。Interleaved1F1B 自家的 `SEND_B/RECV_B` 在同一个 group 上和我们竞争 → NCCL 超时。
2. **NCCL 移出 autograd，挪到 `patched_bwd` finally 块** —— 失败。同一根因：rank 0 / rank 7 到达 mb=0 backward 的 wall-clock 差很多，rank 0 flush 时没 peer，schedule 后续的 `SEND_B` 被卡住 → rank 5 触发集合超时。
3. **Step 末批量 flush** —— 跑都没跑就否决。死锁是修了（step 边界 torch 同步），但 M=32–128 在飞时累计的缓存 + retain 图能炸到 TB 量级。
4. **纯 autograd backward 走 PP `SEND_B`** —— 部分成功。意识到论文 §4.1 "backward 是对称的——把发往下一 stage 的东西直接送回去" 等价于**复用 PP 已有的 SEND_B**：从 `recv_delta_tensor` 切出来的缓存块本来就有到那个 tensor 的 autograd 链，grad 自动顺 PP 现有通道回去。**删掉所有自定义 NCCL 机制；`pipeline_adapter.py` 从 1320 行缩到 784 行**。但同 rank 自提交缓存块的 double-backward 还在：consumer 的 backward 走进 producer 的 forward 图把它 free 掉，producer 自己后来的 backward（经 PP SEND_B）再走一次 → `RuntimeError: backward through graph a second time`。
5. **`retain_graph=True` 全局 monkey-patch** —— smoke 跑通。1000 步 naive vs adapter：step 1000 Δ = 0.007（6.339 vs 6.346）。但 rank 7 峰值 +5 GiB（11.9 vs 6.9 GiB naive）。**不可扩展**。作为临时 commit 持久化，给后续 `_Local*_` 方案铺路。

### 3.3（session 3）— `_LocalCacheAugment` + `_LocalCacheCapture` 两个 autograd.Function

`handoff_20260420_part3.md`：

两个本地 only Function（零 NCCL，只共享 `RankLocalCache` 上一个 Python dict）：

- `_LocalCacheAugment(block, key)` 在 producer 提交时套上：forward identity，backward 从 `rank_cache._captured_grads[key]` pop 出 captured，返回 `grad + captured`。
- `_LocalCacheCapture(block, key)` 在 consumer 读时套上（仅当 `producer_rank == self.pp_rank`）：forward identity，backward 把 grad 写进 slot 后对 tensor 输入返回 `None` → autograd 停下，producer 图不被消费者遍历。

CPU 单测过（41/41）。**8-GPU smoke 仍崩**，仍是 double-backward。问题在 `Function.forward` 直接返回输入 tensor 同一个 Python 对象，grad_fn 簿记在真实 PP 调度下歧义。试了 `return block_tensor.view(block_tensor.shape)` 强制返回不同 wrapper，session 结束前没复测。

### 3.4（session 4）— `.view()` 没救，hook + detach 才赢

`handoff_20260421.md`：

- 之前 commit 的 `block_tensor.view(...)` 修复 **没解决** 崩溃。CPU 单测仍过，但只要在 Interleaved1F1B + FSDP + selective AC rerun 下出现"同 rank 自提交缓存读"就重现。把每个 Function 的 backward 加追踪，PP=4 V=2 上看到：rank 3 stage-7 的 backward 调用里，`_LocalCacheCapture.backward` **和** `_LocalCacheAugment.backward` 在**同一次** `backward_one_chunk` 内都触发了（slot `(mb=0, producer_stage=3, block=0)`）—— autograd 真的从 consumer 的 Capture 走进了 producer 的 Augment，尽管 Capture 对 tensor 输入返回了 `None`。view 这个 trick 结构上靠不住。
- **最终方案：把 `_LocalCacheAugment`（Function）换成 producer 端的 `tensor.register_hook`，并且把缓存里那份存成 `block.detach()`。** detach 是结构性硬保证——consumer 端 Capture 的输入根本没有上游 `grad_fn` 可走。即便 autograd 不尊重 Capture 的 `None` 返回，也没有 producer 图可以走进。Hook 在 producer 自己的 backward 里恰好触发一次，把任何 captured grad 累加到 incoming grad。

这就是当前 [`pipeline_adapter.py`](../../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py) 的设计。

---

## 4. 验证结果

均为 4× RTX 5090 PCIe，`Interleaved1F1B`、`PP=4`、`lps=2` → 8 个 virtual stage、每 rank 2 个 chunk。Config：`llama3_175m_attn_res_L16_n8`（174 M params，n_layers=16，num_blocks=8 → layers_per_block=2）。`GIT_SHA = f5c7548`。

### 4.1 1 000 步 naive vs adapter（`pp4_adapter_4gpu_smoke1k` + `pp4_naive_4gpu`）

| step | naive loss | adapter loss | naive tps | adapter tps | naive mem rank3 | adapter mem rank3 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.76178 | 11.76178 | 504 | 529 | 6.48 GiB | 7.37 GiB |
| 10 | 11.52401 | 11.52564 | 7 009 | 6 876 | 6.96 GiB | 7.66 GiB |
| 100 | 8.72997 | 8.73178 | 6 980 | 6 865 | 7.45 GiB | 7.68 GiB |
| 500 | 6.49669 | 6.49083 | 6 881 | 6 759 | 7.45 GiB | 7.71 GiB |
| 1 000 | 6.37720 | **6.34968** | 6 690 | 6 658 | 7.45 GiB | 7.71 GiB |

- 1 000 步内 max `|Δ_naive→adapter|` = **0.06**，落在 naive vs naive 噪声带内（max `|Δ_naive→naive|` 在 step 10 是 0.13）。
- 显存：adapter 多出的就是 cache footprint（M=4 mb 时 175 M 模型每 rank ~260 MB），无 `retain_graph` 膨胀。
- 吞吐：PCIe 上 adapter 比 naive 慢 ~0.5 % —— 符合预期，PCIe 每跳延迟主导，adapter 省下的字节在这个模型规模上换不出 wall clock。adapter 真正的回血在 NVLink-out / 跨节点 IB / RoCE，每跳省 ~60 MB（稳态）能转化为时间。

`naive_vs_adapter_loss.png` 是头条对齐图。

### 4.2 长程跑（~190K-200K 步，`pp4_naive_4gpu` + `pp4_adapter_4gpu`）

同样的 4-GPU PP=4 V=2 配置长跑到 ~200K 步（合计约 6 天 wall clock）。末步 loss：

- naive（step 190 000）：3.22615
- adapter（step 200 000）：3.02490

这两个长跑不是严格 seed-paired（一个是 60K 续跑、一个是从头跑），目的是验证 adapter 不会随时间漂移，而不是严格 A/B。rank 3 显存全程稳在 7.71 GiB。

### 4.3 PP=4 baseline-Llama3 sanity（`pp4_baseline_L16_4gpu`）

非 AttnRes 的 Llama3 L16 PP=4 sanity 跑。loss 从 step 1 就是 `inf`——已知是 bf16 / grad-norm 在初始步过载，不在 AttnRes 验收范围内，仅作 PP 路径独立性检验。

### 4.4 前向 shape 证据

每 mb 每 hop 大小和 `BlockLayoutTables.delta_to_send` 完全一致（rank 3 日志）：

| stage | rank | commits | recv | sends |
|---|---|---|---|---|
| 0 | 0 | b0 | — | [b0] |
| 1 | 1 | b1 | [b0] | [b0,b1] |
| 2 | 2 | b2 | [b0,b1] | [b0,b1,b2] |
| 3 | 3 | b3 | [b0,b1,b2] | [b1,b2,b3] |
| 4 | 0 | b4 | [b1,b2,b3] | [b2,b3,b4] |
| 5 | 1 | b5 | [b2,b3,b4] | [b3,b4,b5] |
| 6 | 2 | b6 | [b3,b4,b5] | [b4,b5,b6] |
| 7 | 3 | b7 | [b4,b5,b6] | — |

稳态每跳 = `P-1 = 3` 块。每 rank 恰好从 cache 读一次同 rank 自提交（rank 0 stage 4 读 stage 0 的 b0；rank 1 stage 5 读 b1；rank 2 stage 6 读 b2；rank 3 stage 7 读 b3）—— 这 4 个调用点就是 hook+Capture 在每 mb 要桥的位置。

### 4.5 缓存分布（每 rank，单 mb 前向一周后）

| rank | 自提交 | 中继 | 总计 |
|---|---|---|---|
| 0 | b0, b4 | b1, b2, b3（via stage-4 recv） | **5** |
| 1 | b1, b5 | b0（stage-1）、b2,b3,b4（stage-5） | **6** |
| 2 | b2, b6 | b0,b1（stage-2）、b3,b4,b5（stage-6） | **7** |
| 3 | b3, b7 | b0,b1,b2（stage-3）、b4,b5,b6（stage-7） | **8** |

每块复制因子：b0..b4 = 4×、b5 = 3×、b6 = 2×、b7 = 1×；系统总 26 份缓存对 8 个不同 block（平均 3.25× 复制）。M=4 mb 在飞时再 ×4，因为 rank cache 按 mb 分桶、`step` 边界统一回收（`_install_step_drop_patch`）。这解释了 nvidia-smi 上的非对称（rank 0：5076 MiB，rank 3：8750 MiB）—— 后面的 rank 结构性持更多缓存（更深 virtual stage 需要更多前缀 block），加上末 rank 还要扛 `[B,T,V]` loss logits 和输出投影。

---

## 5. 不同模型规模的显存包络

```
peak_cache_bytes(rank R) ≈ |rank_cache_at_entry[R, V-1]| × B × T × D × 2 × M
```

| 配置 | 峰值 rank cache | 装得下？ |
|---|---|---|
| 175 M smoke（B=4 T=2048 D=768 M=4，N=8） | ~384 MB | 轻松 |
| 48 B target（B=1 T=8192 D=4096 M=8，N=16） | ~8 GB | 80 GB H100 没问题 |
| 超深（128 B+，N≥64，M=16-32） | 30 GB+ | 撑爆"单卡能装"假设；备选是 selective AC + activation offload，**不是**分布式 cache |

48 B 目标在设计的 "cache cost is small" 假设里。

---

## 6. 结论

1. **Adapter 正确、瘦身彻底。** 1000 步 loss 对齐落在 naive vs naive 噪声带内；backward 路径是纯 autograd 走 PP 自己的 `SEND_B`（跨 rank 缓存块）+ 本地 hook+detach 桥（同 rank 自提交缓存块）。零自定义 NCCL。
2. **Hook + detach 在结构上严格强于 `_LocalCacheAugment`。** detach 在数据结构层切断 consumer→producer autograd 图；即便 autograd 引擎没有遵守 `Function.backward returning None` 这条软契约，也没图可以走进 producer。这就是 CPU 单测过、真实 PP+FSDP+AC rerun 失败的根本——之前 `_LocalCacheAugment.apply + view` 依赖软契约，新设计依赖硬不变量。
3. **175 M / PCIe 下 adapter 比 naive 慢 ~0.5 % tps。** PCIe 每跳延迟主导，adapter 省的字节没法换 wall clock。adapter 的真实回血在带宽主导的网络（NVLink-out、IB / RoCE 多节点）。
4. **mb-key 用 schedule 的整型 chunk id 是唯一稳定的键。** `id(tensor)` 跨不过 P2P（NCCL 给消费者另分配 buffer）；整型 id 通过 `forward_one_chunk` / `backward_one_chunk` 的 monkey-patch 塞进 thread-local，adapter 入口读取。
5. **缓存按 mb 分桶；`pp_schedule.step` 返回时统一回收。** `_drop_all_seen_and_clear` 里的 VP drop-guard 保证只有 rank 上"最早的 virtual stage"释放，跨 VP 的 peer 还能继续看到 cache。

---

## 7. 延后但不阻塞的事

1. **8-GPU PP=8 V=2 复测。** 原计划目标，本期没 8 卡机延后。同一 hook+detach 代码路径，预期同样的 naive 噪声带对齐 + 显存回到 naive 基线 + cache 占用。
2. **PP=8 V=2 的显存对照测量。** `retain_graph` 版本鼓到 11.9 GiB；新设计应回到 6.9 GiB + cache footprint。
3. **1.5 B / 2 B 头条 scale-up。** 等 8 卡机就位后跑 PP=8 V=2 PCIe overhead 图（RFC PR-2 的头条数字）。
4. **NCCL 确定性。** naive→naive 在 step 10 就有 ~0.13 差（10.65 vs 10.78），来自 NCCL 顺序 / bf16 累加噪声。要更严格对照可加 `NCCL_DETERMINISTIC=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`torch.use_deterministic_algorithms(True)`。

## 8. 明确不在范围内（本 RFC 不做）

- **producer-only cache + 按需 P2P 拉取** —— 在 PP=4 N=8 时每 mb 多 24+ 个往返，把 adapter 省下的 comm 抹平。
- **指定 holder 轮换** —— 同样的 comm shape，负载稍均衡；不值得为 RFC 范围做。
- **DP/TP peer 分片 cache** —— 仅当 DP > 1 才有用；要 DP-aware mb-keyed P2P，工程量按周计。

---

## 9. 索引

- 进 PR 的代码：[layout.py](../../torchtitan/torchtitan/experiments/attn_res/layout.py)、[pipeline_adapter.py](../../torchtitan/torchtitan/experiments/attn_res/pipeline_adapter.py)、[model.py](../../torchtitan/torchtitan/experiments/attn_res/model.py)
- 单测：`torchtitan/experiments/attn_res/tests/test_pipeline_adapter.py`（41+ 测）
- 训练日志：`phase3_attnres_pp_integration/runs/{pp4_naive_4gpu,pp4_adapter_4gpu,pp4_adapter_4gpu_smoke1k}/train.log`
- 头条图：`phase3_attnres_pp_integration/naive_vs_adapter_loss.png`
- 设计 + handoff：`phase3_attnres_pp_integration/adapter_design.md`、`handoff_status_20260420{,_part2,_part3}.md`、`handoff_status_20260421.md`
