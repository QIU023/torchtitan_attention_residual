# 结果目录索引（8 × RTX 5060 Ti 实测，2026-09-26）

所有格子都是生产切分：93 层、block 12、pp8 × vp4，dim 2048，seq 2048，M16，FullAC，seed 42。每个目录里的文件：
- `rank*.json`：每步的峰值记录。`records[0]` 是训练前 titan 的那次重置，所以 `records[k]` 是第 k 步。
- `actions*.json`：逐动作记录（诊断格子才有）。
- `plan*.json`：计划和统计（offload 与 balance 的格子才有）。
- `overlap.md`：trace 分析（计时格子才有）。
- `steps.txt`：每步的 loss 和 grad norm。

| 目录 | 代码 | 说明 |
|---|---|---|
| `s1_mem_base`、`s1_time_base` | 4312 `7814d1f8b` | 基线 |
| `s1_mem_v4`、`s1_time_v4` | V4 `9ac65f173` | |
| `s1_mem_pra`、`s1_time_pra` | PR A 修复前 | 有 `_input_grads` 没清空的问题，已被 `s2_*_pra` 取代 |
| `s2_mem_pra`、`s2_time_pra` | PR A `07687ead7` | 定稿 |
| `s2_mem_mgr`、`s2_time_mgr` | 第二步修复前 | 没有后端，结果和修复后逐字节相同 |
| `s2_mem_obl`、`s2_time_obl` | 第三步第一版 | 有生命周期和 `_drain` 两个问题，已被 `s3_*_obl` 取代 |
| `s3_mem_mgr`、`s3_time_mgr` | 第二步 `7b43e1d01`（树同 `2b7b8cd84`） | 定稿 |
| `s3_mem_obl`、`s3_time_obl` | 第三步 `37967996b`（树同 `e88be9bfa`） | 定稿 |
| `diag_v4`、`diag_pra` | V4 和 PR A | 逐动作记录。当时追踪的实际是第 4 步（`PPMEM_ACTION_TRACE` 的步数差一，后来修了） |
| `diag_obl` | 第三步第一版 | 逐动作记录，追踪第 4 步 |
| `diag_obl2` | 第三步第一版，加上计划输入的导出 | 逐动作记录，追踪第 5 步；计划的输入在 scratchpad 的 `plan_inputs*.pkl`，没有拷进来 |

## 2026-09-27：CheckpointPolicy 重构后（base 是 4312 `ffdd169ef`，main `d0f3bbfd6`）

- 这一批用本地 torch 兼容补丁跑（`../kit_pp_review5_rebase_2026-09-26/pr5_torch_compat_shim.patch`）；KDA 不再需要放宽。
- 探针开关：`PPMEM_CPU_OFFLOAD=none|all|planned`、`PPMEM_BALANCE=1`、`PPMEM_HOST_GBPS=10`、`PPMEM_PEER_GBPS=2`。
- 显存跑 10 步（第 3 到 10 步每个 rank 的峰值相差不超过 0.01 GiB），计时抓第 8 步。
- 五个格子共用一条 cache 血缘：s4 的四格预热 cache0，s5 在同一个 cache0 上再预热一步。
- `rank*.json` 的每条记录里多了 `storage`：存储的累计计数（各后端搬走的字节、`late_fetches`、`off_device_bytes`）。

| 目录 | 代码 | 说明 |
|---|---|---|
| `s5_mem_base`、`s5_time_base` | 4312 `ffdd169ef` | 同 base 的基线 |
| `s4_mem_pra`、`s4_time_pra` | PR A `7ae870508` | |
| `s4_mem_all`、`s4_time_all` | #4765 `7c0f5cd3c`，`cpu_offload=all` | |
| `s4_mem_plan`、`s4_time_plan` | #4764 `e6241b78b`，`cpu_offload=planned` | host 带宽按 10 GB/s 预算 |
| `s4_mem_planbal`、`s4_time_planbal` | #4764 `e6241b78b`，planned 加 `balance` | 对端带宽按 2 GB/s 预算；5060 没有 RDMA，池在对端 host 内存里 |

## 2026-09-27：balance 按所有 rank 之间的差来看（本地改 recipe）

- 配置：pp8 × vp4（93 层，block 12，`PPMEM_LPS=3`），cache 开，**关 AC**，dim 1024，seq 512，M16；只测显存，8 步，取第 5 步。
- 本地补丁 `../balance_probe_local.patch`（不进任何提交），打在 #4764 的树上：
  - `PPMEM_EMULATE_DEVICE_POOL=1`：目的 rank 在 GPU 上占住池的大小，源 rank 占住 staging 的大小，模拟 RDMA 下池放在设备上的显存；数据仍走 mooncake 的 TCP，所以只看显存，不看时间；
  - `PPMEM_PLAN_NO_HOST=1`：plan 只能停到别的 rank，不去 host。
- 四个格子共用一条 cache 血缘（`balance_campaign.sh`），汇总 `balance_table_b1.md`（`balance_table.py`）。

| 目录 | 代码 | 说明 |
|---|---|---|
| `b1_mem_pra` | PR A `7ae870508` | 基线 |
| `b1_mem_bal` | #4764 `e6241b78b`，planned 加 balance，不去 host，模拟设备池 | 只 balance |
| `b1_mem_off` | #4764，planned | 只 host offload |
| `b1_mem_both` | #4764，planned 加 balance，模拟设备池 | 两者都开 |

## 2026-09-27 晚：PR A 叠到 #4656 之上以后（base 是 main `f35966713` 加 #4656）

- 配置同 s4/s5：93 层、block 12、pp8 × vp4（`PPMEM_LPS=3`），dim 2048，seq 2048，M16，FullAC，seed 42；本地 torch 兼容补丁照旧。
- 两格共用一条 cache 血缘（`campaign2.sh s6`：cache0 由两格各跑一步预热，每格拷一份）。显存跑 10 步，计时抓第 8 步（计时格只跑 8 步，学习率调度按 8 步算，所以第 5 步起的 loss 和 10 步的显存格不同；两格计时之间逐位一致）。
- 结果：最大 / 平均 11.47 / 10.80 → 7.12 / 6.23 GiB，10 步 loss 和 grad norm 逐位一致；第 8 步窗口（各 rank 平均）25.10 → 24.94 s，计算 7.45 → 7.38 s。
- `s6_*_b4656` 和 s5 的 main 基线不是同一条 cache 血缘，也不是同一个 base（s5 是 `ffdd169ef`），两者不直接比。

| 目录 | 代码 | 说明 |
|---|---|---|
| `s6_mem_b4656`、`s6_time_b4656` | #4656 `attnres_review1` `f14d681f4` | PR A 的 base |
| `s6_mem_pra`、`s6_time_pra` | PR A `pp_review_optimize` `e8d0a4aec` | |
