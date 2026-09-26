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
