# Kimi K3：CP 下 DistMuon 的 dense layout（修 main 的 h100 CP 格子），2026-10-07

## 状态（不粘贴）

- **10-08 CPU 会话复核（用户贴来 GPU 会话的说明后说"检查"）：**
  - 说明里的事实都对：main 从 `948d65c86` 到 `08f7c391b` 多了 19 个提交，没有一个碰 `kimi_k2_7.py`、`kimi_k3.py`、`flex_shard/`、h100 的 suite 或 integration 定义，三个相关文件和 `948d65c86` 上的逐字节相同；main 的 recipe 和 `dist_muon.py` 里都没有 `dp_shard_cp`；`cfadefab9` 和 main 合并没有冲突。
  - 粘贴区改了两处：
    1. Design 原来是一段三句话，不符合 10-07 的 "Design sections are nested bullets"，改成带标签的嵌套 bullet，内容不变；
    2. H100 那行原来写 "On 4x H100"，实际是 2 张卡（`results_h100_1008/muonfix/cp/fix_ag/run.log` 里 `--nproc_per_node=2`），改成 "On 2 H100s"，并拆成两条。
  - 改后：没有 we/our/us、破折号或非 ASCII 字符，约 240 词。

- **为什么要这个 PR：**
  - main 的 h100 格子 `kimi_k3_mm_allgather_kv_cp` / `kimi_k3_mm_ulysses_cp`（#4639 带进来的）在建 optimizer 时就失败。10-07 的 H100 nightly（main `7a5bedba3`）两格都是 rc=1，5060 上原样复现。
  - #4380 唯一的 CI 覆盖就是这两格，它们不能训练，#4380 跑 ciflow/h100 也是红的。
- **分支：** `k3_cp_muon_layout` = `cfadefab9`（fork，从 main `948d65c86` 拉出，一个提交，已推）。PR 要你在 GitHub 上开：base `pytorch/torchtitan:main`，head `QIU023:k3_cp_muon_layout`。
- **5060 验证（10-07，kit `kit_cpmm_2026-10-06/muon_probe.py`，保留 DistMuon，seed 42，deterministic，同一份 cache）：**
  - 不开 CP（`kimi_k3_debugmodel`，fsdp2）：main 跑两次、修复版跑一次，10 步轨迹逐位相同；
  - CI 的两个 CP recipe：main 在建 optimizer 时失败，修复后 all-gather 和 Ulysses 都训练满 10 步（两者 10 步相同，all-gather 重跑一次也相同）；
  - 叠上 rebase 后的 #4380（`9b03b4af1`）：两个 CP recipe 都训练满 10 步，1 到 4 步和"main + 修复"逐位相同，第 5 步第一张 256 patch 的图被切以后不同。说明 #4380 的切分路径在 CI 的配置（typecheck 开，DistMuon）下能跑。
  - 不加 seed 的话 recipe 每次第 1 步都不同（main 自己两次就是 8.14823 对 8.18923），所以对照一定要用 seed。
- **上游相关：**
  - #4448（jinsooihm，K2.7 CP，09-04 以后没动）用的是同一个办法：两个轴都声明。
  - #4353（shuhuayu，DistMuon TP storage layout）改 `dist_muon.py`；这个 PR 只改 recipe，不和它冲突。
- **10-08 H100 冒烟（box 115.124.123.240 -p 30797，`run_muonfix_h100.sh`）：** main 的 all-gather CP recipe 在建 optimizer 时报同一个 layout 错误；修复后 all-gather 和 Ulysses 都训练满 10 步且相同；叠上 #4380 也训练满 10 步；不开 CP 时 main 两次和修复一次 10 步逐位相同。Test plan 加了一条 H100 的结果。日志 `kit_cpmm_2026-10-06/results_h100_1008/muonfix/`。
- 粘贴区已检查：没有 we/our/us，没有破折号。

标题：`[Kimi K3] Declare the DistMuon dense layouts on the CP-flattened axis`

--- PR body: PASTE BEGIN ---

## Summary

Kimi K3's DistMuon layouts name only `dp_shard`, so under context parallelism, where FSDP stores dense parameters on `dp_shard_cp`, the optimizer fails to build and the h100 cells `kimi_k3_mm_allgather_kv_cp` and `kimi_k3_mm_ulysses_cp` fail before step 1 (`Muon compute layout for parameter 'layers.0.delta_attention.q_proj.weight' declares no axis in storage mesh ['dp_shard_cp']; declared axes: ['dp_shard']`, in the H100 nightly of 2026-10-07).

- `_dense_compute_layout` (`torchtitan_recipes/tests/models/kimi_k2_7.py`) declares one compute sharding on `dp_shard` and on `dp_shard_cp`.
- Kimi K3's dense Muon layouts and the shared ep = 1 routed-expert layout use it.

## Design

- Storage axis: FSDP flattens its shard dims into one axis named by joining them, so with CP the dense storage axis is `dp_shard_cp`.
- Both axes declared:
  - The recipes build the optimizer config before the CP cells set `context_parallel_degree`, so the layout cannot read the degree.
  - `ComputeLayout` allows declarations for mesh variants a parameter does not use and resolves the one its storage mesh has.
  - Runs without CP resolve the same layout as before.

## Test plan

- `ciflow/h100.8`: `kimi_k3_mm_allgather_kv_cp` and `kimi_k3_mm_ulysses_cp`, which fail at optimizer build on main.
- `MODULE=torchtitan_recipes.tests.suites.h100 CONFIG=kimi_k3_debugmodel_mm_allgather_kv_cp2 NGPU=2 ./run_train.sh`, and the same with `kimi_k3_debugmodel_mm_ulysses_cp2`, run their 10 steps.
- On 2 H100s (torch 2.15.0.dev20260906+cu126, seeded):
  - Both recipes fail at optimizer build on main and train their 10 steps with this PR, all-gather and Ulysses identical.
  - Without CP, `kimi_k3_debugmodel` matches main bitwise for 10 steps.
- `pytest tests/unit_tests/cpu/flex_shard tests/unit_tests/cpu/test_integration_test_definitions.py tests/unit_tests/cpu/test_debug_config_defaults.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_skip_dp.py -q` (87 passed).

--- PASTE END ---
