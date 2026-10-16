# 栈上 DEP 和动态 CP 一起开（2026-10-15）

> **10-16 更正：** 栈的基底 `b663c1673` 已含 main 的 #5266（视觉塔 CP-invariant，FSDP 只在 dp_shard 上切，视觉输出梯度在 CP 上 all-reduce）。动态 CP 让各 CP rank 只编码自己那一段，和这个前提冲突，详见 `DYNCP_VS_5266_2026-10-15.md`。按代码推断有两个后果：切开的特征梯度被放大 CP 倍；各 CP rank 上的 tower 副本从第 1 步起分叉。所以本文里开了动态 CP 的格子只证明能跑完、第 1 步 loss 相同，正确性没有被证实。第 2 步起的差此前写成"切分的 bf16 舍入"，没有定位，撤回。GPU 探针（CP2，阈值 128，比较两个 CP rank 上的 tower 参数）10-16 已跑，诊断成立。修复已经折进本地栈 `2eff57f99`，还没有推送；本文的格子都是在修复之前跑的。详见 `DYNCP_5266_FIX_2026-10-16.md`。

## 来由

用户 10-15："栈分支上我就是让她开动态CP和DEP PP的"。栈上 DEP 以前只在 b200 的 PP 格子（pp2 × vpp4，不带 CP）里开过，动态 CP 只在 h100 的 CP 格子（不带 PP）里开过，两个从来没有一起跑过。

## 格子（本机 8 x RTX 5060 Ti，栈 `a89d39417`，venv_1006 + `scratchpad/main_pins`）

- 布局：dp2 × ep2 × pp2 × cp2；Interleaved1F1B，每 rank 4 个 stage，用 core 的切分；CP 用 all-gather KV 和 HeadTail。
- 数据：cc12m-test 原图尺寸（`max_patches` 2304、每边 64），大图在 2160 patch 左右，超过切分阈值 256。
- 设置：seq 2048，每步 4 个 micro-batch，AdamW，seed 42，deterministic，同一个 seed checkpoint，同一份缓存（`out1/cache`），10 步。
- kit：`kit_2026-10-15/combo/`（`combo_cells.py`、`run_combo.sh`、`run_probe.sh`、`probe/sitecustomize.py`、`replica_dyncp_probe.patch`）。

| 格子 | 第 1 步 | 第 2 步 | 第 3 步 | 第 5 步 |
|---|---|---|---|---|
| 对照 | 8.07753 | 7.27177 | 5.60801 | 4.53794 |
| 只开 DEP | 8.07753 | 7.27174 | 5.60817 | 4.53652 |
| 只开动态 CP | 8.07767 | 7.27278 | 5.61251 | 4.53935 |
| 两个都开（栈原样） | 8.07753 | 7.27174 | 5.60817 | 4.53652 |
| 两个都开 + 副本补丁 | 8.07767 | 7.27276 | 5.61183 | 4.54046 |

- 4 格加补丁那格都是 rc 0，跑满 10 步，8 个 rank 都加载了 seed。
- 栈原样两个都开时，10 步的 loss 和 grad norm 在日志精度（loss 5 位小数）下全部和只开 DEP 相同：动态 CP 没有生效。
- 第 2 步起的差：只开 DEP 和对照之间也有（7.27174 对 7.27177），是已知的 DEP 第 2 步差（tower 和文本共用的 flex 重编译）。

## 原因

- `MoonViTCPEncoder.forward` 只在 `self._cp_subgroups` 非空时切图。子组由 `KimiK3Model.parallelize` 里的 `install_vision_cp` 设到 stage 0 的 tower 上。
- DEP 的副本由 `pipeline_kimi_k3` 调 `build_vision_replica` 生成：在 `pipeline_llm`（会调各 stage 的 `parallelize`）之前，从未并行化的模型 deepcopy，只做 `_parallelize` 和 AC，没有调 `install_vision_cp`。
- 开 DEP 后，所有编码都走副本（前向在 `no_grad` 下，反向重算），stage 0 的 tower 不再编码，所以切分永远不发生。
- 探针（`K3C_PROBE=1`，每个 tower 实例第一次 forward 时打印子组数）：
  - 栈原样：rank 0 到 3 各有两个 tower，stage 0 的子组数是 1，副本是 0；rank 4 到 7 只有副本，子组数是 0。
  - 补丁后：12 个实例的子组数全是 1。

## 修法（本地验证过，已撤回，没有提交）

- `pipeline_kimi_k3` 建好副本后，在 `activate_spmd()` 里给副本也调 `install_vision_cp(replica, model_config.vision_encoder, parallelism_context)`。
- 改的是 K3 自己的流水线构建函数，由模型把两个机制组合起来；`vision_dep` 包不引用 `vision_cp`。
- 没开 CP 或没开动态 CP 时 `install_vision_cp` 直接返回，所以只开 DEP 的格子（包括 b200 的 CI 格子）不受影响。
- 建子组是集合操作，每个 rank 都会走 `pipeline_kimi_k3`，次序一致。
- 预测：补上后第 1 步和只开动态 CP 相同。实测 8.07767 对 8.07767（日志精度），成立。

## 折进栈（10-15，用户："把这一行折进栈上的 DEP 提交 8aed70043（照 10-14 的做法，先推备份分支再 force-push）"）

- 备份：`backup/k3_mxqat_stack_20261012_pre_dep_dyncp_fix` = `a89d39417`，先推，推之前核对过远端 ref。
- 重建：
  - 在 `8aed70043` 上打补丁，ufmt 把 import 移到第一方块里。
  - `--amend --no-edit` 得到 `d44bd7e25`。
  - 再 cherry-pick 后面 5 个提交，没有冲突：`d34cc462f`（#4656）、`d2eb5a9c2`（#4963）、`6c861cd15`、`ed6c2ddd8`、`c7ca6fb1a`（LoRA）。
- 核对：
  - `git diff a89d39417 c7ca6fb1a` 只有这 5 行。
  - 5 个 cherry-pick 的 patch-id 逐个和原提交相同。
  - flake8 通过。
  - DEP 的 CPU 测试 22 passed（63 subtests）。
- 复核格（新树，同一个 seed，同一份缓存）：
  - 只开 DEP：10 步在日志精度下和旧栈相同，修法在不开动态 CP 时什么都不做。
  - 两个都开：第 1 步 8.07767，和只开动态 CP 相同；10 步在日志精度下和本地补丁那次相同。
- 推送：`--force-with-lease` 以 `a89d39417` 为前值，`k3_mxqat_stack_20261012` = `c7ca6fb1a`。

## 待用户决定（已决定：折进 DEP 提交）

- 是否把补丁折进栈上的 DEP 提交 `8aed70043`：照 10-14 的做法，先推备份分支，再 force-push。
- 也可以单独加一个提交。
- PR 侧：#4380 和 #4381 现在互不包含，修法只在两者都在的树上才有意义。后合的那个要带上这一处。
