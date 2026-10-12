# 叠栈分支完整实验 + LoRA PR rebase 与 18 格矩阵（2026-10-12 下午）

用户："这两个分支的改法是什么 现在能叠到昨天copy的那个Elfie分支了吗？本机八卡重新cover一下完整实验。实际上Elfie提到了LoRA，我们有一个LoRA PR 是时候rebase然后重新完整测试了 现在上游有完整5D 有必要对LoRA PR全部cover一遍（18/18矩阵）"；中途："MXFP4 NF4这些都在Elfie copied分支做 不要另外自己造 确认一下"。

## 环境：main 固定的三个包

main `b663c1673` 的 requirements.txt 固定 `spmd_types@2ae49fce3`、`torch_remat@4fd000acb`、`torch_checkpointing@dcbb3a59c`。venv_1006 里是 PyPI 的 spmd_types 0.2.5、torch_remat `d302699b1`，没有 torch_checkpointing。用 venv 原样跑，开类型检查的 CI 格子（b200 `kimi_k3_mm`、`kimi_k3_mm_muon`，h100 两个 CP 格子）第 1 步就在 `torch_remat.recompute_needs_tensor` 里报 `SpmdTypeError: Tensor has Varying type on axis default_pg`（mm 还有 mesh_ep 的 Partial 错）。只换 spmd_types 没用，换上 torch_remat 的固定版本就好了。做法：`uv pip install --no-deps --target scratchpad/main_pins <三个固定版本>`，放在 PYTHONPATH 最前面，不改共享 venv。之前 [[k3-mm-ci-cell-red-on-main]] 记的 "mm 在 main 上红" 很可能也是这个。

上午 #4656/#4963 的对比（`ATTNRES_4656_4963_ON_5033_2026-10-12.md`）用的是 venv 自带的旧 torch_remat，几棵树一样，结论不受影响，但没有用 main 的固定版本。

## 叠栈分支的完整实验

`k3_mxqat_stack_20261012` 在 `98d3c658c`（后来加了 LoRA 提交，见下）对 main `b663c1673`，8 × 5060，main_pins + elfie_ao，seed 42，deterministic，20 步，每对格子在同一份缓存（两棵树各 10 步预热）的副本上跑。kit `kit_stack_full_2026-10-12/`，日志在 scratchpad 的 `stack_full_1012/`。全部 rc 0。

| 格子 | GPU | 和 main 逐位相同的行 | 第一个不同的步 | 第 1 步 loss / grad norm（两边） | 第 20 步 loss main / 栈 | 峰值 GiB main / 栈 |
|---|---:|---:|---:|---|---|---|
| b200 `kimi_k3_mm`（FSDP2 × TP2 × EP2，类型检查） | 4 | 80 / 80 | 无 | 8.20190 / 2.3906 | 3.41525 / 3.41525 | 0.54 / 0.54 |
| b200 `kimi_k3_mm_muon`（FSDP2 × EP2，DistMuon，类型检查） | 2 | 40 / 40 | 无 | 8.21196 / 2.3906 | 4.96127 / 4.96127 | 0.86 / 0.86 |
| h100 CP2 all-gather（类型检查；栈里开着 dynamic CP 256） | 2 | 8 / 40 | 5 | 8.21377 / 2.5938 | 5.54164 / 5.53112 | 0.54 / 0.54 |
| h100 CP2 Ulysses（同上） | 2 | 8 / 40 | 5 | 8.21377 / 2.5938 | 5.54164 / 5.54415 | 0.54 / 0.54 |
| b200 FSDP2 × TP2 × EP2 × PP2 × VPP4（栈里开着 DEP） | 8 | 36 / 160 | 2 | 8.02080 / 1.9609 | 3.06905 / 3.07125 | 0.73 / 0.76 |
| Elfie 的 MX QAT recipe（只有栈） | 1 | | | 8.21498 / 2.5625 | 5.50557 | |

- 两个类型检查的格子 20 步逐位相同：类型检查的 recipe 不开编译，attention residual 走 eager，列表载体在 eager 下本来就和 main 逐位相同。
- CP 格子从第 5 步开始不同：第 5 步是第一张图被 dynamic CP 切开的地方（main 默认关着 dynamic CP），符合 #4380 的设计。
- PP 格子从第 2 步开始不同：栈里这个格子开着 DEP，并且编译开着（#4656 的常量源个数图），两者都会改数值。
- Elfie 的 MXFP4 checkpoint 集成测试（`tests/integration_tests/test_kimi_k3_mxfp4_checkpoint.py`）：1 passed。

## LoRA PR #4576 的 rebase

- #4576（`lora_review2` = `72bbcb639`，基于 09-09 的 main）扩展的是旧的 `components/lora.py`。main 现在的 LoRA 是 `models/common/lora.py`（mixin）加 `config/transform/lora.py`（`LoRATransform`），GroupedLinear、DistMoE、按 FQN 选目标都已经在上游。原 PR 还剩的东西：导出（只导 adapter、合并进底座）、QLoRA（NF4、packed MXFP4 冻结底座）、packed 底座的 TP。
- 用户："MXFP4 NF4这些都在Elfie copied分支做 不要另外自己造"。Elfie 分支上的 MXFP4：MX QAT（bf16 主权重前向假量化）、packed MXFP4 checkpoint 导入（`decode_mxfp4`、packed pair reader、`MXFP4CheckpointPolicy`）、K3 的 MXFP4 配置；没有 packed 冻结底座，也没有 NF4。所以：
  - 我在 main 上写好、测过的 packed MXFP4 底座（40 个 LoRA CPU 测试通过）没有提交，丢掉了，补丁只作记录存在 kit 的 `discarded_own_mxfp4_qlora.patch`；
  - NF4 不做；
  - LoRA 和 MXFP4 的组合只在 Elfie 分支上，用她的 `MXQATTransform` 加 `LoRATransform`（relations 里已经是 MX QAT 在 LoRA 之前，她有一个测试检查 LoRA 包住 QAT 线性层）。
- rebase 后的 LoRA PR = main `b663c1673` 上一个提交 `7d20016a5`（worktree `scratchpad/wt_lora`，没推）：`trainable_state_dict`、`merge_lora_state_dict`。合并按 `state_dict()` 的键改写底座权重（W + (alpha / rank) B A，fp32 算再转回原 dtype），去掉 adapter 键，模型本身不动。不能像原 PR 那样临时换掉模块参数：FSDP2 的 state_dict 前置钩子会把参数换回它自己的 sharded 参数，换进去的合并权重会悄悄丢掉。TP 下 state_dict 是每个 rank 的本地分片，adapter 的切法和底座一致，每个 rank 合并自己的分片。
  - 测试（`tests/unit_tests/cpu/models/common/test_lora.py`，main 原有 29 个加 6 个，35 passed）：可训练参数正好是 adapter；adapter 为零时合并结果等于底座；普通、堆叠、grouped 三种投影的合并公式；合并权重的前向和 LoRA 模块一致（fp64）；检查点包装器路径；2 rank gloo 的 TP 下每个 rank 的合并分片等于不切分合并的对应切片。
  - flake8、ufmt 干净。
- 叠栈分支加了它的压缩提交 `992ae1657`，另加一个测试：合并把 adapter 折进 MX QAT 的主权重。栈上 LoRA、MX QAT、transforms 的 CPU 测试 85 passed（加测试后 MX QAT 13 passed）。栈已快进推到 `992ae1657`。
- `lora_review2` 是已发布 PR 的头，没推，要用户的话。

## 18 格矩阵（运行中）

kit `kit_lora_mm18_2026-10-12/`（`mm18_probe.py`、`run_mm18.sh`）。main 现在只能用 Python 配置，原来的命令行 `--parallelism.*` 和 `mx4ckpt` 都不能用了：每格的并行度从 `MM18_CELL` 读，recipe 在 tp = cp = 1 下构建（recipe 构建时会检查 DistMuon 对 TP、attention 对 CP），之后再设 tp、cp，CP 格子套 `ContextParallelTransform`（KV all-gather + head-tail 负载均衡，KDA 用 CP inner）。K3 debug 模型（main 的 17 层），cc12m-test，seq 256，每步 4096 token、每个 dp rank 每个 micro-batch 256 token，PP 8 个 micro-batch，AdamW，seed 42，deterministic，10 步。每组先单卡写 seed checkpoint，每格只加载模型并检查日志里的加载行。

main 要求 MoE 模型 ep ≥ tp，所以凡是有 tp 的格子都带 ep；原来的 `fsdp2_tp2_pp2`、`fsdp2_tp2_cp2` 在 main 上就和带 ep2 的格子一样了，改成 ep4。18 格：dp1、fsdp2、pp2、tp2×ep2、ep2×fsdp2、cp2、pp4、tp4×ep4、cp4、pp8、ep8×fsdp8、ep4×fsdp2×tp2×pp2、ep2×fsdp2×tp2×pp2、fsdp2×pp2×cp2、ep2×fsdp2×pp2×cp2、ep4×fsdp2×tp2×cp2、ep2×tp2×pp2×cp2、ep2×fsdp2×tp2×cp2。三组在两棵树上都能构建（18 / 18）。

组：main 上 base、lora；栈上 base、lora、lora_mxqat（Elfie 的策略选出 32 个 routed experts 投影做 MX QAT，LoRA 加在 284 个投影上）。第一版调度把单卡的 dp1 放到了两张卡上，main 的两组 dp1 要补跑。
