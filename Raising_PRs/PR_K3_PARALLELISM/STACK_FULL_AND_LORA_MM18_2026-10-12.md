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
- （第一版，最终版见下一节）rebase 后的 LoRA PR = main `b663c1673` 上一个提交 `7d20016a5`（worktree `scratchpad/wt_lora`，没推）：`trainable_state_dict`、`merge_lora_state_dict`。合并按 `state_dict()` 的键改写底座权重（W + (alpha / rank) B A，fp32 算再转回原 dtype），去掉 adapter 键，模型本身不动。不能像原 PR 那样临时换掉模块参数：FSDP2 的 state_dict 前置钩子会把参数换回它自己的 sharded 参数，换进去的合并权重会悄悄丢掉。TP 下 state_dict 是每个 rank 的本地分片，adapter 的切法和底座一致，每个 rank 合并自己的分片。
  - 测试（`tests/unit_tests/cpu/models/common/test_lora.py`，main 原有 29 个加 6 个，35 passed）：可训练参数正好是 adapter；adapter 为零时合并结果等于底座；普通、堆叠、grouped 三种投影的合并公式；合并权重的前向和 LoRA 模块一致（fp64）；检查点包装器路径；2 rank gloo 的 TP 下每个 rank 的合并分片等于不切分合并的对应切片。
  - flake8、ufmt 干净。
- 叠栈分支加了它的压缩提交 `992ae1657`，另加一个测试：合并把 adapter 折进 MX QAT 的主权重。栈上 LoRA、MX QAT、transforms 的 CPU 测试 85 passed（加测试后 MX QAT 13 passed）。栈已快进推到 `992ae1657`。
- `lora_review2` 是已发布 PR 的头，没推，要用户的话。

## LoRA PR 的最终提交（傍晚，矩阵抓出三个 bug 之后）

上面一节写的 `7d20016a5` 是第一版。18 格矩阵在 LoRA PR 上抓出三个 bug，三个提交都改写过（都没推）：

| 提交 | 内容 |
|---|---|
| `8393252ad` | lora: export the adapters alone or folded into the base weights（合并改成在本地分片上算；GPU 测试 `tests/unit_tests/gpu/models/common/test_lora.py`） |
| `91212bd8b` | lora: adapters keep the base weight's placement outside TP |
| `b61c686dc` | optim: a pipeline rank with nothing to train still steps（容器、LR scheduler、`clip_grad_norm_`；CPU 测试 2 个，GPU 测试 `tests/unit_tests/gpu/distributed/test_clip_grad_norm.py`） |

测试：CPU 200 passed（test_lora、components/optim 整个目录、transforms、dist_moe、hi_mid_lo、distributed utils、invalid_loss；`test_lora_forward` 照旧除外），GPU 2 passed。三处新注释都是约束说明，保留。

栈上对应两个新提交：`db3bdac93`（placement），`0fa53c2e3`（导出 + 空 rank），都已快进推到 fork。栈上 lora.py、optimizer.py、lr_scheduler.py、distributed/utils.py 和 PR 头逐字节相同。

### main 上 LoRA 构建就失败

main 的 `_adapter_sharding` 只认 `dense_param_placement(tp=S(0))` 和 `dense_param_placement(tp=S(1))`，别的底座直接断言失败。默认全目标的 LoRA（`lora_build_all.py`，meta 上构建）：main 上只有 llama3、muse_glimmer 能构建，deepseek_v3、deepseek_v4、gpt_oss、qwen3_5/6/8、kimi_k2_5、kimi_k3 全部失败；改完都能构建。qwen3 的失败是 main 自己的权重共享检查（lm_head），和这里无关。挡住的底座（`lora_base_placements.py`）有两类：

- TP 不切的投影，TP 上是 R（不开 SP 时是 I）：router gate、MLA 的 `wkv_a` / `wq_a`、DSV4 的 compressor / indexer、K3 的 `attention_res_proj` / `ffn_res_proj` / `forget_a` / `routed_up` / `routed_down`；
- 视觉塔的列 / 行并行线性层：CP 上是 I，不是 R，和 dense 的判等不成立。

### bug 1：视觉塔 adapter 的 rank 维被 TP 切开

第一版 placement 提交把“不是 dense S(0) / S(1) 的底座”一律给两个 adapter 都用底座的 placement。视觉塔的列并行底座落进这个分支，`lora_a` 的 rank 维被 TP 切成一半，TP 格子第一步就报 `mm: mat1 and mat2 shapes cannot be multiplied (192x4 and 8x256)`（tp4 是 192x2）。现在按底座 placement 里 TP 那一项分类：S(0) / S(1) 走列 / 行规则，R / I 两个 adapter 都用底座的 placement，其他情况报错；需要复制的那个 adapter 用底座的 placement，只把 TP 换成 R，所以 DP、CP 跟底座一致（dense 底座时就是原来的 `dense_param_placement(tp=R)`）。

### bug 2：stacked w13 在 FSDP × TP 下导出失败

训练 10 步正常，导出检查报 `Cannot unflatten tensor with _StridedShard placement`。出错的是 stacked 的 dense `w13`（layer 0 的 `feed_forward.w13`、每层的 `moe.shared_experts.w13`）：在 `('dp_shard', 'tp')` 上的 placement 是 `(_StridedShard(dim=1, sf=2), Shard(1))`，DTensor 的 matmul 先 flatten 再 unflatten，DTensor 做不了 unflatten。routed experts、视觉塔和其他模块都正常（探针 `kit3/merge_probe.py`）。现在在本地分片上算：B 和底座的输出行切法相同，A 只在 FSDP 切了它的 rank 维时 all-gather（张量很小），结果用底座的 placement 包回 DTensor，不 gather 任何底座权重。新的 4 卡 GPU 测试（llama3 debug，TP2 × FSDP2，和 full tensor 算的期望比）在旧代码上复现了这个错误，在新代码上通过。CPU 上那个 TP 测试只调 `_parallelize`，参数是本地张量，走不到 DTensor，所以之前没测出来。

### bug 3：整个 rank 冻结时优化器、scheduler、梯度裁剪都挂

第一版优化器提交只跳过冻结的 model part。只给 MLA 投影加 LoRA（arm `lora_mla`）时，K3 debug 的 8 段切分里 stage 0、3、5 只有 KDA 层：pp4 × vpp2 下 rank 0、1、3 各有一个冻结 stage 加一个可训练 stage，pp8 下 rank 0、3、5 整个冻结。实测（`run_frozen.sh`，同一个 seed）：

| 树 | pp4 × vpp2 | pp8 |
|---|---|---|
| 没有优化器提交（`36dcfcd8c`） | `Optimizer pattern '.*' matched no parameters` | |
| 第一版优化器提交（`7f3069a6a`） | 10 步，导出检查通过 | `optimizer got an empty parameter list` |
| 补了容器和 scheduler | 10 步 | `No backend type associated with device type cpu` |
| 再补 `clip_grad_norm_`（最终版） | 10 步 | 10 步，8 个 rank 都存了 checkpoint，导出检查通过 |

pp8 和 pp4 × vpp2 第 1 步的 loss 和 grad norm 都是 8.10062 / 0.1309，冻结的 rank 在 PP 归约里贡献 0。`clip_grad_norm_` 那一处：没有梯度的 rank 上 `get_total_norm([])` 返回 CPU 上的 `tensor(0.)`，NCCL 的 all-reduce 不接受；main 的注释写着支持“没有梯度的 PP rank”，实际不通。现在在 `device_type` 上放一个 0。

## 18 格矩阵

kit `kit_lora_mm18_2026-10-12/`（`mm18_probe.py`、`run_mm18.sh`）。main 现在只能用 Python 配置，原来的命令行 `--parallelism.*` 和 `mx4ckpt` 都不能用了：每格的并行度从 `MM18_CELL` 读，recipe 在 tp = cp = 1 下构建（recipe 构建时会检查 DistMuon 对 TP、attention 对 CP），之后再设 tp、cp，CP 格子套 `ContextParallelTransform`（KV all-gather + head-tail 负载均衡，KDA 用 CP inner）。K3 debug 模型（main 的 17 层），cc12m-test，seq 256，每步 4096 token、每个 dp rank 每个 micro-batch 256 token，PP 8 个 micro-batch，AdamW，seed 42，deterministic，10 步。每组先单卡写 seed checkpoint，每格只加载模型并检查日志里的加载行。

main 要求 MoE 模型 ep ≥ tp，所以凡是有 tp 的格子都带 ep；原来的 `fsdp2_tp2_pp2`、`fsdp2_tp2_cp2` 在 main 上就和带 ep2 的格子一样了，改成 ep4。18 格：dp1、fsdp2、pp2、tp2×ep2、ep2×fsdp2、cp2、pp4、tp4×ep4、cp4、pp8、ep8×fsdp8、ep4×fsdp2×tp2×pp2、ep2×fsdp2×tp2×pp2、fsdp2×pp2×cp2、ep2×fsdp2×pp2×cp2、ep4×fsdp2×tp2×cp2、ep2×tp2×pp2×cp2、ep2×fsdp2×tp2×cp2。三组在两棵树上都能构建（18 / 18）。

组：main 上 base、lora；栈上 base、lora、lora_mxqat（Elfie 的策略选出 32 个 routed experts 投影做 MX QAT，再套 LoRA；导出检查数到的 LoRA 模块和 lora 组一样是 290 个）。第一版调度把单卡的 dp1 放到了两张卡上，main 的 dp1 补跑了；cp4 在 seq 256 下不满足 seq % (cp × 128) == 0，改成 512 后补跑（cp4 读的数据和别的 dp1 格子不同，只和别的组的 cp4 比）。

### 结果（10-12 晚，全部在 8 × 5060，main_pins）

判定（`gate_mm18.py <组> <每个模型的 LoRA 模块数>`）：每格 rc 0、跑满 10 步、日志里有 seed 加载行；LoRA 组还要求每个 rank 都打出 `EXPORT_CHECK`，worst 为 0，合并的模块数 = 290 × 卡数 / pp。

| 组 | 树 | 结果 |
|---|---|---|
| main base | `b663c1673` | 18 / 18（dp1、cp4 补跑） |
| main lora | `b663c1673` | 构建就失败：`_adapter_sharding` 的断言（见上面“main 上 LoRA 构建就失败”） |
| 栈 base | `5bdeb1d63` | 18 / 18；每格第 1 步 loss 和 main 同一格逐位相同（dp1 组 8.12572，dp2 组 8.11338，ep8×fsdp8 8.12882，cp4 8.13709） |
| LoRA PR | `b61c686dc` | 18 / 18，导出检查 18 格全部 worst 0 |
| LoRA PR 冻结 stage（`lora_mla`） | `b61c686dc` | pp4 × vpp2、pp8 都跑满 10 步，3 个冻结 part / 3 个整 rank 冻结，导出检查通过；第 1 步 8.10062 / 0.1309 两格相同 |
| 栈 lora_mxqat | `0fa53c2e3` | 17 / 18 按逐位检查；ep8×fsdp8 的 1 bf16 ulp 已定位（下），按本地批次对齐的检查重跑 18 / 18 |
| 栈 lora | `0fa53c2e3` | 18 / 18，逐位检查，导出检查 18 格全部 worst 0；每格第 1 步和 LoRA PR 组同一格逐位相同（dp1 组 8.13623，dp2 组 8.18727，ep8×fsdp8 8.16220） |

LoRA PR 组第 1 步：dp1 组（dp1、pp2、pp4、pp8）8.13623 逐位相同，TP 格子差 ≤ 2.3e-5，CP2 3.7e-6；dp2 组（fsdp2、ep2×fsdp2）8.18727，带 TP/PP/CP 的格子差 ≤ 4.2e-5；和 base 组 TP 格子的差同一量级（base tp4×ep4 是 2.1e-5）。LoRA 组的 seed 是 LoRA 模型自己初始化的，底座权重和 base 组不同，所以第 1 步和 base 组不比。

ep8×fsdp8 在 lora_mxqat 组的那 1 个 ulp（`kit3` 的 `mm18_export_locate.py` 定位）：只有 `layers.12.moe.routed_experts.w2`（`LoRAMXQATGroupedLinear`，bf16，8 卡上每个 rank 1 个专家），差的元素上底座 0.00224 和 delta 抵消到 −2.4e-5，合并结果和期望差 1.19e-7 = 那个量级的 1 个 bf16 ulp（第一次跑是另一个元素，4.77e-7）。期望值改成按本地批次（每次 1 个专家）做 bmm 后差为 0：sm120 上 cuBLAS bmm 的结果随 batch 数变，检查在 full tensor 上一次算 8 个专家，合并在每个 rank 上算 1 个，fp32 末位不同，碰到 bf16 舍入边界翻一位。合并没有错。`mm18_train_export.py` 的分组期望现在按合并的本地专家数分批算（placement 错了照样会报），用它重跑这一格：8 个 rank worst 0，第 1 步 8.16249 和原来一样。

### LoRA PR 推送

用户（10-12）："强推 如果18/18全过"。18 / 18 和两个冻结格子都过之后，一条 `&&` 链：确认远端 `lora_review2` = `72bbcb639`，推备份 `backup/lora_review2_pre_20261012` = `72bbcb639`，再 `--force-with-lease` 推 `lora_review2` = `b61c686dc`。PR #4576 现在 3 个提交，9 个文件，+531 / −19。PR 的标题和 body 还是 09-10 的（QLoRA），body v2 在 `PR_BODY_LORA_v2_2026-10-12.md`，等用户贴。

### 10-13：pyrefly 和 #4656 的 PR 分支

- 按 CI 的方式（pyrefly 0.45.1，整个仓库，只读）跑：`b61c686dc` 比 main 多 3 个错误，都在 `_merged_weight` 里（adapter 被当作 `Parameter`，`redistribute` / `placements` / `to_local` 报错），CI 的 lint 会挂。本地提交 `fb4079831`：底座是 DTensor 时断言两个 adapter 也是；pyrefly 回到 main 的 19 个错误，LoRA CPU 39 passed、GPU 合并测试 1 passed。没推，`lora_review2` 仍是 `b61c686dc`，推要用户的话。栈上同一行是 `ab65ad8b8`（已推）。只是类型断言，不改数值，矩阵不重跑。
- 用户："先不同步，要等shuhuayu review"。#4656 的 PR 分支 `k3_ac_reuse_attention` 保持 `7dea4cbaf`（它和现在的 main 在 `model.py`、`sharding.py` 上冲突）；rebase 后的版本在 `attnres_review1` = `507a2f7d5`，#4963 的 PR 分支 `pp_review_optimize` 已经带着它。
