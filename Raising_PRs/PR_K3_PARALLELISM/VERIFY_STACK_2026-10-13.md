# 栈分支 `k3_mxqat_stack_20261012` 本机全面验证（2026-10-13 至 10-14）

用户："本机能验证的都尽量全部验证"，目的是确认能不能交给 Elfie 跑 2.8T。分支头 `b49021d5e`（每个 PR 一个提交，LoRA 已压成一个），8 × 5060，venv_1006，main 的三个固定依赖在 `scratchpad/main_pins`，Elfie 的 torchao 源码在 `scratchpad/elfie_ao`。kit `kit_verify_stack_2026-10-13/`，日志在 scratchpad 的 `verify_stack_1013/`。

## 最重要的发现：Elfie 的验证入口在栈上跑不起来（已在 worktree 里修，未提交）

她 2.8T 用的 `scripts/validation/kimi_k3/run.sh` 调 `validate_distmuon.py`（full128、full256）和 `hybridep_stage_memory.py`（memory32）。这些脚本按她旧基底的引擎接口写，rebase 时的适配提交 `4b8559c51` 只改了 dispatcher 部分，脚本在栈上从没执行过。本机一跑，按出现顺序：

1. `TrainingConfig(max_norm=...)`、`TrainingEngine.Config(optimizer=..., lr_scheduler=...)`：main 挪进了 `Optim.Config`。
2. `engine.initialize(compile_config=...)`：main 没有这个参数。
3. `model.update_from_config(config=...)`：main 的 #4908 删掉了，模型配置要在构建时拿到并行配置（K3 读 `enable_sequence_parallel` 和 `expert_parallel_degree`），改成先建 `ParallelismConfig` 再 `build_model_config`。
4. preflight 读 `config.optimizer`：改成 `config.optim.optimizer`。dp1 时 main 不建 `dp` mesh，种子用的 dp rank 在不开 DP 时取 0。
5. 训练循环用的 `engine.prepare_step`、`forward_backward_microbatch`、`optimizer_step`、`engine.optimizers`：main 是 `forward_backward(microbatch_groups=..., global_loss_token_counts=..., global_routing_token_counts=...)`、`optim_step()`、`engine.optim.optimizers`。micro-batch 改用 main 的 `TokenizedTrainingMicrobatch`（`loss_token_counts`、`routing_token_counts`），计数用 main 的 `get_mtp_token_counts` 算，按 `trainer.train_step` 的做法在 dp 上求和；PP 时一组是一个 PP step，不开 PP 时每个 micro-batch 一组。
6. `--balanced-routing` 构造的 router 带 `_debug_force_load_balance`：main 改成专门的 `RoundRobinTokenChoiceTopKRouter`（固定的均衡路由编号，打分仍由 gate 算），换成它。
7. `finite_diagnostics` 的 forward hook 读每个模块输出的数值：main 的 AC 基于 torch_remat，重算时"存下来不重算"区域的输出只是占位符，一读就报错（`layers.16::attention.wq_b.linear.out is a placeholder ...`）。按 torch_remat 文档的做法，hook 在 `remat.is_recomputing()` 时直接返回。
8. `pipeline_memory.RecomputeMatmulsSelectiveAC` 只覆盖了旧的按算子 `get_save_ops`，main 的区域 AC 根本不调它：`--activation-checkpoint recompute-matmuls` 在栈上不报错，但静默等于普通 `SelectiveAC`，dense matmul 不重算。改成 `RegionAC`：`save_regions=["*"]`，`recompute_regions=["*routed_experts.w13.*", "*.linear"]`（main 的区域名：dense 是 `<模块>.linear`，expert 是 `<模块>.grouped_mm`，TP 通信 `tp_gather` / `tp_reduce`，所以 `*.linear` 只命中 dense linear，通信区域照存）。她的测试改成用 main 的 `Module.remat_should_recompute` 检查几个真实区域名。和她原来的按算子策略不完全等价（原来 attention 里的 bmm 也重算），需要她确认。

9. 她的 fused loss（`torchtitan/components/loss.py` 的 `LinearCrossEntropyLoss`，她的提交带来的）：main 的 loss 约定是 `__call__(pred, labels, global_loss_token_counts=None)`，引擎按关键字传；她的签名还是旧名 `global_valid_tokens`，新名字落进 `**kwargs`，loss 不除 token 数。开 `--fused-loss`（full256 模式）时，loss 和梯度放大到每个 rank 的 token 数倍：full256 缩小版第 1 步 loss 264192 = 32768 × 8.06，grad norm 11648，被 `max_norm=1.0` 裁剪压回去，训练不报错但梯度尺度错了。改成 `global_loss_token_counts`。她的测试原来按位置传 token 数，两种名字都过；改成按引擎的关键字传后，旧代码失败（1 failed），新代码 3 passed。CI recipe 和 18 格都不用这个 loss，不受影响。
10. `hybridep_stage_memory.py`（memory32）有同样的差异：`config.optimizer`、`initialize(compile_config=...)`、`num_valid_tokens`，按第 1、2、5 条同样改。它写死 HybridEP、EP32、完整 Kimi-K3，本机原样跑不了；debug 模型、standard dispatcher、EP8 的冒烟副本见下。

改动只在 `scratchpad/wt_mxqat_fix` 的工作区，没有提交。完整 diff 存在 kit 的 `elfie_scripts_port.patch`（随改动更新）。10-14 对三个移植过的脚本跑了 ufmt（只有 import 顺序和换行，她原来的提交是 ufmt 干净的，格式问题来自移植），她的脚本单测 15 passed；排在 GPU 队列最后再按最终状态把入口跑一遍（`g2h_elfie_final.sh`）。这一遍的 dp1 报 `FileNotFoundError: /tmp/torchinductor_root/triton/0/.../l2norm_bwd_kernel.source`：磁盘只剩 34G，低于看门狗的 40G 阈值，它每分钟清一次不受保护的 JIT 缓存，默认的 `/tmp/torchinductor_root` 不在保护名单里（scratchpad 的 `/claude-` 路径在）。停掉这一遍，改成每次运行把缓存放在自己的输出目录下（`g2i_elfie_final.sh`，输出 `elfie_entry_final2`）重跑。结果（worktree diff 哈希 `a2939e344e18`，即要提交的状态）：preflight rc 0（98 个 QAT master，DistMuon + AdamW）；dp1 PASS，第 1 / 10 步 loss 8.410406 / 7.885763、grad norm 5.59375，和移植后第一次（run5）逐位相同；full128 缩小版 8/8 PASS，第 1 步 8 个 rank 的 loss 和 grad norm 和 run6 逐位相同（第 10 步差 1e-3 量级：这次每个运行都从空缓存编译，run6 用的是默认缓存）；full256 缩小版 PP2 × EP4 8/8 PASS，第 1 步每个 rank loss 约 2.02、grad norm 0.0889；PP8 8/8 PASS，第 1 步 loss 8.125、grad norm 0.1387。

## 结果（到目前为止）

- Elfie 的入口，debug 模型 + 她的发布格式 fixture（451 对 packed），standard dispatcher：
  - fixture 生成、`validate_kimi_k3_mxfp4_checkpoint.py`：通过。
  - preflight：98 个 QAT master，优化器 DistMuon + AdamW。
  - 单卡 10 步：PASS。
  - full128 缩小版（PP1、EP4、dense FSDP 8 路、expert FSDP 2 路、selective AC、diagnose 2 步、profiler）：8/8 PASS。
  - full128 + `--diagnostic-state`：8/8 PASS。
  - full256 缩小版 PP2 × EP4（seq 2048、mb 2 × 8、fused loss、balanced routing、recompute-matmuls、reshard-after-backward）：8/8 PASS（这次 recompute-matmuls 还是失效的旧策略）。
  - full256 缩小版 PP8：8/8 PASS。
  - 移植后的 recompute-matmuls 对 selective（full256 缩小版 PP2 × EP4，6 步）：峰值 1.09 GiB 对 1.44 GiB，低 24%，策略生效（移植前它和 selective 完全一样）。
  - fused loss 修正后（10 步，全部 8/8 PASS）：PP2 × EP4 第 1 步每个 rank 的 loss 约 2.02（= 8.06 / 4 个 dp rank），grad norm 0.0889；同形状不开 fused loss（main 的 `CrossEntropyLoss`）每个 rank 2.025 到 2.029，grad norm 第 1 / 5 / 10 步 0.0889 / 0.0869 / 0.0845，和 fused loss 逐位相同（fused loss 的 loss 是 bf16，所以打印值只差 bf16 的取整）；PP8 第 1 步 loss 8.125、grad norm 0.139。修正前同一形状是 264192 和 11648。
  - memory32（`hybridep_stage_memory.py`）冒烟副本（debug 模型、standard dispatcher、EP8、她的 fixture、11 层前缀）：移植后 8 次前向、10 次反向都跑完，最后的梯度检查读 `FSDPParam.unsharded_accumulated_grad_data`，本机 torch nightly 里已经没有（只有 `unsharded_accumulated_grad`，返回 unsharded 参数的 `.grad`）。换成后者（DTensor 取本地分片）后 PASS，梯度是 fp32，她检查的"fp32 累加梯度"成立。这一处是 FSDP2 内部接口随 torch 版本变的，同样改进她的脚本。
  - 她的汇总脚本 `summarize_run.py`（`--warmup-steps 2`）读这三次的输出都通过它自己的断言：8 个 rank 全 PASS、10 步、各解析 1 份 Kineto trace；第 10 步最后一个 stage 的 loss 求和 full128 7.941、full256 PP2 × EP4 8.0（4 个末级 rank 各约 2.0）、PP8 8.0625，探针参数每步都有变化（最少 3033 个元素）。
- CI 格子（`b49021d5e` 对 main，10-12 的协议：同一份预热缓存的副本，20 步）：`kimi_k3_mm` 20/20、`kimi_k3_mm_muon` 20/20 逐位相同；两个 CP 格子前 4 步相同、第 5 步起不同（dynamic CP 第一次切图），第 20 步和 10-12 那次一样（all-gather 5.53112、Ulysses 5.54415）；PP × VPP4 第 2 步起不同（DEP + #4656 编译区），第 20 步和 10-12 不同但 main 自己也变了（3.06905 到 3.06835，缓存重新冷编译），第 1 步相同；Elfie 的 MX QAT recipe 20 步，第 1 / 20 步和 10-12 相同；MXFP4 checkpoint 集成测试 1 passed。
- 开类型检查的组合（CI 的 `kimi_k3_mm` / `kimi_k3_mm_muon` 再套变换，10 步）：
  - MX QAT（mm、mm_muon）：`SpmdTypeError: _MXFakeQuantize.apply: autograd Function is not registered for SPMD type checking`。torchao 的 fake quant Function 没有在 spmd_types 里注册，MX QAT 开不了类型检查。她的 2.8T 入口不开类型检查，不受影响；要进开类型检查的 CI 格子，需要她的 MX QAT 代码注册它。验证：在 recipe 里把她 torchao 的三个 Function（`_MXFakeQuantize`、`_MXQATNativeGroupedMM`、`_MXQuantizedForwardFakeQuantizedBackward`）用 `spmd.register_local_autograd_function` 注册（和 main 对 `_MXFP8LinearFunction`、`_NVFP4LinearFunction` 的做法一样，kit `stack_probe_tc_reg.py`），mm（TP2 × EP2 × FSDP2）和 mm_muon（EP2 × FSDP2，DistMuon）开类型检查都 10 步 rc 0。修法就是在 titan 的 `quantization/mx_qat` 里加这几行注册。
  - LoRA、QLoRA（mm）：`Partial type on axis mesh_ep cannot propagate through non-linear ops`，出在视觉塔行并行 linear 的 adapter：`lora_a` 按输入维切，`u = lora_a(x)` 是 TP 上的部分和（这个 recipe 里 TP 轴叫 mesh_ep），再进 `lora_b._linear` 时，Linear 的 autograd Function 类型规则不接受 Partial 输入。数学上对（18 格不开类型检查时导出逐位精确）。dense 行并行底座的 adapter 切法和 main 原来一样。对照（llama3 debug，TP2 + SP，CI 的类型检查开关，3 步）：不带 LoRA 时 main `b663c1673` 和 PR `fb4079831` 都 rc 0；带 `LoRATransform(rank=8, alpha=16.0)` 时两边都 rc 1，报同一个错，位置相同（`lora.py` 的 `_linear` 里 `self.lora_b._linear(...)` 收到 `lora_a` 的 Partial 输出）。所以"LoRA + TP + 类型检查"是 main 的 LoRA 原有的缺口，不是 #4576 带来的；修法是行并行底座先归约 `lora_a` 的输出再进 `lora_b`，或给这一步一个保留 Partial 的类型规则，属于 main 的后续问题。
- MX QAT + DistMuon：seed 生成在单卡上报 `DistMuon requires DTensor parameters`（DistMuon 本身的限制，参数要被 FSDP 切成 DTensor）。改用 MX QAT 组的 seed（只存模型），格子去掉单卡，存档续跑改成带 FSDP 的 5 种布局（fsdp2、ep2 × fsdp2、fsdp2 × pp2、fsdp2 × pp2 × vpp2、fsdp2 × cp2），排在 GPU 单测之后。
  - 结果（10 种布局，各 10 步，K3 debug 模型 mm 数据，MX QAT 加 DistMuon）：10/10 rc 0，每个 rank 都加载了 seed。第 1 步按数据流一致：fsdp2 = ep2 × fsdp2 = 8.11612，pp2 = pp4 = pp8 = 8.12757，fsdp2 × pp2 × cp2 = ep2 × fsdp2 × pp2 × cp2 = 8.11572，cp2 8.12737，cp4 8.13820，ep8 × fsdp8 8.13140；第 10 步 6.41 到 6.75，全部有限。
- GPU 单测：第一次把 `tests/unit_tests/gpu` 整个目录放在一个 pytest 进程里跑，所有多进程测试（DTensorTestBase、spawn 的 FSDP 测试，栈和 main 自己的都算）一律失败，单跑都通过。原因在 main 的 `test_ema.py`：`TestEMACpuOffload.setUpClass` 用 `os.environ.setdefault` 写了 `LOCAL_RANK=0`、`RANK=0`、`WORLD_SIZE=1`、`MASTER_PORT=29602`，不清理；之后 spawn 的子进程继承 `LOCAL_RANK=0`，所有 rank 落到同一张卡，NCCL 报 `Multiple Ranks are using the same GPU/Partition`。带上这几个环境变量单跑 LoRA 的裁剪测试即复现。CI 不受影响，因为 CI 分两条线跑（`-m "not multi_gpu"` 单卡，`-m multi_gpu` 8 卡），`test_ema` 只在单卡线。这是 main 测试隔离的问题，不是栈的。改成按 CI 的两条线在栈和 main 上各跑一遍（单卡线两棵树并行，各自缓存和端口），结果见下。
- 顺带发现 LoRA PR（#4576）的两个 GPU 测试漏了 `multi_gpu` 标记：`test_clip_grad_norm.py`（2 卡）、`gpu/models/common/test_lora.py`（4 卡）都是 DTensorTestBase，没有标记就落在单卡线，`with_comms` 发现卡数不够直接 skip，8 卡线又不选它们，CI 里永远不会真跑。main 的约定是模块级 `pytestmark = pytest.mark.multi_gpu`（#4127 起，`test_mxfp8_fsdp.py` 的注释写明了这个原因）。已在 `wt_lora` 和栈 worktree 加上（未提交），`--collect-only` 确认：`-m multi_gpu` 选中 2 个，`-m "not multi_gpu"` 0 个。推 PR 分支要用户点头。main 自己也有 5 个文件同样漏标（`test_async_linear`、`test_dist_moe`、`test_distributed_linear`、`test_tensor_parallel_feed_forward`、`test_offset_rmsnorm`），不归这个 PR。
- GPU 单测单卡线（`-m "not multi_gpu"`，`CUDA_VISIBLE_DEVICES=0`/`1`，两棵树并行，各自缓存）：两边都失败的 36 个是本机环境（torchao 的 mxfp8 算子、sm120 上的 nvfp4 cutlass、没装 helion、qwen3_5 deltanet 缺模块、qwen3_5 RoPE 编译对不上）；只在栈上失败 2 个，都在栈新加的 `test_moe_shared_experts_stream.py`：`make_routed_experts_config() missing 1 required keyword-only argument: 'enable_ep'`，main 的 #5280（10-11）把 `enable_ep` 改成必填。只在 main 上失败 0 个。
- GPU 单测 8 卡线（`-m multi_gpu`，8 卡，新缓存）：栈 41 passed、8 failed、11 skipped、1 xfailed（16 分 27 秒）；main 35 passed、8 failed、2 skipped、1 xfailed（9 分 29 秒）。8 个失败两边相同（sm120 上的 mxfp8 / nvfp4 FSDP 测试）；只在栈上失败 0 个。栈多出的 9 个 skip 是 MoonEP 的 GPU 测试（没有库）。加了标记的两个 LoRA 测试都被选中并通过：合并测试 13.1 秒，裁剪测试不到 9.3 秒（不在最慢 40 个里）。栈上 #4380 的两个 vision CP 测试冷缓存共 388 秒（257 + 131），CI 这条线是 30 分钟的任务，#4380 要留意。
- MoonEP（栈提交 `92afeeeee`）在栈上跑不起来：本机用假包（`scratchpad/fake_moonep_async`）加多播 shim（`scratchpad/moonep_hook/mc_shim`，`FAKE_MULTICAST=1`）跑 K3 debug 模型 FSDP 4 × EP 4（模型配置按 #4908 的要求带 EP 4 构建，kit `kit_moonep/moonep_stack_probe.py`）：
  - 同形状的标准 EP 3 步正常（loss 8.17923 / 7.98715 / 7.82595）。
  - MoonEP 第一次前向：`TypeError: MoonEPRoutedExperts.forward() got an unexpected keyword argument 'padding_mask_T'`。main 的 #5046（10-10）让 `MoE` 总是把 `padding_mask_T` 传给 routed experts，`RoutedExperts.forward` 加了这个关键字参数，MoonEP 的覆盖没有。pyrefly 的 `bad-override` 也指出这一处。
  - 补上这个参数（`del padding_mask_T`，和基类一样不用；patch 存 `kit_moonep/moonep_padding_mask_kwarg.patch`，worktree 已还原）后，下一处：`ops.py:85` 的 `activation(gate, up, offsets=...)` 报 `BinaryActivationFn.__call__() takes 2 positional arguments`。main 的 #5086（10-06）把门控激活改成接收 `w13` 输出的 `[..., 2, F]`（gate、up 在编译区内 unbind）。MoonEP 的 gate、up 是两次 GEMM 的结果，要么先 stack（多一次拷贝），要么让两次 GEMM 直接写进 `[R, 2, F]`；这是 #4751 rebase 时要决定的。
  - 测试也要跟：`test_moonep.py`、`test_moe_shared_experts_stream.py` 的 `make_routed_experts_config` 都缺 `enable_ep`（#5280）。
  - PR 分支本身（`16ff9da7f`，基于 main `3f087cf15`，10-06）没有这三处问题，三个 main 改动都在它的基底之后；问题在 10-12 把它搬到 `b663c1673` 时没有运行过（真库要交换机多播，本机测试默认跳过）。
  - Elfie 的 2.8T 不用 MoonEP（full128 standard，full256 / memory32 hybridep，都是 main 自己的 dispatcher 和 `RoutedExperts`），不受影响。`shared_experts_stream` 默认关，CI 格子 mm / mm_muon 20 步和 main 逐位相同，默认路径不受影响。
- 静态检查（栈对 main 改过的 90 个 .py，含未提交的修改）：
  - CI 固定版本的 flake8 hook（`pre-commit run flake8`，bugbear 22.4.25）：Passed。venv 里新版 bugbear 报的 B905 / B907 等 main 自己也有 90 + 56 处，CI 不查。
  - ufmt（2.3.0 + black 22.12.0 + usort 1.0.5，和 hook 一致）：已提交的栈上 2 个文件要格式化：`torchtitan/config/transform/mx_qat.py` 的 import 顺序（适配提交 `4b8559c51` 把 `torchtitan.protocols.module` 原地改成 `torchtitan.models.base.module`），`tests/unit_tests/cpu/config/test_transforms.py` 多一个空行（`92afeeeee` 搬 MoonEP 测试时）。未提交的 3 个 Elfie 脚本也要格式化（移植时的改动）。main 自己有 3 个文件不合格。
  - codespell、license 头、行尾空白、文件末换行：干净。
  - pyrefly（0.45.1，只读，不带 `--remove-unused-ignores`）：栈 48 个错误、main 2 个（两边共有的 2 个：`video.py`、`dist_moe/runtime.py`）。栈多出的 46 个：43 个在 Elfie 的 `scripts/validation/kimi_k3`（同目录模块按裸名导入，pyrefly 找不到；以及松散的类型标注；脚本运行都没问题）和 `scripts/checkpoint_conversion/create_mxfp4_reference.py`（本机没装 `compressed_tensors`）；库代码 2 个：MoonEP 的 `forward` 覆盖（上一条，运行时真错）、`KimiK3StateDictAdapter.get_hf_storage_reader` 少了 main 新加的 `thread_count` 关键字（main 的 checkpointer 按位置只传 `path, from_quantized`，运行时不出错，但覆盖和基类签名不一致，打包 MXFP4 读取器的线程数也不随 `thread_count` 走）。pyrefly 的 project-excludes 只排除 `torchtitan/experiments`、`torchtitan/rl` 和 `**/tests/**`，`scripts/` 在检查范围内；Elfie 的这些脚本如果进她的 PR，CI 的 pyrefly 会红。
- `validate_ep_placement.py` 用假 `scontrol`（`fake_slurm/bin/scontrol`，按 `nvlblk<i // 块大小>` 生成 ActiveFeatures）跑：解析和判断逻辑都对。full256（64 个 4 卡节点，EP 32 = 8 节点，PP 8）在 18 托盘一个 NVL72 的连续分配下第 2 组跨域，报 `EP group 2 crosses NVLink domains`；16 节点一块时 PASS，8 组两两落在 4 个域；memory32（8 节点）PASS。脚本假定 ranks = EP × PP，full128（EP 64、PP 1、128 卡，DP 2）原样传参会报 `Expected 64 ranks, got 128`，要把 `--pp` 当成"EP 组数"传 2。
- 存档续跑（MX QAT + DistMuon，5 种带 FSDP 的布局；第一遍从 seed 连续训 10 步、第 5 步存完整状态，第二遍拷第 5 步存档和第一遍的缓存续训到 10 步）：10 次都 rc 0，续跑都从第 6 步开始。fsdp2 第 6 到 10 步 5/5 逐位相同；ep2 × fsdp2、fsdp2 × cp2、fsdp2 × pp2、fsdp2 × pp2 × vpp2 第 6 步相同、第 7 步起差约 1e-4（例：ep2 × fsdp2 第 7 步 6.88021 对 6.87937）。第 6 步相同说明参数和数据都恢复了；第 7 步起的差可能是第一遍跑在冷缓存、第二遍跑在它的副本上（已知冷缓存和副本本身就会算出不同结果），也可能是优化器状态没完全恢复。对照和定位（全部跑完）：
  - 冷缓存对缓存副本（`run_resume_control.sh`：第一遍的缓存拷一份，从 seed 再连续跑一遍，记 r1b）：fsdp2 10/10 相同；其余四种只有第 1 步相同，第 2 步起就分开。所以第一遍（冷缓存）和续跑（副本）本来就不可比，原来那张比较作废。
  - 全部在副本上（`run_resume_warm.sh`：r1c 在另一份副本上从 seed 再跑，r2b 从 r1b 自己第 5 步的存档在副本上续跑）：五种布局 r1b 对 r1c 都 10/10 逐位相同（副本上训练是确定的）；fsdp2、ep2 × fsdp2、fsdp2 × cp2、fsdp2 × pp2 的续跑 5/5 逐位相同，续跑精确。
  - fsdp2 × pp2 × vpp2 的续跑只有第 6 步相同（loss 7.07272、grad norm 4.5625），第 7 步起不同（6.88059 对 6.88040，grad norm 5.5312 对 5.5625）。r1b 对 r1c 10/10 相同，所以不是不确定性：第 5 步的参数和第 6 步的数据都恢复了，第 6 步的更新不同，是每个 rank 两个虚拟 stage 时有优化器相关的状态没有完整恢复。栈没有改 DistMuon 和 checkpointer（只有 LoRA PR 在 `optimizer.py`、`lr_scheduler.py` 各 8 行处理空参数组，这里不触发），正在 main `b663c1673` 上用同一布局定位：DistMuon（arm `base_muon`）和 AdamW（arm `base`）各跑冷缓存一遍、副本一遍、从副本第 5 步续跑一遍（`run_vpp_resume_main.sh`，输出 `resume_vpp_main2`）。main 上的结果（`b663c1673` 干净树，同一 seed，fsdp2 × pp2 × vpp2）：AdamW 的续跑 5/5 逐位相同；DistMuon 的续跑只有第 6 步相同，第 7 步起不同（6.87135 对 6.87156），和栈上一样。所以这是 main 的 DistMuon 在一个 rank 有两个虚拟 stage 时续跑不精确，不是栈带来的（MX QAT 和 #4576 都不涉及）。存档里优化器状态的 key 数和 pp2 一样（939 个、465 个参数），不是存档少了 key。main 上 DistMuon 副本对副本（r1c）10/10 逐位相同，所以这个偏差是确定的，不是噪声。Elfie 的入口不设 `pipeline_parallel_schedule`，用默认的 1F1B（每个 rank 一个 stage），验证也不续跑，不受影响；正式训练如果用 Interleaved1F1B 之类每个 rank 多个 stage 的调度，DistMuon 的续跑会有这个偏差，要告诉她。
- 2.8T 的 key 映射（`k3_full_key_coverage.py`，只用 CPU）：完整 Kimi-K3 在 meta 上 2,771 个 state key、2.780T 元素；官方索引 497,220 个 key 合成 249,924 个逻辑权重（247,296 个 packed），`to_hf` 正好产出这 249,924 个，缺 0 多 0；`from_hf` 回到全部 2,771 个 key；MX QAT 选中 184 个 Titan 参数（92 个 MoE 层各一个 `w13`、一个 `w2`），packed 权重全部被 QAT 覆盖，被选中的也全是 packed。
- CPU 单测（全部，GPU 屏蔽）：栈 1671 + 80 passed，main 1562 + 80 passed；只在栈上失败的 0 个；两边都失败的 3 个是 main 在本机就挂的（flux 两个、embedding 一个）；10 个只在 main 上失败（parallelism_context、cp_attention all-gather）。收集错误的 6 个文件是源码 torchao 版本号 `unknown`，第二遍用 nightly torchao 跑 80 passed。
- Elfie 的脚本单测（`scripts/validation/kimi_k3`）：移植后 15 passed。

## 本机跑不了的

- `validate_ep_placement.py`：要 Slurm 的 `scontrol` 读节点分配。
- `hybridep_stage_memory.py`（memory32）、`--dispatcher hybridep` / `deepep`：要 DeepEP hybrid-ep 和 NVSHMEM；MoonEP 要交换机 multicast。
- RL 的 `rl_grpo_kimi_k3_debug_batch_invariant`：本机没有 vLLM。
- 多节点、93 层真实模型、真实发布权重的加载（只查了索引和 key 映射）。

## 10-14 收尾（用户："清理磁盘，MoonEP从栈上面拿掉，LoRA PR完全ready了吗？"）

- 清理：本次和几次已结束实验（MoonEP 性能、CP 数值、DEP、#4656、LoRA 18 格等）的编译缓存、checkpoint 和运行目录按字面路径删除，删除前把日志、进度、JSON、脚本、patch 打包成 scratchpad 的 `run_logs_archive_20261014.tgz`（188M，3841 个文件）。磁盘可用从 22G 到 89G；worktree 和 kit 没动。
- MoonEP 拿掉：`git rebase --onto 525fd3e47 92afeeeee`，栈 = `0d00a4f09`，先推备份 `backup/k3_mxqat_stack_20261012_pre_moonep_drop` = `b49021d5e`，再 force-with-lease 推。新旧 head 之差正好是 MoonEP 提交（patch-id 相同），树里没有 MoonEP、`shared_experts_stream`，`moe.py` 等和 main 相同；相关 CPU 测试 98 passed；mm、mm_muon 3 步，第 1、3 步和旧栈逐位相同。九个未提交的修复在 rebase 前存成 patch，rebase 后原样打回。
- LoRA PR #4576：头 `fb4079831`（= `lora_review2`），草稿，标题 `[DO NOT review][Kimi K3][LoRA] ...`；线上 body 和 v2 文件相同，只少 `## Relation to other PRs` 一节；和 main `dbfba1da8`（比基底多 28 个提交）`git merge-tree` 无冲突；上游 CI 等维护者批准（`ciflow/rl` 机器人评论）。还差两个 GPU 测试的 `multi_gpu` 标记（`wt_lora` 未提交）。
- 修复折进栈（用户"2件都确认"，auto mode 拦下 cherry-pick 后用户放行）：九个文件和 `mx_qat.py` 的 import 顺序折进适配提交（`70e7263cd`），其后六个提交 cherry-pick 无冲突，`multi_gpu` 标记折进 LoRA 提交（说明改为 PR-4576 在 `676e443e0` 的 5 个提交）。栈 = `7f8b1bf4d`，先推备份 `backup/k3_mxqat_stack_20261012_pre_fix_fold` = `0d00a4f09` 再 force-with-lease。核对：新树 = 旧 head + 验证过的九个文件 diff + import 顺序；LoRA PR 改过的文件和 `676e443e0` 逐字节相同；CPU 114 passed；mm、mm_muon 3 步第 1 步相同，mm 第 3 步在新冷缓存上是 6.60117，换到上一次运行缓存的副本上是 6.59915，和上一次 3 步逐位相同（冷缓存造成，不是代码）。

## 10-14 在 `7f8b1bf4d` 上全部重跑（用户："全部重跑"）

脚本 `r7f8b/driver_7f8b.sh`（scratchpad `verify_stack_1013/r7f8b`），太平洋时间 01:58 到 05:53。树 `7f8b1bf4d`，工作区干净。

- 2.8T key 映射：通过，和 10-13 一样（缺 0 多 0，`from_hf` 回到 2,771 个 key，QAT 184 个和 packed 一一对应）。官方 config 在 `/tmp/k3cfg/config.json`，和 `phase13_k3like_48b_posttrain/official_k3/reference` 的索引放进 `r7f8b/k3_view`。
- CPU 全部单测：1668 passed，3 failed（flux 两个、embedding 一个，main 也挂），6 个文件收集错误用 nightly torchao 补跑 80 passed；只在栈上失败 0 个，失败名单和 10-13 的栈完全相同。比 10-13 少 3 个通过是 MoonEP 的 CPU 测试随提交去掉了。
- 18 格（`cells_mm18.txt`，K3 debug、cc12m-test、seed 42、deterministic、10 步）：
  - base 18 / 18；第 1 步和 10-12 逐位相同（dp1 组 8.12572、dp2 组 8.11338、ep8 × fsdp8 8.12882、cp4 8.13709）。
  - MX QAT 18 / 18；第 1 步和 10-13 逐位相同（8.12757、8.11612、8.13140、8.13820）。
  - MX QAT + DistMuon 10 / 10（FSDP 布局），第 1 步和上一轮相同。
  - LoRA 18 / 18，`gate_mm18.py`：每个 rank 都有导出检查，worst 0，模块数全对。
  - QLoRA（MX QAT + LoRA）18 / 18，同上；ep8 × fsdp8 用按本地专家分批的期望值，直接 worst 0。
- 冻结 stage（只给 MLA 投影加 LoRA）：pp4 × vpp2、pp8 都失败，stage 1 报 `RuntimeError: stage 1: backward produced no gradient for any input`（`kimi_k3/pipeline_parallel/stage.py`），其余 rank NCCL 超时。定位：同一 seed、同样两格，在去掉 #4963 的临时树（`7f8b1bf4d` 上反向打 `1cfa0b99e` 的 diff，`scratchpad/wt_stack_no4963`）上都通过（pp4 × vpp2 导出 25 个模块 worst 0，3 个冻结 part 按设计跳过）。原因：main 和 #4656 把收到的注意力残差块拼成一个总是要梯度的叶子，所以前面全冻结时也有梯度；#4963 改成每块单独、输入梯度单独收集后，冻结的 stage 0 后面那个 stage 只有不需要梯度的 hidden、没有消费存下的块，所有输入梯度为 None，检查报错。#4963 已经对"自己提交的块上游不可训练"有同样的豁免，这里缺同一个豁免：所有输入的 receive metadata 都不需要梯度时，按冻结上游处理，不报错。10-12 那两个冻结格子跑在 LoRA PR 的树上（main + LoRA），没有在栈上跑过；10-14 早先的表把它写成"`0fa53c2e3` 通过"是错的。
- 存档续跑（`run_resume_all.sh`，五种布局，全部在副本缓存上）：r1b 对 r1c 五种都 10 / 10；续跑 fsdp2、fsdp2 × cp2、fsdp2 × pp2、fsdp2 × pp2 × vpp2 5 / 5 逐位相同，ep2 × fsdp2 第 6 步相同、第 7 步起不同（6.88123 对 6.88147）。上一轮不精确的是 vpp2、这轮是 ep2 × fsdp2，所以和 VPP 无关，10-14 早先"DistMuon 在 VPP 下续跑不精确、是 main 的问题"的说法作废。正在做重复性检查：同一存档再续跑两次（`run_resume_repeat.sh`）。
- CI 格子（`b663c1673` 对 `7f8b1bf4d`，同一份预热缓存的副本，20 步）：mm、mm_muon 20 / 20 逐位相同；两个 CP 格子前 4 步相同、第 5 步起不同（动态 CP），栈第 20 步和前两轮相同（all-gather 5.53112、Ulysses 5.54415）；pp × vpp4 第 1 步相同、第 2 步起不同（DEP、#4656 编译区）；Elfie 的 MX QAT recipe 20 步；MXFP4 checkpoint 集成测试 1 passed。
- Elfie 的入口：preflight、dp1（8.410406 / 5.59375）、full128 8 / 8、full256 PP2 × EP4 8 / 8（loss 约 2.02、grad norm 0.0889）、PP8 8 / 8（8.125 / 0.1387），和上一轮相同；memory32 冒烟 8 个 rank 都 PASS，梯度 fp32。
- GPU 单测（CI 的两条线）：单卡线 179 passed、36 failed，8 卡线 41 passed、8 failed、2 skipped；两条线的失败名单都和 main 完全相同，只在栈上失败 0 个。
