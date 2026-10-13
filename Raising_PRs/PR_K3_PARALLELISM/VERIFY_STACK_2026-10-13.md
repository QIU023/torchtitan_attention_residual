# 栈分支 `k3_mxqat_stack_20261012` 本机全面验证（2026-10-13，进行中）

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

改动只在 `scratchpad/wt_mxqat_fix` 的工作区，没有提交。完整 diff 存在 kit 的 `elfie_scripts_port.patch`（随改动更新）。

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
- CI 格子（`b49021d5e` 对 main，10-12 的协议：同一份预热缓存的副本，20 步）：`kimi_k3_mm` 20/20、`kimi_k3_mm_muon` 20/20 逐位相同；两个 CP 格子前 4 步相同、第 5 步起不同（dynamic CP 第一次切图），第 20 步和 10-12 那次一样（all-gather 5.53112、Ulysses 5.54415）；PP × VPP4 第 2 步起不同（DEP + #4656 编译区），第 20 步和 10-12 不同但 main 自己也变了（3.06905 到 3.06835，缓存重新冷编译），第 1 步相同；Elfie 的 MX QAT recipe 20 步，第 1 / 20 步和 10-12 相同；MXFP4 checkpoint 集成测试 1 passed。
- 开类型检查的组合（CI 的 `kimi_k3_mm` / `kimi_k3_mm_muon` 再套变换，10 步）：
  - MX QAT（mm、mm_muon）：`SpmdTypeError: _MXFakeQuantize.apply: autograd Function is not registered for SPMD type checking`。torchao 的 fake quant Function 没有在 spmd_types 里注册，MX QAT 开不了类型检查。她的 2.8T 入口不开类型检查，不受影响；要进开类型检查的 CI 格子，需要她的 MX QAT 代码注册它。
  - LoRA、QLoRA（mm）：`Partial type on axis mesh_ep cannot propagate through non-linear ops`，出在视觉塔行并行 linear 的 adapter：`lora_a` 按输入维切，`u = lora_a(x)` 是 TP 上的部分和（这个 recipe 里 TP 轴叫 mesh_ep），再进 `lora_b._linear` 时，Linear 的 autograd Function 类型规则不接受 Partial 输入。数学上对（18 格不开类型检查时导出逐位精确）。dense 行并行底座的 adapter 切法和 main 原来一样，判断是 main 的 LoRA 在"类型检查 + TP"下本来的问题；用 llama3 TP2 + 类型检查在 main 和 PR 上各跑一次确认（排在 GPU 单测之后）。
- MX QAT + DistMuon：seed 生成在单卡上报 `DistMuon requires DTensor parameters`（DistMuon 本身的限制，参数要被 FSDP 切成 DTensor）。改用 MX QAT 组的 seed（只存模型），格子去掉单卡，存档续跑改成带 FSDP 的 5 种布局（fsdp2、ep2 × fsdp2、fsdp2 × pp2、fsdp2 × pp2 × vpp2、fsdp2 × cp2），排在 GPU 单测之后。
- 2.8T 的 key 映射（`k3_full_key_coverage.py`，只用 CPU）：完整 Kimi-K3 在 meta 上 2,771 个 state key、2.780T 元素；官方索引 497,220 个 key 合成 249,924 个逻辑权重（247,296 个 packed），`to_hf` 正好产出这 249,924 个，缺 0 多 0；`from_hf` 回到全部 2,771 个 key；MX QAT 选中 184 个 Titan 参数（92 个 MoE 层各一个 `w13`、一个 `w2`），packed 权重全部被 QAT 覆盖，被选中的也全是 packed。
- CPU 单测（全部，GPU 屏蔽）：栈 1671 + 80 passed，main 1562 + 80 passed；只在栈上失败的 0 个；两边都失败的 3 个是 main 在本机就挂的（flux 两个、embedding 一个）；10 个只在 main 上失败（parallelism_context、cp_attention all-gather）。收集错误的 6 个文件是源码 torchao 版本号 `unknown`，第二遍用 nightly torchao 跑 80 passed。
- Elfie 的脚本单测（`scripts/validation/kimi_k3`）：移植后 15 passed。

## 本机跑不了的

- `validate_ep_placement.py`：要 Slurm 的 `scontrol` 读节点分配。
- `hybridep_stage_memory.py`（memory32）、`--dispatcher hybridep` / `deepep`：要 DeepEP hybrid-ep 和 NVSHMEM；MoonEP 要交换机 multicast。
- RL 的 `rl_grpo_kimi_k3_debug_batch_invariant`：本机没有 vLLM。
- 多节点、93 层真实模型、真实发布权重的加载（只查了索引和 key 映射）。
