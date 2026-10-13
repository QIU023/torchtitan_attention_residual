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
  - fused loss 修正后的 full256 两个形状、不开 fused loss 的同形状参照、memory32 冒烟：运行中。
- 2.8T 的 key 映射（`k3_full_key_coverage.py`，只用 CPU）：完整 Kimi-K3 在 meta 上 2,771 个 state key、2.780T 元素；官方索引 497,220 个 key 合成 249,924 个逻辑权重（247,296 个 packed），`to_hf` 正好产出这 249,924 个，缺 0 多 0；`from_hf` 回到全部 2,771 个 key；MX QAT 选中 184 个 Titan 参数（92 个 MoE 层各一个 `w13`、一个 `w2`），packed 权重全部被 QAT 覆盖，被选中的也全是 packed。
- CPU 单测（全部，GPU 屏蔽）：栈 1671 + 80 passed，main 1562 + 80 passed；只在栈上失败的 0 个；两边都失败的 3 个是 main 在本机就挂的（flux 两个、embedding 一个）；10 个只在 main 上失败（parallelism_context、cp_attention all-gather）。收集错误的 6 个文件是源码 torchao 版本号 `unknown`，第二遍用 nightly torchao 跑 80 passed。
- Elfie 的脚本单测（`scripts/validation/kimi_k3`）：移植后 15 passed。

## 本机跑不了的

- `validate_ep_placement.py`：要 Slurm 的 `scontrol` 读节点分配。
- `hybridep_stage_memory.py`（memory32）、`--dispatcher hybridep` / `deepep`：要 DeepEP hybrid-ep 和 NVSHMEM；MoonEP 要交换机 multicast。
- RL 的 `rl_grpo_kimi_k3_debug_batch_invariant`：本机没有 vLLM。
- 多节点、93 层真实模型、真实发布权重的加载（只查了索引和 key 映射）。
