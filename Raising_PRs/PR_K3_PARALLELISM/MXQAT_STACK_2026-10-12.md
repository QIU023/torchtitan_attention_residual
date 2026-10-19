# Elfie 的 MX-QAT 分支 + 我们的 PR 叠栈（2026-10-12）

> **10-19：** 当前栈头是 `2eff57f99`，不是下面的 `92afeeeee`。逐提交和各 PR 现在的 head 对照、各次验证在当前栈头上还算不算数，见 `STACK_REASSESS_2026-10-19.md`。

用户要求：从 `elfiegg/torchtitan` 的 `work/kimi-k3-mx-qat-distmuon` 拷一个分支，先 rebase 到 main，再把 #4380、#4381、#4751、#4656、#4963 各压成一个提交叠上去，推送。

## 当前状态

- fork 分支 `k3_mxqat_stack_20261012` = `92afeeeee`，已推（新分支，不是 PR 分支）。底是 upstream main `b663c1673`（10-12）。worktree：`scratchpad/wt_mxqat`。
- 叠了 4 个 PR：#4380、#4381、#4751 各一个提交。#4656 和 #4963 没有叠，原因见下面一节，等用户决定。

| 提交 | 内容 |
|---|---|
| `ae2a0002c` … `d3b5271d6` | Elfie 的 9 个提交（原 `50c7ab939` … `b7b5eb2b9`），rebase 时只做文本冲突解决 |
| `4b8559c51` | 适配 main 的 API（一个提交，见下） |
| `de07679ff` | #4380 压缩（PR 头 `097068e0f`，9 个提交，底 `01b1b69e3`） |
| `525fd3e47` | #4381 压缩（PR 头 `4e2ec143b`，10 个提交，底 `db050eb3f`） |
| `92afeeeee` | #4751 压缩（PR 头 `16ff9da7f`，10 个提交，底 `3f087cf15`） |

提交都没有 trailer。

## Elfie 分支适配 main（`4b8559c51`）

- `model_registry` / `kimi_k3_configs` 换成 `build_model_config(..., parallelism=ParallelismConfig(enable_sequence_parallel=False))` / `MODEL_FLAVORS`；recipe 从 `torchtitan_recipes.tests.models.kimi_k3` 取，`ConfigManager().parse_args` 换成 `ConfigLoader().load`（main 的 CLI 不再能改 `--training.steps`，那条断言删了，测试改名 `test_standard_model_entrypoint`）。
- `RouterGateLinear` / `CastLinear` 换成 `HiMidLoLinear`（K3 的 lm_head 两边都是普通 `Linear`，router gate 两边都是自定义算子，语义不变）；`Module`、`dtensor_safe`、`DistMuon` 的新路径。
- MX QAT 在 LoRA 之前、MX QAT 自身互斥：从类属性 `run_after` / `conflicts_with` 移到 `relations.py` 的 `_BUILTIN_PRECEDES` / `_BUILTIN_CONFLICTS`（main 的写法）。
- MX QAT 的 dense linear 关掉 `inplace_wgrad_accum`：GEMM 读的是 fake-quant 之后的副本，不是叶子参数，main 的检查会直接报错。
- LoRA 测试用 `target_modules=["projection"]` 代替旧的 `LinearLoRAHandler`（main 的 grouped LoRA 要求 rank 是 8 的倍数）。
- `validate_distmuon.py` 的 `--dispatcher` 用 `TokenDispatcherTransform`，DeepEP / HybridEP 的 `num_max_tokens_per_rank` = `seq_len * microbatch_size`（脚本没有 CP / TP，等于 Elfie 底上按训练形状推出来的值）。
- `LinearCrossEntropyLoss` 去掉 `compile_config` 参数（main 没有 `CompileConfig`）。

## 各 PR 压缩时的适配

- #4380：main 的 #5277 把 `torchtitan_recipes/tests/suites/h100.py` 挪到 `tests/integration_tests/h100.py`，两行 `dynamic_cp_min_patches = 256` 跟过去；`vision_encoder.py` / `flavors.py` 只是 import 冲突。
- #4381：B200 的 `vision_dep.enabled = True` 跟到 `tests/integration_tests/b200.py`；`model.py` 里 main 新加的 `vision_bank_indices_T`、`aux_loss_denominators`、`local_compile_regions` 和 DEP 的 `vision_embeds`、`vision_dep` 都保留（两条路径在 `_prepare_multimodal_embeds` 里本来就能组合）；`BaseModel` 改从 `torchtitan.models.base.model` 导入。
- #4751：main 的 `TokenDispatcherTransform` 现在要显式传 `num_max_tokens_per_rank`（不再从训练 context 推），MoonEP 的 `routed_experts` 转换接在 main 的版本上，测试里 `_MOONEP` 补 `num_max_tokens_per_rank=512`；共享专家的加法用 main 的 `remat.region(torch.add, "shared_add", recompute=False)`，MoonEP 的侧流等待保留，原来的 `recompute_needs_tensor(shared_TD)` 去掉（main 的区域不重放这个加法，不需要保存它的输入）；GPU 测试的 `Module` 导入路径。

## #4656、#4963 为什么没叠

main 的 #5033（acisseJZhong，10-09）把 attention residual 做成 `AttentionResidual` 可配置函数，`@local_compile("attention_residual", dynamic=True)`，`local_compile` 强制 `fullgraph=True`，K3 的 `local_compile_regions` 默认就开着它；它和 Triton override（`torchtitan_recipes/overrides/kimi_attention_residual.py`，827 行，带 GPU 测试）的接口都是堆叠张量 `block_residual_TND`。#4656 把载体改成列表。

- 照搬（列表传进编译函数）：dynamo 按列表长度和有没有 partial 各编一张图。按 flavor 数：debugmodel 17 层、块大小 4、5 个块，要 9 张图；Kimi-K3 93 层、块大小 12、8 个块，要 15 张。默认上限 8，`fullgraph=True` 下超限直接 `FailOnRecompileLimitHit`（CPU 上用 eager backend 复现过）。
- 在调用处先 stack：只编一张图，但每次调用都多一份 `[T, k, D]` 拷贝作为图输入被存下，比 main 还费显存，#4656 的意义（shuhuayu r4099737626 的 21 对 6）就没了。
- #4963 叠在 #4656 上，跟着卡住。

这是 #4656 自己 rebase 到 #5033 时就要解决的设计问题，不在叠栈里临时决定。可选：等 #4656 正式 rebase 再叠；或者这个分支上叠列表版本，同时把 `attention_residual` 移出默认 `local_compile_regions`（丢掉 #5033 的编译加速）。

## 测试（5060 盒子，只用 CPU，`CUDA_VISIBLE_DEVICES=`）

- torchao：nightly `0.19.0.dev20261012` 没有 `mx_fake_quantize` / `mx_fake_quantized_grouped_mm`；Elfie README 说要"兼容的 TorchAO"，实际是她的 `elfiegg/ao` 分支 `pr/mx-fake-quant-ste`（`6352062`），用源码放 PYTHONPATH（`scratchpad/elfie_ao`），不装进 venv。
- Elfie 的 8 个 CPU 测试文件：66 passed。
- 栈顶合并跑（Elfie 的 + #4380/#4381/#4751 的 CPU 测试 + main 的 config、K3、loss、LoRA、HiMidLo、quantization、trainer 测试）：450 passed，1 failed = `test_lora.py::test_lora_forward` 的 inductor `CppCompileError`，干净 main 上同样失败（这台机器的 C++ 编译环境），与栈无关。`test_integration_test_definitions.py` 17 passed（要用有版本号的 nightly torchao 跑，源码 torchao 的版本是 `unknown`，`packaging` 解析不了）。
- 各 PR 自己的数：#4380 plan 7、#4381 DEP 22（5 + 17）、#4751 MoonEP ops 和 transform 测试都过。
- 没跑：所有 GPU 测试（5060 归 SATS-OPRD）、`tests/integration_tests/test_kimi_k3_mxfp4_checkpoint.py`、`scripts/` 下要检查点或 GPU 的脚本。
