# FP8 激活存储：报告原文、titan 现状、要做什么（2026-09-29）

用户 09-29 问："具体来说 这里应该做什么？titan没有这个先例吧？它难道不是offload和balance的前置吗？现在有统一管理器这个东西吗？在哪个分支或者pr里面"

## 1. 原文

- **K3 报告 §5.2.2（`phase13_k3like_48b_posttrain/official_k3/report.txt` 第 1355 到 1363 行）：** "Unified activation manager We design a unified storage abstraction for activations, in which every tensor saved for the backward pass is associated with a pluggable storage backend. Recomputation, quantization, and offload/remote-offload are merely storage policies under this abstraction and can be freely composed at tensor granularity; policies are declared via lightweight annotations on tensors, fully decoupled from the model code. ... In Kimi K3, most activations use block-wise FP8 quantization [58, 30] combined with offload/remote-offload, and element-wise operators are configured with recomputation."
- **[58] 是 K2 报告（arXiv 2507.20534，09-29 在线核对原话）：** "Inputs of MoE up-projections and SwiGLU are compressed to FP8-E4M3 in 1×128 tiles with FP32 scales. Small-scale experiments show no measurable loss increase." 以及 "Due to potential risks of performance degradation that we observed during preliminary study, we do not apply FP8 in computation." 同一节还有：LayerNorm、SwiGLU、MLA up-projection、MoE down-projection 重算；"All remaining activations are offloaded to CPU RAM."
- **[30] 是 DeepSeek-V3 报告。** V3 的 GEMM 本身跑 FP8，缓存 FP8 激活是 FP8 计算顺带的结果；K2、K3 是 BF16 计算，FP8 只用在存储上。
- 所以 K3 的做法：计算保持 BF16；保存给反向的张量在存下时压成 FP8 E4M3，沿 hidden 维每 128 个元素一个 FP32 scale，反向读时再还原成 BF16。每个元素 1 + 4/128 ≈ 1.03 字节，约为 BF16 的 0.52 倍。量化后的数据可以留在设备上，也可以再交给 offload 或远程 offload。

## 2. titan 和 torch 里有什么

- **"BF16 计算、只在保存时压成 FP8"：main 没有。** main 的 K2.5 recipe 明确没做：`torchtitan/models/kimi_k2_7/config_registry.py:225` 写着 "The report uses BF16 compute; its FP8 path only compresses saved activations."
- **最接近的先例是 `MXFP8Linear` 的 `input_activation_format_for_backward="mxfp8"`**（`torchtitan/quantization/mxfp8/README.md` 的 "Input Activation Storage"，由 `linears_saving_inputs_for_backward_in_mxfp8` 按模块 opt in）：把 WGRAD 要用的输入以 MXFP8 存下。它依附在 MXFP8 计算上（Blackwell 的 MXFP8 GEMM），是 GEMM 算子自己的保存格式，不是一个可以和 offload 组合的存储策略。README 也写了它和 activation checkpointing 的交互问题。
- **`graph_trainer`（experiments，编译路径）的 `memory_policy.py`：** 在 trace 出来的前反向图上，给每个保存的激活打 MUST_SAVE、MUST_RECOMPUTE 或 MUST_CPU_OFFLOAD 标签，由 `--compile.memory_policy` 选策略；CPU offload 用 torch 的 `ao::offload / reload / wait_tensor`。这是 titan 里离"统一管理器"最近的东西，但只在编译路径，没有量化，也没有远程 offload。
- **torch 0928 nightly：** `CheckpointPolicy` 只有 MUST/PREFER 的 SAVE、RECOMPUTE、CPU_OFFLOAD 六个值，没有量化；`torch_remat` 的 `SavedTensorInfo` 只有 `kind`（CHECKPOINT_INPUT、BACKWARD、SAVE_OUTPUT）和 `context`，看不出是哪个算子保存的。

## 3. 我们的统一管理器在哪

- **`ActivationStorage`，`torchtitan/distributed/activation_storage.py`，由 #4765 引入**（分支 `k3_pp_offload` = `pp_offload_review1` = `2140058c5`，draft，不在 main）：每个保存给反向的张量经 `torch_remat.saved_tensors_hooks` 进来，策略按张量返回一个 `CheckpointPolicy`；SAVE 留给 autograd，CPU_OFFLOAD 交给 `route(chunk)` 指定的后端（#4765 只有 `HostBackend`）；显存都在计算流上分配；反向按层预取。
- **#4764**（`k3_pp_balance` = `224bdbaf4`）在它上面加 `RemoteBackend`（mooncake，tcp / rdma / nvlink_intra）和规划器 `PPMemoryController`（profile 一步，把各 rank 峰值压向均值，决定哪些 stage micro-batch 去 host、哪些停到别的 rank）。
- **重算不经过这个管理器：** 由 titan 现有的 AC（`torch_remat` 的 region、checkpoint，FullAC、RegionAC）决定，管理器只看到 AC 决定保存的张量，所以两者按张量组合，但不是同一个策略函数；策略返回 RECOMPUTE 时管理器会报错。
- **现在只接在 K3 的 PP 路径上：** 在 `pipeline_kimi_k3` 里、`pp_memory.cpu_offload != "none"` 或 `balance` 打开时才建。
- **对照报告 §5.2.2：** 可插拔后端、策略与模型代码解耦、计算流上分配、按层预取、远程 offload 都有；量化没有。

## 4. 要做什么

1. **编解码：** 存下时 BF16 → FP8 E4M3，1×128 块，FP32 scale；读回时还原成 BF16。放在计算流上（它是计算，量化完原张量就可以释放）。先用纯 torch 写对，再换 torchao 的 1×128 blockwise 量化 kernel（titan 已把 torchao 当可选依赖）。
2. **策略词汇：** "量化"和"放在哪"是两件事。`CheckpointPolicy` 没有量化这个值，所以策略的返回值要多一个格式（bf16 或 fp8），和 SAVE / CPU_OFFLOAD / 远程正交：量化后留在设备、去 host、去别的 rank 都可以。
3. **选哪些张量：** K2 只压 MoE up-projection 的输入和 SwiGLU 的输入；K3 说"大部分激活"。现在的策略只知道 `kind` 和 (stage, micro-batch, layer)，分不出是哪个算子存的。要在 `capture_context` 里带上产生张量的模块 FQN（`graph_trainer` 用的也是 `module_fqn`），策略才能按"专家的 w1、w3 输入"这样的规则选。这就是报告说的"annotations on tensors, decoupled from the model code"。
4. **数值证据：** offload 和 balance 是逐位相同的，FP8 是有损的。要按数值表规则：C4 上 100 步，第 1、10、50、100 步，同一份 cache，一行噪声底；外加每个张量还原后的相对误差。K2 报告对 FP8 存储只有一句 "Small-scale experiments show no measurable loss increase."，没有给数据，要自己测。
5. **放在哪个 PR：** 它需要 #4765 的管理器，和 #4764 无关。建议作为第三个 PR 叠在 #4765 上，和 #4764 并列；不并进 #4765、#4764，那两个保持"数值逐位不变"。

## 5. 是不是 offload 和 balance 的前置

- **管理器是前置，已经在 #4765 里**；offload（#4765）和 balance（#4764）都挂在它上面。
- **FP8 不是机制上的前置：** offload 和 balance 在 BF16 上工作，逐位不改数值。FP8 和它们组合后，要搬、要存的字节约减半。
- **要复现报告的显存数字时，FP8 是必需的：** `PP_ACTIVATION_STORAGE_REPORT_2026-09-25.md` §4 里"报告配方"那几行（模型值）按 FP8 存储算了每层约 15 个单位；不压 FP8，搬的字节翻倍，H100 上 25 到 50 GB/s 的 host 链路会饱和。
