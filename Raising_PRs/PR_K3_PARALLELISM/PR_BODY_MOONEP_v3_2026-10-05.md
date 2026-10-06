# PR 4751 body v3（`k3_moonep_seam` = `6e3de1b8d`），2026-10-05

## 状态（不粘贴）

- **10-06 用户说"不需要 GPU 的直接改"，已改（CPU 会话）：**
  - `moonep_review1` = `16ff9da7f`：rebase 到 main `3f087cf15`，原来的 7 个提交加上 3 个新提交；12 个文件，+1150 / −7。
    - rebase 的两处冲突：`pyproject.toml` 两行都保留；`moe.py` 按 main #4956 的写法，side stream 拿到的 `shared_TD` 也走 `remat.recompute_needs_tensor`（只多一层 `if shared is None` 分支）。其余 11 个文件逐文件比过，增删行和 rebase 前一致。
    - `8de0e6a97`：三个 op 的 region 名由 module 给出（dispatcher 用 `ep_communication.dispatch` / `.combine`，和 DeepEP 同名；experts 用 `moonep_experts`），`recompute=False` 不变。
    - `6741d20a5`：stream 测试加第二个用例，即 GPU 会话探针的测试版：两个 micro-batch，side stream 上的累加前 sleep，主 stream 在 MoE 输入的反向里读共享专家的梯度。
    - `16ff9da7f`：refusal 测试去掉没用的 `disable_cuda_graphs=True`。
  - 本机只跑得了 `test_moonep_ops.py`（1 passed）：main 现在的核心 import 链经过 KDA，需要 attn-gym 的 CuTeDSL，Windows 装不了。其余测试只做了 black 22.12（钉的版本）和 py_compile，要在 GPU 机器上跑。
  - PR 分支 `k3_moonep_seam` 仍是 `6e3de1b8d`，等 GPU 验证和用户同意后再同步。
  - 粘贴区改了三处：删掉逐参数梯度那句（没有噪声基线；梯度正确性由 GPU 单测对 fp32 稠密参考的逐项比较支撑）；microbench 表 MoE 行加 "(stream off)"；"forward stream" 改成 "main stream"。
  - **Test plan 的计数还没改**：GPU 应是 11 个（多了一个 stream 用例），CPU 要在 Linux 上重数，GPU 会话跑完再填。结果表是 `6e3de1b8d`（rebase 前）测的，rebase 后第 1 步是否逐位不变也要 GPU 会话确认。

- **10-05 CPU 会话复核 diff 和 body（用户："拉取，检查moonep分支diff和body"）：**
  - 三个分支一致：`moonep_review1` = `k3_moonep_seam` = #4751 head = `6e3de1b8d`，main `db050eb3f` 上 7 个提交，12 个文件、+1089 / −7。线上 body 已是 v3（08:03 UTC），标题仍带 "[DO NOT Review]"；mergeable 状态是 dirty（和 main 在 `pyproject.toml` 冲突），rebase 等用户的话。
  - diff 逐行看过：
    - c2 修复后 `_experts` 用 `out.mul_(weights[:, None])`，反向 `unscaled_grad_hidden` 先 `.float()` 拷一份算权重梯度，再原地乘权重，先后顺序对，没有覆盖还要用的值；和修复前逐位相同（GPU 会话随机张量和 6 个场景都比过）。
    - 新增注释都是一行约束；docstring 都是一行或两行的"做什么"；没有日志路径、没有实测数字；`_init_self_buffers`、`allocate_pools`、dispatch 的空表检查和我 10-04 提交的一致。
    - c3（核心 `moe.py` 的 `shared_experts_stream`）按用户的话留在本 PR。`x_TD` 上的 hook 保留，理由（side stream 上参数梯度的累加 autograd 不排序，FSDP 在反向结束前就会读）已写进 body，符合"用 hook 要写明缺哪个接缝"的规则。
  - body：粘贴区没有 we/our/us、破折号和非 ASCII 字符，正文约 845 词。要注意三处：
    1. **逐参数梯度那句没有噪声基线，也只报了中位数。** 按数值验收规则，第 1 步梯度要对着一个噪声基线报，并且报到每个参数。状态区记着两个离群值：第 12 层 KDA 的 `A_log` 在 efsdp = 2 / HSDP 下是 1.2e-2 / 1.5e-2（efsdp = 1 是 3.0e-3），第 0 层 `ffn_res_proj` 是 0.31 到 0.44；而 "standard EP 换一种规约顺序" 的逐参数基线没跑。审查的人问最大值时现在答不上来。建议：下次 H100 补这条基线（standard EP 在另一种布局下对 standard EP 逐参数比）再写，或者这句只保留 routed experts 的 5.4e-3 / 5.5e-3 / 5.8e-3，并写明没有基线。
    2. microbench 表的 "MoE layer with shared experts" 一行没写 `shared_experts_stream` 是开还是关（状态区说 stream 的数字这个 head 上没测，那这一行是关的）。Summary 刚介绍了这个开关，加半句 "(stream off)" 能避免误读。
    3. Design 里 hook 那句说 "makes the forward stream wait"，代码里等待的是前向时的当前 stream（主 stream），写成 "the main stream" 更准确。
  - 其余和代码一致：池在初始化时分配、dispatch 拒绝没 combine 的 plan、只重算激活、权重梯度用 `<grad_out @ W_down, hidden>`、slot 梯度异步规约、池大小 1.85 GB 的算法（896 × 3072 × 3584，EP 64）。

- **为什么重写：** PR head 从 `5e4596dc7` 变成 `6e3de1b8d`（c1 到 c5，c2 并入了修复，c3 共享专家 stream 留在本 PR）。v2 描述的是旧代码：反向重算 GEMM、combine 里乘权重、池在第一次前向懒分配，这些都不对了。要改的地方按 `MOONEP_ITEMS_10_20_2026-10-04.md` 的清单。
- **和 v2 比改了什么：**
  - Summary 加一条 `shared_experts_stream`。
  - Design 重写反向（只重算激活，6 个 grouped GEMM，路由权重在 expert op 里乘，权重梯度用 `<grad_out @ W_down, hidden>`，slot 梯度异步规约）、池在初始化时分配、dispatch 拒绝上一次没 combine 的 plan、共享专家 stream 的 hook 为什么需要（缺的接缝写一句）。
  - 代价段去掉"4 次 GEMM 对 3 次"，加 gate/up 保存和池大小（第 11 条：EP 64 时每个 rank 权重池、梯度池各约 1.85 GB）。
  - Results：路由行数表只留路由和 dispatch 两列（那张表是旧代码上测的，步时一列和新代码无关，去掉）；新增 microbench 表（old head 对本 head）；数值表换成本 head 上的 8 个格子；第 12 条的逐参数梯度一句。这三项在 H100 上跑，跑完填。
  - Test plan：CPU 28 passed（本机，`6e3de1b8d`）；GPU 10 passed on 4 H100s（`edb3f6f88`，树和 `6e3de1b8d` 相同）。
- **数字来源：** `kit_moonep_perf_2026-10-03/h100_moonep_final_1005.sh`，结果在本机 scratchpad 的 `h100_1004/moonep_final/`。
- **10-05 H100 填数（`6e3de1b8d`）：** microbench（old 和本 head 同一轮）、端到端 8 格（20 步）、第 12 条第 1 步逐参数梯度都已填进粘贴区。端到端和修 c2 之前的 c5 每一步都相同（c2 的修复逐位不变）。
  - 第 12 条的细节（不进 body）：和 efsdp = 1 逐参数比，只有第 12 层 KDA 的 `A_log` 超过 3 倍（3.0e-3 → 1.2e-2 / 1.5e-2），它不是 MoonEP 负责的参数；差最大的第 0 层 `ffn_res_proj`（0.31 到 0.44）在 efsdp = 1 也是 0.33，是梯度本身极小的注意力残差参数。efsdp = 2 和 HSDP 下"标准 EP 换一种规约顺序"的噪声基线没跑。
  - 共享专家 stream 的 MoE 层数字是 10-04 在 c5 上测的（72.57 → 71.02 ms），本 head 上没测，所以没进表。
- **粘贴前提：** PR 分支已经是 `6e3de1b8d`；#4751 是 draft，和 main 只在 `pyproject.toml` 冲突（推之前就有），rebase 等你的话。

- **10-06（PR head `16ff9da7f`，rebase 到 main `3f087cf15`）：** 端到端表在新 head 上重测（H100，同一份 cache；torch 68e0ae4 加 `local/mpp_shim`，见 `MOONEP_DIFF_REVIEW_2026-10-06.md`）。数值整体移动来自 main（纯 main 的 standard 第 1 步也是 8.18811）；MoonEP 和 standard EP 第 1 步三种布局都逐位相同。Test plan：CPU 39 passed（5060），GPU 11 passed（4 × H100）。microbench 没有重跑：新提交只改名和加测试。

- **10-06 更正：** 删掉"换布局最多变 1.1e-2"那半句。三种布局的起点权重不同（第 1 步 loss 就不同，standard EP 跨布局逐参数梯度相对差的中位数是 0.53 到 1.34），所以它不是规约顺序的噪声基线。按用户的话不补测基线。

--- PR 4751 body v3: PASTE BEGIN ---

## Summary

Adds MoonEP, the expert-parallel transport of the Kimi K3 report, as an optional EP backend: it keeps every rank's routed rows at `S x K` (its tokens times top-k) by prefetching copies of hot experts onto other ranks. Kimi K3 only until other models are validated.

- `MoonEPTokenDispatcher` (`models/common/token_dispatcher.py`), beside DeepEP and HybridEP.
- `MoonEPRoutedExperts` (`distributed/moonep/experts.py`), with MoonEP's buffer, pools and `torch.library` ops in `distributed/moonep/`.
- `TokenDispatcherTransform` gains `routed_experts`; `TokenDispatcherTransform(dispatcher=MoonEPTokenDispatcher, routed_experts=MoonEPRoutedExperts)` selects MoonEP.
- `MoE.Config.shared_experts_stream` (off by default) starts the shared experts on a side stream, so their GEMMs overlap the routed branch's communication, as the K3 report does.
- `config/validation.py`: MoonEP requires expert parallelism and is refused outside Kimi K3.

## Design

MoonEP moves expert weights as well as tokens: after dispatch it prefetches copies of hot experts into slots on other ranks, each rank's grouped GEMMs run over its own experts and then its slots, and in backward the slots' gradients are reduced into their home experts. The buffer and the pools are allocated once per process, with the model's buffers, and shared by every MoE layer; each rank copies its unsharded expert weights into its pool rows, since FSDP owns that storage.

Dispatch, the expert computation and combine are `torch.library` ops with an ordered effect, so selective and full activation checkpointing save their outputs rather than replay them over the shared pools; under `RegionAC` each op is a retained region. A dispatch refuses to start while an earlier one still waits for its combine.

The expert op scales its rows by their routing weights, so combine only sums. Its backward refills the pools for its plan, recomputes only the activation from the saved gate and up projections, and takes the routing-weight gradient as `<grad_out @ W_down, hidden>`, as in the K3 report, so the expert output is not kept. The slot gradients are reduced in fp32 on MoonEP's stream behind the input-gradient GEMMs, after an all-reduce on the EP group orders the slot writes.

With `shared_experts_stream`, a hook on the MoE input makes the main stream wait for the shared experts' backward: autograd orders the input gradient across streams, but not the side stream's parameter-gradient accumulation, which FSDP reads before the backward ends.

MoonEP's buffer is static: every dispatch carries exactly `num_max_tokens_per_rank` tokens, which the transform derives from the training shape.

Requirements:

- Hopper or newer behind an NVSwitch (NVLink multicast).
- MoonEP `33327eb` with `nvidia-cutlass-dsl` 4.6.2. MoonEP's version string has stayed 0.0.1 across API changes, so `moonep.py` checks the prefetch signature at import.
- Tokens and expert weights enter MoonEP in bf16; slot gradients are reduced in fp32.
- LoRA on the routed experts is not supported.
- No compile region may contain the MoonEP ops, which have no fake implementations; titan's local regions still compile, including SiTU-GLU inside the expert computation.

Costs, per MoE layer and micro-batch: the refill adds a second NVLink weight prefetch and a second local-expert copy, and the gate and up projections are kept for backward. In exchange the pools do not grow with the number of layers: for Kimi K3 (896 experts of 3072 x 3584) at EP 64 the bf16 weight pool and the fp32 slot-gradient pool take about 1.85 GB each per rank.

The gate and up rows are titan's `[F, D]`, where MoonEP documents `[H, H']`. The prefetch copies each expert as a block, so this matters only once quantized experts bring scale tensors, which MoonEP re-tiles per expert.

## Results

4 H100s behind an NVSwitch, MoonEP `33327eb`.

Routed rows per rank, Kimi K3 debug model with 128 experts and top-8, FSDP 4 x EP 4, seq 512, measured on an earlier revision of this branch (the routing and the dispatch are MoonEP's and unchanged since):

| routing | hottest rank over mean, experts on their home ranks | MoonEP dispatches at exactly `S x K` |
|---|---:|---:|
| natural | 1.54 | 1280 of 1280 |
| biased toward rank 0's experts | 3.19 | 1280 of 1280 |

Forward plus backward of two MoE layers at Kimi K3's expert width (latent 3584, hidden 3072), 4096 tokens and top-8 per rank, 32 experts, median of 20 iterations and peak memory per rank; the previous head of this PR recomputed the expert GEMMs in backward:

| | previous head `5e4596dc7` | this revision |
|---|---:|---:|
| routed experts, uniform routing | 72.20 ms / 8.02 GiB | 64.46 ms / 8.58 GiB |
| routed experts, skewed routing | 73.48 ms / 8.02 GiB | 65.11 ms / 8.58 GiB |
| MoE layer with shared experts (stream off) | 76.82 ms / 8.10 GiB | 69.53 ms / 8.66 GiB |


Loss / grad norm, Kimi K3 debug model (8 experts, top-2), seq 512, deterministic, one warm compile cache; the 32-sample debug set is memorised, so the runs stop at step 20:

| cell | step 1 | step 10 | step 20 | max relative loss gap to standard EP of the same layout |
|---|---:|---:|---:|---:|
| standard EP, FSDP 4 x EP 4 | 8.18811 / 2.1875 | 5.82440 / 10.7500 | 4.46201 / 8.0000 | |
| MoonEP, FSDP 4 x EP 4 | 8.18811 / 2.1875 | 5.82486 / 10.6875 | 4.46116 / 8.0000 | 1.60e-3 |
| standard EP, dp_shard 4 x EP 2 | 8.19370 / 2.2031 | 5.83694 / 10.3750 | 4.42467 / 8.3125 | |
| MoonEP, dp_shard 4 x EP 2 | 8.19370 / 2.2031 | 5.83617 / 10.3750 | 4.42168 / 8.3750 | 2.03e-3 |
| standard EP, HSDP 2 x 2 x EP 2 | 8.17808 / 2.2031 | 5.85455 / 10.3125 | 4.43081 / 8.7500 | |
| MoonEP, HSDP 2 x 2 x EP 2 | 8.17808 / 2.2031 | 5.85582 / 10.3125 | 4.43006 / 8.7500 | 1.89e-3 |

Standard EP run twice is identical on all 20 steps, and MoonEP under full activation checkpointing matches it under selective.

## Test plan

- `pytest tests/unit_tests/cpu/test_transforms.py tests/unit_tests/cpu/test_moonep_ops.py -q` (39 passed).
- `pytest tests/unit_tests/gpu/test_moonep.py tests/unit_tests/gpu/test_moe_shared_experts_stream.py -q` (11 passed on 4 H100s; `test_moonep.py` skips without MoonEP or NVLink multicast).

## Relation to earlier revisions of this PR

Earlier revisions subclassed `GroupedExperts` with a buffer and pools per MoE layer, and later recomputed the expert GEMMs in backward; this revision keeps one buffer and one set of pools per process and recomputes only the activation.

--- PASTE END ---
