# PR 4751 body v3（`k3_moonep_seam` = `6e3de1b8d`），2026-10-05

## 状态（不粘贴）

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

With `shared_experts_stream`, a hook on the MoE input makes the forward stream wait for the shared experts' backward: autograd orders the input gradient across streams, but not the side stream's parameter-gradient accumulation, which FSDP reads before the backward ends.

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
| MoE layer with shared experts | 76.82 ms / 8.10 GiB | 69.53 ms / 8.66 GiB |


Loss / grad norm, Kimi K3 debug model (8 experts, top-2), seq 512, deterministic, one warm compile cache; the 32-sample debug set is memorised, so the runs stop at step 20:

| cell | step 1 | step 10 | step 20 | max relative loss gap to standard EP of the same layout |
|---|---:|---:|---:|---:|
| standard EP, FSDP 4 x EP 4 | 7.99090 / 2.4219 | 4.89817 / 7.1562 | 3.63110 / 4.7500 | |
| MoonEP, FSDP 4 x EP 4 | 7.99090 / 2.4219 | 4.90077 / 7.1875 | 3.62509 / 4.6875 | 2.90e-3 |
| standard EP, dp_shard 4 x EP 2 | 7.99403 / 2.5469 | 4.88217 / 6.8438 | 3.62059 / 4.5312 | |
| MoonEP, dp_shard 4 x EP 2 | 7.99403 / 2.5469 | 4.88064 / 6.8438 | 3.60856 / 4.5000 | 3.83e-3 |
| standard EP, HSDP 2 x 2 x EP 2 | 7.99649 / 2.5156 | 4.85360 / 6.7812 | 3.58470 / 4.6875 | |
| MoonEP, HSDP 2 x 2 x EP 2 | 7.99649 / 2.5156 | 4.84769 / 6.6875 | 3.57996 / 4.6875 | 4.63e-3 |

Standard EP run twice is identical on all 20 steps, MoonEP under full activation checkpointing matches it under selective, and moving standard EP to the other two layouts changes its loss by up to 1.6e-2.

Step-1 gradients against standard EP of the same layout, per parameter `||a - b|| / ||a||`: the routed experts' gradients are within 5.4e-3, 5.5e-3 and 5.8e-3 in the three layouts, and the median over all 465 parameters is 5.2e-3 to 5.3e-3.

## Test plan

- `pytest tests/unit_tests/cpu/test_transforms.py tests/unit_tests/cpu/test_moonep_ops.py -q` (28 passed).
- `pytest tests/unit_tests/gpu/test_moonep.py tests/unit_tests/gpu/test_moe_shared_experts_stream.py -q` (10 passed on 4 H100s; `test_moonep.py` skips without MoonEP or NVLink multicast).

## Relation to earlier revisions of this PR

Earlier revisions subclassed `GroupedExperts` with a buffer and pools per MoE layer, and later recomputed the expert GEMMs in backward; this revision keeps one buffer and one set of pools per process and recomputes only the activation.

--- PASTE END ---
