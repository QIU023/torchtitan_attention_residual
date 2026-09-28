# MoonEP #4751 diff 重审（2026-09-28）

用户："MoonEP那边diff重新审核 我的理解是我们不应该在调用moonep库之外做太多的改动 但是我记得好像diff还很多 这真的是对的吗 校验"

审的是 PR #4751 的 head `f556ab4fd`（`k3_moonep_seam`，`refs/pull/4751/head` 已核对），base 是 main `6c2dadbb3`（09-18）。一共 8 个提交、13 个文件、+1072/−12，其中测试 469 行，生产代码约 600 行。前面两份审计（`DIFF_AUDIT_MOONEP_2026-09-19.md`，以及 `DIFF_REVIEW_ALL_2026-09-27.md` 的 MoonEP 一节）的结论在下面引用，不再重推。对照物有两个：MoonEP 公开版 `33327eb`（09-20，目前 MoonEP 的最新提交；源码在 `/tmp/moonep_src`），以及 titan 自己的 DeepEP、HybridEP 集成。

## 0. 结论

**不对，理由有三层。**

1. **专家侧那一半，写的是 main 已经删掉的接口。** 分支的底之后 main 又进了 63 个提交：
   - #4871、#4872 删掉了 `GroupedExperts`（连同 `w1_EFD`/`w2_EDF`/`w3_EFD`、`_grouped_mm`、`inner_experts`）。现在是 `RoutedExperts` 直接持有两个 `GroupedLinear`：融合 gate、up 的 `w13`（`[E, 2, F, D]`），以及 `w2`（`[E, D, F]`）。
   - #4810 去掉了分支挂靠的 `MoE.parallelize` 和 dispatcher 的 `wire_meshes`，`init_buffer` 改由 `RoutedExperts` 调。
   - #4905 把 `ParallelDims` 改名成 `ParallelismContext`。
   
   所以 `MoonEPGroupedExperts(GroupedExperts)`、`KimiLatentMoE.parallelize` 的覆盖、`check_moonep_mesh(parallel_dims)` 都不是解冲突就能过去的，要按新结构重写。
2. **即使只看分支本身，也有约 170 行生产代码和 87 行测试超出了"调用 MoonEP"**（按代码估算，没写出替代实现）。其中一部分是公开版发布以后已经没有必要的，见 §2。
3. **公开版上没验证过。** 迁到 `33327eb` 的两个提交只跑过 CPU 测试和两卡 H100 上的交错探针。on-device test 和 h100 格都没在公开版上跑过（09-27 审计）。body 的 Design、Requirements 还在描述迁移之前的代码，见 §4。

## 1. 哪些是调用 MoonEP 本来就要的（保留）

- **dispatcher 子类**（`MoonEPTokenDispatcher`，约 140 行）：建 `Buffer`，EP=1 时退回本地 dispatch，dispatch 和 combine 各包一个 autograd Function（约 60 行），反向就是同一个 plan 上的另一个 kernel。titan 的 DeepEP 也有对应的部分，dispatcher 类约 106 行。
- **专家侧的三件事**：GEMM 之前 `prefetch_weight`；GEMM 按 `cu_seqlens` 算"本 rank 的专家行加预取槽"；反向之后 `reduce_grad`。MoonEP 搬的是专家权重，不只是 token，所以专家侧必须改，这是 MoonEP 和 DeepEP 的根本区别（09-19 审计）。
- **网格约束**：`efsdp == 1`，不支持 `dp_replicate`（`check_moonep_mesh`，18 行）。MoonEP 把每个 rank 整块专家映射到 NVLink 上，FSDP 切开的专家给不了这个。
- **反向时按本 plan 重新预取**：09-22 在两卡 H100 上确认过，1F1B 交错时不这样做，梯度会错。
- **其他**：`make_token_dispatcher_config` 里加 `"moonep"`（11 行），K3 按后端选专家类（7 行），一个 h100 CI 格（17 行）。

## 2. 超出调用的部分

| 项 | 在哪 | 为什么不需要 | 依据 |
|---|---|---|---|
| 每个 MoE 层一个 `Buffer`、一套池、一套表 | `token_dispatcher.py:1035` `init_buffer` 按 dispatcher 建；dispatcher 每层一个（`moe.py:139`）；`moonep.py:110-194` 每层给 3 个投影各建 bf16 预取池、fp32 规约池、fp32 本地梯度；`moe.py:825` `attach` 每层建 `[2*own]` 表 | MoonEP README（公开版）："Each rank's prefetch pool is process-global and shared by all layers"。titan 的 DeepEP 用 `get_buffer()` 拿进程级的 `ElasticBuffer`，只在组变了或要更大时才重建。反向本来就按本 plan 重新预取，所以各层共用一套池也成立 | 读代码；09-22 已确认"每层一套"，没有改。显存随 MoE 层数线性增长（按代码推算，没实测） |
| 单独的 `[2*own]` 表，每次前向把本地权重和槽都拷进去 | `moe.py:859` `_refresh_rows`，`moe.py:867` `_refill_slots` | 09-19 说它不可再缩，前提是 MoonEP 没有给 `[E + B]` 连续范围提供分配器。公开版之后池只要求"每个 rank 一段连续，段间可以留空"（`api.py` 的 `_validate_rank_strided_pool`），所以每个 rank 的块可以建成 `[2*epn]`：前一半放本地权重的拷贝，后一半是预取槽，这块本身就是 GEMM 要的计算视图。预取时传 `pool[:, epn:]`，本地权重传 `pool[rank, :epn]`，不用另外建表、不用把槽拷进表里 | 读公开版的 API 约定，没跑 |
| 权重和梯度每步的转置拷贝 | `_refresh_rows` 把 3 个参数转置成 `[in, out]`；反向把 3 个梯度转置回来再 `.contiguous()` | 公开版的预取只检查本地张量和池的后两维一致（`api.py` 的 `prefetch_weight`），按 128×128 的 TMA 块搬运，不管哪维是输入。池直接用 titan 自己的布局就不用转置 | 读 kernel 和断言，没跑；09-19 当时只提了这个问题，没有结论 |
| 三遍专家 GEMM | `_MoonEPExpertFunction`：前向在 `no_grad` 下算一遍，反向重算一遍再求梯度 | 这是"表加拷贝"写法带来的，MoonEP 不要求；可以在前向留下中间结果。body 的代价列表里也没写 | 09-22 已确认 |
| core 的两个类变量 | `token_dispatcher.py:175` 和 DeepEP、HybridEP 各加 `static_token_capacity`，Base 加 `requires_ep`；`update_ep_token_dispatcher_config` 改成读类变量；另有 87 行测试 `test_ep_token_dispatcher_capacity.py` | 只为接 MoonEP 的话，在原来按名字列出 DeepEP、HybridEP 的那处加上 MoonEP，EP=1 时跳过就够了（约 4 行）。另外 09-22 的笔记写"两个类变量合成了一个"，现在的代码里还是两个 | 读代码 |
| K3 里的 `attach` | `kimi_k3/moe.py:58` 覆盖 `parallelize`，把专家、dispatcher、池接起来 | 专家类在 common 里，接线却在模型里，每个用 MoonEP 的模型都得再写一遍。有了进程级的 buffer 和池，就不需要 attach 了 | 读代码 |
| 选后端的 flavor | `kimi_k3/config_registry.py:95` `kimi_k3_debugmodel_moonep` | CLAUDE.md 的 flavor 规则说选后端的 flavor 不加。09-22 加它是照 upstream 的 `qwen3_moe_deepep` 先例，两者冲突，要用户定（09-27 审计已提）。不加的话，h100 recipe 在函数里直接用 `model_registry(..., moe_comm_backend="moonep")` 就行 | 规则和先例 |

合计：上表能删或能缩的，大约 170 行生产代码，加上 87 行测试（估算）。09-19 量的注释比例是 9.9%，#4577 是 4.3%。

## 3. 按 main 现在的结构重写时，要先定的一点

- 公开版的 `prefetch_weight`、`reduce_grad` 固定要 gate、up、down 三个投影，每个都是连续的 `[epn, H, H']`，而且三个都不能是 None（`api.py` 的断言）。
- main 现在把 gate 和 up 融合成一个 `w13` `[E, 2, F, D]`。`w13[:, 0]` 在专家之间不连续，不能当成单独的投影传进去。
- 可选的办法：
  - 请 MoonEP 接受任意个数的投影（给 MoonEP 提一个需求）；
  - 把整个 `w13` 按 `[epn, 2F, D]` 当成一个投影传进去，另外两个位置怎么填，要看 MoonEP 愿不愿意放开断言。
  
  这一点要在读 main 的 `RoutedExperts`、`GroupedLinear` 接缝之后定，不能照搬分支的写法。

## 4. body（`PR_BODY_MOONEP.md`）里过时的句子

- Design 里 "The two kernels are called directly rather than through `Buffer.prefetch_weight` and `Buffer.reduce_grad`"：现在的代码调的正是这两个公开接口。
- Requirements 里 "one fp32 `[E, in, out]` gradient table"、"Two EP-group barriers per MoE layer"：公开版在 kernel 里自己做 fence，`[E]` 表也已经去掉（状态区第 6 行自己写了）。
- cutlass 的版本要求还是旧的 4.6.0 加一行补丁；公开版已经升到 CuTe DSL 4.6.2。
- 代价列表里没写三遍专家 GEMM，也没写每层一套池、显存随层数增长。

## 5. 建议（一个立场）

- 不再在 `f556ab4fd` 上 rebase。按 main 现在的结构重写一个薄的集成：
  - 传输层：进程级的 `Buffer` 和池，参照 DeepEP 的 `get_buffer`。每个 rank 的池块直接当计算视图，用 titan 自己的权重布局，不另建表，不转置。
  - 专家侧：先查 main 的 `RoutedExperts`、`GroupedLinear` 接缝再动手（#4577 规则），`w13` 的问题按 §3 定。
  - core：只在 `make_token_dispatcher_config` 和 `update_ep_token_dispatcher_config` 里点名 MoonEP，别的不动。
  - flavor 按用户的决定处理。
- 重写之后必须在 NVSwitch 机器上验证（multicast 为 1，见记忆 `moonep-needs-switch-multicast`）：on-device test、h100 格、对 standard dispatcher 的数值，以及和 DeepEP 的对比。这台 5060 和已经停掉的单卡 H100 都跑不了 MoonEP。
- #4751 是已发布的 draft：重写先推 review 分支，同步 PR 分支要等用户说。
