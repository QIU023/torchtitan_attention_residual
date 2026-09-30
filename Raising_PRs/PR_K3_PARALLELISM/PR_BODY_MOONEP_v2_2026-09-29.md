# PR 4751 body v2（MoonEP 重写，`moonep_review1`），2026-09-29

## 状态（不粘贴）

- **09-30 复查（CPU 这边，用户："检查DEP和MoonEP body和diff，现在这两个可以去H100跑了吗？"）：** diff 和 Design 对得上；reduce 前的栅栏语义成立（本 rank 写完槽梯度，再发 all-reduce，当前流等它完成才进 `reduce_grad`）。粘贴区补了两处：Requirements 加一条 bf16（`prefetch_rows` 的权重池固定 bf16，`dispatch_tokens` 把 token 转 bf16），代价那条写明权重预取也跑两次。
- **09-29 深夜（用户："MoonEP rebase main后 直接覆盖draft PR分支"）：** PR 分支 `k3_moonep_seam` 和 review 分支 `moonep_review1` 都 force-with-lease 推到 `30157477b`，upstream `refs/pull/4751/head` 已是它。旧 PR head `f556ab4fd` 在 `backup/k3_moonep_seam_pre_rewrite_20260928`，旧 review head `1633dcd79` 在 `backup/moonep_review1_pre_20260929b`。
  - 换到 upstream main `46ec3f232` 上，三个提交 `5ae489e21`、`16c6aa691`、`30157477b`，重放没有冲突（range-diff 三个都相同）。
  - 另外在第一个提交里修了新 torch 类型存根下多出的 4 个 pyrefly 错误，只动类型：三个 autograd `Function` 的 `forward` 加 `# pyrefly: ignore[bad-override]`（和同文件的 `backward` 一样）；`torch.autograd.grad` 的输入标成 `list[torch.Tensor]`。
  - 检查：pyrefly 16 个错误，和 main 逐条相同；pyflakes、ufmt、pre-commit（分范围）干净；CPU 53 passed、13 subtests passed；5060 上接假 MoonEP 的两卡 GPU 单测 2 passed；4 卡 h100 格子对 standard，第 1 步逐位相同，之后相对差 5.7e-05 到 1.5e-03，在 standard 自己换缓存的噪声底（2.4e-03）以内（`kit_moonep_rewrite_2026-09-29/local/e2e/run_e2e_0929n.sh`，只进 logbook）。
  - **GitHub 上的 body 还是旧实现的**（Design、Requirements 说的是发布前的代码）：用下面的粘贴区整段替换。Results 仍是 Pending (H100)，要真 MoonEP 的正确性和负载数字（H100 SXM，NVSwitch multicast）。标题建议见下，要不要去掉 "[DO NOT Review]" 你定。
- **09-29 夜（overnight T3，审计文档 §7）：** review 分支 `moonep_review1` = `1633dcd79`，upstream main `a182e530a` 上 3 个提交：原来两个重放后是 `a93b3ad28`、`58b18c61f`，加一个修复 `1633dcd79`。旧 head `a505f74a8` 在 `backup/moonep_review1_pre_20260929`。
  - 修复一：规约前补一道栅栏。MoonEP 的规约远程读各 rank 的槽梯度，自己不带 barrier，要求调用方先保证所有 rank 写完；重写版写完马上规约，中间没有跨 rank 同步。现在在 EP 组上做一次单元素 all-reduce。
  - 修复二：`Buffer` 设 `explicitly_destroy=True`，和 DeepEP 一样，不在 GC 时跑 `destroy()` 的同步和 barrier。
  - 检查：CPU 34 passed、13 subtests passed，pyflakes 和 ufmt 干净。09-29 在 H100 上真实 MoonEP 的 GPU 单测 2 passed，那是修复之前的 head，新 head 还要在 H100 上再跑一次。
  - 负载统计的方案在 `kit_overnight_2026-09-29/moonep/`（128 个专家、top-8、偏置路由加自然路由对照，每层每个 rank 的负载，MoonEP 每次 dispatch 是否正好 S × K 行），下次上 H100 约 25 分钟（负载格 15 分钟，smoke 10 分钟）。粘贴区的 Results 和 Test plan 已改成负载统计加步时，Requirements 加了 LoRA 和规约栅栏两条。
- **用户 09-28：** "直接重写 备份一下当前moonep tree 推到我晚上提供4 h100的时候能直接smoke然后打开正式pr的程度"。
- **备份：** 旧实现 `f556ab4fd` 在 fork 的 `backup/k3_moonep_seam_pre_rewrite_20260928`，本地 tag 同名。PR 分支 `k3_moonep_seam` 还是 `f556ab4fd`，今晚 smoke 通过、你说同步以后再推。
- **分支（09-29 白天，已被上面的 `1633dcd79` 取代）：** review 分支 `moonep_review1` = `a505f74a8`，main `5dc97a3e7` 上的 2 个提交（代码、测试），10 个文件 +605/−3：生产代码 +428（旧实现 +595），测试 +177（旧实现 +477），Kimi K3 模型目录 0 行（旧实现 47 行）。没有 trailer。审计见 `DIFF_AUDIT_MOONEP_2026-09-28.md`，今晚的 smoke 套件在 `kit_moonep_rewrite_2026-09-29/`（先看 `README.md`）。
- **本地验证（8 × 5060，没有 NVSwitch，用的是仿 MoonEP 公开版 API 的假包，结果只进 logbook）：** 见 `kit_moonep_rewrite_2026-09-29/LOCAL_CHECKS.md`。
- **标题建议：** `[MoE] MoonEP as an expert-parallel comm backend`
- **开 PR 的方式：** 我建议同步 `k3_moonep_seam` 以后，把 #4751 从 draft 转成 ready，这样保留原来的讨论。换这份 body 时，Results 用今晚的 H100 表替换。

--- PR 4751 body v2: PASTE BEGIN ---

## Summary

Add MoonEP (MoonshotAI/MoonEP, the expert-parallel transport of the Kimi K3 report), which keeps every rank's routed token count at `S x K` by prefetching copies of hot experts, as `moe_comm_backend="moonep"` for every model built with `make_routed_experts_config`.

- `MoonEPTokenDispatcher` (`models/common/token_dispatcher.py`, beside DeepEP and HybridEP): dispatch and combine through one process-global MoonEP `Buffer`; the transport and the NVLink pools live in `distributed/moonep/moonep.py`.
- `MoonEPRoutedExperts` (`models/common/moe.py`): `RoutedExperts` whose grouped GEMMs run over this rank's experts followed by the expert copies prefetched into its slots, and whose backward reduces the copies' gradients into their home experts.
- `make_token_dispatcher_config` and `make_routed_experts_config` (`models/common/config_utils.py`): `"moonep"` selects both classes; `update_ep_token_dispatcher_config` fills MoonEP's static per-rank token count the way it fills DeepEP's.
- `torchtitan_recipes/tests/h100.py`: `kimi_k3_moonep_fsdp4_ep4`, the Kimi K3 debug model at FSDP 4 x EP 4, in the h100 suite.

## Design

MoonEP moves expert weights as well as tokens. After dispatch it prefetches the experts its planner copies into slots on other ranks; each rank's grouped GEMMs cover its `E / R` experts followed by its `E / R` slots, with the planner's `cu_seqlens` as offsets; in backward the slots' gradients are reduced into their home experts. Weights and gradients live in NVLink pools of `2 E / R` rows per rank and projection, allocated once per process and shared by every MoE layer, as MoonEP intends. Each rank copies its unsharded expert weights into the first half of its rows before the prefetch, so the parameters stay ordinary tensors under FSDP. Gate and up get a pool each, since MoonEP's prefetch takes three contiguous projections and `w13` interleaves gate and up per expert.

A later layer overwrites the shared pools, so the expert forward keeps no graph: the backward refills the pools for its own plan, which also keeps interleaved pipeline schedules correct, and recomputes the expert GEMMs. Dispatch and combine are autograd functions whose backward is the other kernel on the same plan. Routing weights are applied in `combine`, as in the standard dispatcher, so the router trains through the same path. MoonEP's `reduce_grad` reads every rank's slot gradients without a barrier of its own, so the backward writes them and runs an all-reduce on the EP group before reducing.

MoonEP sizes its buffer for a static per-rank token count, so every dispatch carries exactly `num_max_tokens_per_rank` tokens, which `update_ep_token_dispatcher_config` derives from the training shape.

Requirements and costs:

- Hopper or newer behind an NVSwitch: MoonEP's buffers assert NVLink multicast.
- MoonEP at its public release (`33327eb`) with `nvidia-cutlass-dsl` 4.6.2, the version it pins; Attention Gym's KDA kernels run on the same version.
- The weight prefetch and the expert forward GEMMs run twice per micro-batch, since the backward refills the shared pools and recomputes; the standard path without activation checkpointing runs the GEMMs once.
- Tokens and expert weights enter MoonEP in bf16, the dtype its kernels take, so the routed experts compute in bf16 whatever the training's parameter dtype; the slot gradients are reduced in fp32.
- Dispatch and combine are not `torch.library` ops like DeepEP's, so model compile breaks the graph at each of them.
- One single-element all-reduce on the EP group per MoE layer backward, which orders the slot-gradient writes before MoonEP reads them.
- LoRA on the routed experts is not supported: its adapters cover this rank's experts, not the copies in its slots.

## Results

Pending (H100): the tokens each EP rank receives in every MoE layer, with standard EP and with MoonEP, under natural routing and under a router biased toward the experts of one rank, and the step time of each.

## Test plan

- `pytest tests/unit_tests/cpu/test_moe.py tests/unit_tests/cpu/test_integration_test_definitions.py -q`: the `moonep` backend builds the MoonEP experts and dispatcher, and the h100 suite registers the new cell.
- `pytest tests/unit_tests/gpu/test_moonep.py -q` (needs the `moonep` package and NVLink multicast): on two GPUs the MoonEP experts match a dense fp32 reference in output, input gradient and expert weight gradients, once with every token routed to one rank's experts, where tokens must reach the prefetch slots, and once with uniform routing.
- `python -m tests.integration_tests.run_tests <output_dir> --test_suite h100 --test_name "kimi_k3_fsdp+moonep" --ngpu 4`.
- Load and step time (pending, 4 H100s behind an NVSwitch): the Kimi K3 debug model with 128 experts and top-8 at FSDP 4 x EP 4, standard EP against MoonEP, natural routing and a router biased toward the experts of rank 0. Per MoE layer and micro-batch, the tokens each rank receives with static placement (max over mean) and, on MoonEP, whether every dispatch puts exactly S x K rows on each rank; the step time of each cell.

## Relation to earlier revisions of this PR

Earlier revisions subclassed `GroupedExperts` and allocated a buffer, pools and a weight table per MoE layer. Main has since replaced `GroupedExperts` with `RoutedExperts` owning `GroupedLinear` projections, and this revision is a rewrite on that structure, with one buffer and one set of pools per process.

--- PASTE END ---
