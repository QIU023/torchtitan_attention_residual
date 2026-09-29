# PR 4751 body v2（MoonEP 重写，`moonep_review1`），2026-09-29

## 状态（不粘贴）

- **用户 09-28：** "直接重写 备份一下当前moonep tree 推到我晚上提供4 h100的时候能直接smoke然后打开正式pr的程度"。
- **备份：** 旧实现 `f556ab4fd` 在 fork 的 `backup/k3_moonep_seam_pre_rewrite_20260928`，本地 tag 同名。PR 分支 `k3_moonep_seam` 还是 `f556ab4fd`，今晚 smoke 通过、你说同步以后再推。
- **分支：** review 分支 `moonep_review1` = `a505f74a8`，main `5dc97a3e7` 上的 2 个提交（代码、测试），10 个文件 +605/−3：生产代码 +428（旧实现 +595），测试 +177（旧实现 +477），Kimi K3 模型目录 0 行（旧实现 47 行）。没有 trailer。审计见 `DIFF_AUDIT_MOONEP_2026-09-28.md`，今晚的 smoke 套件在 `kit_moonep_rewrite_2026-09-29/`（先看 `README.md`）。
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

A later layer overwrites the shared pools, so the expert forward keeps no graph: the backward refills the pools for its own plan, which also keeps interleaved pipeline schedules correct, and recomputes the expert GEMMs. Dispatch and combine are autograd functions whose backward is the other kernel on the same plan. Routing weights are applied in `combine`, as in the standard dispatcher, so the router trains through the same path.

MoonEP sizes its buffer for a static per-rank token count, so every dispatch carries exactly `num_max_tokens_per_rank` tokens, which `update_ep_token_dispatcher_config` derives from the training shape.

Requirements and costs:

- Hopper or newer behind an NVSwitch: MoonEP's buffers assert NVLink multicast.
- MoonEP at its public release (`33327eb`) with `nvidia-cutlass-dsl` 4.6.2, the version it pins; Attention Gym's KDA kernels run on the same version.
- The expert forward GEMMs run twice per micro-batch, once more than the standard path without activation checkpointing.
- Dispatch and combine are not `torch.library` ops like DeepEP's, so model compile breaks the graph at each of them.

## Results

Pending (H100).

## Test plan

- `pytest tests/unit_tests/cpu/test_moe.py tests/unit_tests/cpu/test_integration_test_definitions.py -q`: the `moonep` backend builds the MoonEP experts and dispatcher, and the h100 suite registers the new cell.
- `pytest tests/unit_tests/gpu/test_moonep.py -q` (needs the `moonep` package and NVLink multicast): on two GPUs the MoonEP experts match a dense fp32 reference in output, input gradient and expert weight gradients, once with every token routed to one rank's experts, where tokens must reach the prefetch slots, and once with uniform routing.
- `python -m tests.integration_tests.run_tests <output_dir> --test_suite h100 --test_name "kimi_k3_fsdp+moonep" --ngpu 4`.

## Relation to earlier revisions of this PR

Earlier revisions subclassed `GroupedExperts` and allocated a buffer, pools and a weight table per MoE layer. Main has since replaced `GroupedExperts` with `RoutedExperts` owning `GroupedLinear` projections, and this revision is a rewrite on that structure, with one buffer and one set of pools per process.

--- PASTE END ---
