# PR #4576 body v2（LoRA 重做后，2026-10-12，10-13 改）

状态说明（不进 GitHub）：

- 分支：`lora_review2` = `fb4079831`，main `b663c1673` 上四个提交：`8393252ad`、`91212bd8b`、`b61c686dc`、`fb4079831`（pyrefly 修正，10-13 用户同意后快进推送），都没有 trailer。10-12 在 18 / 18 之后强推过一次，备份 `backup/lora_review2_pre_20261012` = `72bbcb639`。
- 栈分支上这四个提交压成一个 `b49021d5e`，源码改动和 PR 头逐字节相同，另外多一条只在栈上的 MX QAT 合并测试。
- 范围变化：QLoRA（NF4、packed MXFP4 冻结底座）整个拿掉。QLoRA 用 Elfie 的 MX QAT（#4791）加本 PR 的 LoRA 组合；NF4 不做。
- 标题建议：`[LoRA] Export merged adapters, place adapters for every base sharding, step frozen pipeline ranks`。原标题里的 "[DO NOT review, draft ...]" 是否去掉由用户决定。
- 10-13 用户："5060这个LoRA 没有上游数值对比吧？body里面可以放5060的"。粘贴区的 Test plan 加了 5060 上的 18 格（通过与否 + 每个 rank 的合并逐位检查，标明 8 x RTX 5060 Ti）和两个冻结 stage 格子，Relation 加了一句 MX QAT + LoRA 在叠栈分支上同样 18 格通过。这些都不和上游比数值。明细在 `STACK_FULL_AND_LORA_MM18_2026-10-12.md`。单测计数是 `fb4079831` 上实测的。
- 10-13 改了粘贴区三处：Summary 不再说 "every ... vision language model"（muse_glimmer 是 VLM，在 main 上能构建），改成列出建不起来的模型；Relation 写明 QLoRA 是 #4791 的 MX QAT 加本 PR 的 LoRA；删掉 Pending (H100) 那一行。

--- PASTE BEGIN ---

## Summary

With its default targets, LoRA on main fails to build the deepseek_v3, deepseek_v4, gpt_oss, qwen3_5, kimi_k2_5 and kimi_k3 debug models, has no way to export a trained adapter, and fails on a pipeline rank with nothing to train; this PR fixes all three.

- `torchtitan/models/common/lora.py`: `trainable_state_dict` and `merge_lora_state_dict`; adapter shardings derived from the TP entry of the base placement.
- `torchtitan/components/optim/optimizer.py`, `torchtitan/components/optim/lr_scheduler.py`, `torchtitan/distributed/utils.py`: a pipeline rank whose model parts are all frozen gets an empty optimizer and scheduler and still joins the grad norm all-reduce.

## Design

- Export:
  - `trainable_state_dict(model)` returns the trainable parameters keyed as in `state_dict()`; under LoRA, the adapters.
  - `merge_lora_state_dict(model)` returns `state_dict()` with each adapted weight replaced by $W + \frac{\alpha}{r} B A$ and the adapter keys removed, so a state dict adapter converts it like the unadapted model.
  - The merge runs in fp32 on each rank's local shards and returns DTensors with the base's placements; no base weight is gathered.
  - B is sharded like the base's output rows, so its local shard lines up with the base's; A is gathered only where FSDP shards its rank dim.
  - DTensor ops on the sharded tensors fail on a stacked `w13` under FSDP and TP, whose `_StridedShard` placement cannot be unflattened.
- Adapter placement:
  - `_adapter_sharding` asserted that every base other than `dense_param_placement(tp=S(0))` is `dense_param_placement(tp=S(1))`.
  - Two kinds of base failed that assert at build: bases that TP does not shard (router gate, MLA `wkv_a` and `wq_a`, Kimi K3 `forget_a`) and vision encoder linears, which are `I` rather than `R` on CP.
  - The adapters now follow the TP entry of the base placement: `S(0)` and `S(1)` keep the column and row rules, `R` and `I` give both adapters the base placement, and any other placement raises.
  - The replicated adapter is the base placement with TP set to `R`, so its DP and CP entries are the base's; for a dense base this is `dense_param_placement(tp=R)` as before.
- Frozen pipeline ranks:
  - Targets that skip whole stages leave model parts with nothing to train; with MLA only targets, Kimi K3's KDA only stages are frozen, and a non interleaved schedule freezes whole ranks.
  - The optimizer container skips a frozen part with a log line; a trainable part whose pattern matches nothing still raises.
  - A rank with no trainable part gets an `Optimizer` with one empty param group and an LR scheduler container with no scheduler and no state.
  - `clip_grad_norm_` gives such a rank a zero norm on the accelerator for the pipeline all-reduce; it was a CPU `tensor(0.)`, which NCCL rejects.

## Relation to other PRs

- The earlier scope of this PR (NF4 and packed MXFP4 frozen bases) is dropped. QLoRA for Kimi K3 is the MX QAT of #4791 with `LoRATransform` applied after it, which wraps the QAT linears and freezes their weights; on a branch that stacks both, that combination passes the same 18 layouts.
- #3189 and #2485 (PEFT format export) were closed without merging; the two state dicts here are what such a converter starts from.

## Test plan

```
pytest tests/unit_tests/cpu/models/common/test_lora.py                          # 39 passed (10 new)
pytest tests/unit_tests/cpu/components/optim/test_optimizer_param_groups.py     # 32 passed (2 new)
pytest tests/unit_tests/cpu/components/optim/test_lr_scheduler.py               # 10 passed (1 new)
pytest tests/unit_tests/gpu/models/common/test_lora.py                          # 1 passed, 4 GPUs
pytest tests/unit_tests/gpu/distributed/test_clip_grad_norm.py                  # 1 passed, 2 GPUs
```

- The GPU merge test (TP2 x FSDP2) fails on the previous merge with the `_StridedShard` error; the clip test fails on main with `No backend type associated with device type cpu`.
- `pyrefly check` reports the same errors as main.
- Kimi K3 debug model with `LoRATransform(rank=8, alpha=16.0)` on all 290 supported projections, on 8 x RTX 5060 Ti: 18 / 18 parallelism layouts pass; on main the model does not build.
  - Layouts: dp1, FSDP2, PP2, PP4, PP8, TP2 x EP2, TP4 x EP4, CP2, CP4, EP2 x FSDP2, EP8 x FSDP8, EP4 x FSDP2 x TP2 x PP2, EP2 x FSDP2 x TP2 x PP2, FSDP2 x PP2 x CP2, EP2 x FSDP2 x PP2 x CP2, EP4 x FSDP2 x TP2 x CP2, EP2 x TP2 x PP2 x CP2, EP2 x FSDP2 x TP2 x CP2.
  - Each layout loads one seed checkpoint and trains 10 steps; then on every rank `merge_lora_state_dict` equals $W + \frac{\alpha}{r} B A$ computed on the gathered tensors, bit for bit, and `trainable_state_dict` is exactly the adapters.
- Frozen pipeline ranks, with LoRA on the MLA projections only: PP4 x VPP2 (three ranks hold a frozen stage next to a trainable one) and PP8 (three ranks fully frozen) train 10 steps and pass the same merge check. Without the optimizer commit, PP4 x VPP2 raises `matched no parameters`; PP8 also needs the empty optimizer, scheduler and grad norm.

--- PASTE END ---
