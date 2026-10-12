# PR #4576 body v2（LoRA 重做后，2026-10-12）

状态说明（不进 GitHub）：

- 分支：本地 worktree `scratchpad/wt_lora` = `b61c686dc`（main `b663c1673` 上三个提交 `8393252ad`、`91212bd8b`、`b61c686dc`，没有 trailer）。PR 头 `lora_review2` 还是 09-10 的 `72bbcb639`，推之前要用户点头，先备份。
- 范围变化：QLoRA（NF4、packed MXFP4 冻结底座）整个拿掉。MXFP4 用 Elfie 的 MX QAT（#4791），和 `LoRATransform` 能组合；NF4 不做。
- 标题建议：`[LoRA] Export merged adapters, place adapters for every base sharding, step frozen pipeline ranks`（原标题里的 "[DO NOT review, draft ...]" 是否去掉由用户决定）。
- GPU 结果按规则写 Pending (H100)。5060 上的 18 格矩阵、冻结 stage 格子和导出检查都在 `STACK_FULL_AND_LORA_MM18_2026-10-12.md`。
- 测试数是 `b61c686dc` 上实测的。

--- PASTE BEGIN ---

## Summary

With its default targets, LoRA on main fails to build every MoE, MLA and vision language debug model, has no way to export a trained adapter, and fails on a pipeline rank with nothing to train; this PR fixes all three.

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
  - Bases that TP does not shard (router gate, MLA `wkv_a` and `wq_a`, Kimi K3 `forget_a`) and vision encoder linears (`I` rather than `R` on CP) failed that assert at build: deepseek_v3, deepseek_v4, gpt_oss, qwen3_5, kimi_k2_5 and kimi_k3 could not build with the default targets.
  - The adapters now follow the TP entry of the base placement: `S(0)` and `S(1)` keep the column and row rules, `R` and `I` give both adapters the base placement, and any other placement raises.
  - The replicated adapter is the base placement with TP set to `R`, so its DP and CP entries are the base's; for a dense base this is `dense_param_placement(tp=R)` as before.
- Frozen pipeline ranks:
  - Targets that skip whole stages leave model parts with nothing to train; with MLA only targets, Kimi K3's KDA only stages are frozen, and a non interleaved schedule freezes whole ranks.
  - The optimizer container skips a frozen part with a log line; a trainable part whose pattern matches nothing still raises.
  - A rank with no trainable part gets an `Optimizer` with one empty param group and an LR scheduler container with no scheduler and no state.
  - `clip_grad_norm_` gives such a rank a zero norm on the accelerator for the pipeline all-reduce; it was a CPU `tensor(0.)`, which NCCL rejects.

## Relation to other PRs

- The earlier scope of this PR (NF4 and packed MXFP4 frozen bases) is dropped; MXFP4 for Kimi K3 is the MX QAT of #4791, which composes with `LoRATransform`.
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
- Kimi K3 18 cell parallelism matrix with LoRA, an export check on every rank, and MLA only targets under pp8 and pp4 x vpp2: Pending (H100).

--- PASTE END ---
