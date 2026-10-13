# PR #4576 body v2（LoRA 重做后，2026-10-12，10-13 改）

状态说明（不进 GitHub）：

- 分支：`lora_review2` = `fb4079831`，main `b663c1673` 上四个提交：`8393252ad`、`91212bd8b`、`b61c686dc`、`fb4079831`（pyrefly 修正，10-13 用户同意后快进推送），都没有 trailer。10-12 在 18 / 18 之后强推过一次，备份 `backup/lora_review2_pre_20261012` = `72bbcb639`。
- 栈分支上这四个提交压成一个 `b49021d5e`，源码改动和 PR 头逐字节相同，另外多一条只在栈上的 MX QAT 合并测试。
- 范围变化：QLoRA（NF4、packed MXFP4 冻结底座）整个拿掉。QLoRA 用 Elfie 的 MX QAT（#4791）加本 PR 的 LoRA 组合；NF4 不做。
- 标题建议：`[LoRA] Export merged adapters, place adapters for every base sharding, step frozen pipeline ranks`。原标题里的 "[DO NOT review, draft ...]" 是否去掉由用户决定。
- 10-13 用户："body修复这种confuse language"。粘贴区不再提没发布过的中间版本（"previous merge"、冻结格子在中间版本上的报错），Design、Relation、Results 里含糊的说法改成说清楚主语和条件。
- 10-13 用户："body篇幅太长了。。。"。粘贴区压缩：Design 每条一行，Results 并成一张表（GPU 数、第 1 步 loss 和参考的差、第 10 步 loss、合并检查），去掉 grad norm 和第 5 步，附注 3 条。
- 10-13 用户："5060这个LoRA 没有上游数值对比吧？body里面可以放5060的"，"为什么没有表格？乱写，参照4312"。Results 节按 4312 的格式重写：设定一行、kit 链接、复现命令、四张表（dp1 组、dp2 组、各自数据流、冻结 stage）和脚注，数字由脚本从 `b61c686dc` 的日志生成。以下是改表格之前的记录：粘贴区的 Test plan 加了 5060 上的 18 格（通过与否 + 每个 rank 的合并逐位检查，标明 8 x RTX 5060 Ti）和两个冻结 stage 格子，Relation 加了一句 MX QAT + LoRA 在叠栈分支上同样 18 格通过。这些都不和上游比数值。明细在 `STACK_FULL_AND_LORA_MM18_2026-10-12.md`。单测计数是 `fb4079831` 上实测的。
- 10-13 改了粘贴区三处：Summary 不再说 "every ... vision language model"（muse_glimmer 是 VLM，在 main 上能构建），改成列出建不起来的模型；Relation 写明 QLoRA 是 #4791 的 MX QAT 加本 PR 的 LoRA；删掉 Pending (H100) 那一行。

--- PASTE BEGIN ---

## Summary

With its default targets, LoRA on main fails to build the deepseek_v3, deepseek_v4, gpt_oss, qwen3_5, kimi_k2_5 and kimi_k3 debug models, cannot export a trained adapter, and fails on a pipeline rank with nothing to train; this PR fixes all three.

- `torchtitan/models/common/lora.py`: `trainable_state_dict`, `merge_lora_state_dict`, and adapter shardings that follow the TP entry of the base placement.
- `torchtitan/components/optim/optimizer.py`, `lr_scheduler.py`, `torchtitan/distributed/utils.py`: a rank whose model parts are all frozen gets an empty optimizer and scheduler and still joins the grad norm all-reduce.

## Design

- Export:
  - `trainable_state_dict` returns the trainable parameters keyed as in `state_dict()`; under LoRA, the adapters.
  - `merge_lora_state_dict` returns `state_dict()` with $W + \frac{\alpha}{r} B A$ in place of each adapted weight and no adapter keys.
  - The merge runs in fp32 on local shards and keeps the base's DTensor placements; only A is gathered, where FSDP shards its rank dim.
  - It works on local shards because DTensor ops cannot unflatten the `_StridedShard` placement of a stacked `w13` under FSDP and TP.
- Adapter placement:
  - On main, `_adapter_sharding` asserts that every base is `dense_param_placement(tp=S(0))` or `dense_param_placement(tp=S(1))`; bases that TP does not shard (router gate, MLA `wkv_a`, `forget_a`) and vision linears (`I` on CP) fail that assert when the model is built.
  - The TP entry now decides: `S(0)` and `S(1)` keep the column and row rules, `R` and `I` give both adapters the base placement, anything else raises.
  - The replicated adapter takes the base placement with TP set to `R`; for a dense base this is the placement main already uses.
- Frozen pipeline ranks:
  - A model part with nothing to train is skipped with a log line; a trainable part whose pattern matches nothing still raises.
  - A rank with nothing to train gets one empty param group, no scheduler state, and a zero grad norm on the accelerator; on main that rank's norm is a CPU `tensor(0.)`, which the NCCL all-reduce rejects.

## Relation to other PRs

- This PR no longer adds the NF4 and packed MXFP4 bases of its first version; QLoRA for Kimi K3 is the MX QAT of #4791 with this PR's LoRA applied after it.
- #3189 and #2485 (PEFT export) closed unmerged; a converter can start from these two state dicts.

## Results

8 x RTX 5060 Ti, `kimi_k3_debugmodel` (17 layers, dim 256) on the multimodal `cc12m-test`, 4096 tokens per step, seed 42, deterministic. `LoRATransform(rank=8, alpha=16.0)` adapts all 290 supported projections. Every cell loads one seed checkpoint, trains 10 steps, then checks on every rank that `merge_lora_state_dict` equals $W + \frac{\alpha}{r} B A$ on the gathered tensors. Measured at `b61c686dc`; the last commit, `fb4079831`, only adds a type assertion. The cell configs, the merge check and the runners are in [this kit](https://github.com/QIU023/torchtitan_attention_residual/tree/673a0323/Raising_PRs/PR_K3_PARALLELISM/kit_lora_mm18_2026-10-12), not in this PR.

Step 1 matches its reference in every layout, identical without TP or CP and within 0.005% with them, and the merge is exact on every rank.

| cell | GPUs | loss, step 1 | step 10 | merge |
| --- | ---: | ---: | ---: | --- |
| dp1 [1] | 1 | `8.13623` | `7.09263` | exact |
| pp2 | 2 | `8.13623`<br>identical | `7.09142` | exact |
| pp4 | 4 | `8.13623`<br>identical | `7.09344` | exact |
| pp8 | 8 | `8.13623`<br>identical | `7.09142` | exact |
| tp2 x ep2 | 2 | `8.13642`<br>+0.0023% | `7.09488` | exact |
| tp4 x ep4 | 4 | `8.13640`<br>+0.0021% | `7.09517` | exact |
| cp2 | 2 | `8.13626`<br>+0.0004% | `7.09520` | exact |
| ep2 x tp2 x pp2 x cp2 | 8 | `8.13628`<br>+0.0006% | `7.09160` | exact |
| fsdp2 [1] | 2 | `8.18727` | `7.17045` | exact |
| ep2 x fsdp2 | 2 | `8.18727`<br>identical | `7.17250` | exact |
| ep4 x fsdp2 x tp2 x pp2 | 8 | `8.18761`<br>+0.0042% | `7.17034` | exact |
| ep2 x fsdp2 x tp2 x pp2 | 8 | `8.18761`<br>+0.0042% | `7.17220` | exact |
| fsdp2 x pp2 x cp2 | 8 | `8.18745`<br>+0.0022% | `7.17252` | exact |
| ep2 x fsdp2 x pp2 x cp2 | 8 | `8.18745`<br>+0.0022% | `7.17073` | exact |
| ep4 x fsdp2 x tp2 x cp2 | 8 | `8.18752`<br>+0.0031% | `7.17260` | exact |
| ep2 x fsdp2 x tp2 x cp2 | 8 | `8.18752`<br>+0.0031% | `7.17260` | exact |
| ep8 x fsdp8 [1] | 8 | `8.16220` | `7.07305` | exact |
| cp4 [1] | 4 | `8.14816` | `7.34267` | exact |
| pp4 x vpp2, MLA only [2] | 4 | `8.10062` | `7.97847` | exact |
| pp8, MLA only [2] | 8 | `8.10062`<br>identical | `7.97848` | exact |

- [1] reference row: dp1, fsdp2, ep8 x fsdp8 and cp4 (512 token sequences) each read a different sample stream; the rows under a reference read the same samples as it
- [2] `target_modules` = the MLA projections, so the stages that hold only KDA layers have nothing to train: pp4 x vpp2 puts a frozen stage next to a trainable one on three ranks, and pp8 leaves three ranks with nothing to train; both layouts rely on this PR's optimizer, scheduler and grad norm changes
- step 10 is shown, not compared: each GPU group compiled on its own cache, and the 32 sample set repeats within 10 steps

## Test plan

```
pytest tests/unit_tests/cpu/models/common/test_lora.py                          # 39 passed (10 new)
pytest tests/unit_tests/cpu/components/optim/test_optimizer_param_groups.py     # 32 passed (2 new)
pytest tests/unit_tests/cpu/components/optim/test_lr_scheduler.py               # 10 passed (1 new)
pytest tests/unit_tests/gpu/models/common/test_lora.py                          # 1 passed, 4 GPUs
pytest tests/unit_tests/gpu/distributed/test_clip_grad_norm.py                  # 1 passed, 2 GPUs
```

The GPU merge test covers a stacked `w13` under TP2 x FSDP2; the clip test raises `No backend type associated with device type cpu` on main. `pyrefly check` reports the same errors as main.

--- PASTE END ---
