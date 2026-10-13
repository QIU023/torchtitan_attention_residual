# PR #4576 body v2（LoRA 重做后，2026-10-12，10-13 改）

状态说明（不进 GitHub）：

- 分支：`lora_review2` = `fb4079831`，main `b663c1673` 上四个提交：`8393252ad`、`91212bd8b`、`b61c686dc`、`fb4079831`（pyrefly 修正，10-13 用户同意后快进推送），都没有 trailer。10-12 在 18 / 18 之后强推过一次，备份 `backup/lora_review2_pre_20261012` = `72bbcb639`。
- 栈分支上这四个提交压成一个 `b49021d5e`，源码改动和 PR 头逐字节相同，另外多一条只在栈上的 MX QAT 合并测试。
- 范围变化：QLoRA（NF4、packed MXFP4 冻结底座）整个拿掉。QLoRA 用 Elfie 的 MX QAT（#4791）加本 PR 的 LoRA 组合；NF4 不做。
- 标题建议：`[LoRA] Export merged adapters, place adapters for every base sharding, step frozen pipeline ranks`。原标题里的 "[DO NOT review, draft ...]" 是否去掉由用户决定。
- 10-13 用户："5060这个LoRA 没有上游数值对比吧？body里面可以放5060的"，"为什么没有表格？乱写，参照4312"。Results 节按 4312 的格式重写：设定一行、kit 链接、复现命令、四张表（dp1 组、dp2 组、各自数据流、冻结 stage）和脚注，数字由脚本从 `b61c686dc` 的日志生成。以下是改表格之前的记录：粘贴区的 Test plan 加了 5060 上的 18 格（通过与否 + 每个 rank 的合并逐位检查，标明 8 x RTX 5060 Ti）和两个冻结 stage 格子，Relation 加了一句 MX QAT + LoRA 在叠栈分支上同样 18 格通过。这些都不和上游比数值。明细在 `STACK_FULL_AND_LORA_MM18_2026-10-12.md`。单测计数是 `fb4079831` 上实测的。
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

## Results

8 x RTX 5060 Ti, `kimi_k3_debugmodel` (17 layers, dim 256, a 2 layer vision tower) on the multimodal `cc12m-test` set, 256 token sequences (512 for cp4), 4096 tokens per step, 8 pipeline micro-batches, AdamW, seed 42, deterministic. `LoRATransform(rank=8, alpha=16.0)` with its default targets adapts all 290 supported projections; on main this model does not build with it. Every cell loads one seed checkpoint and trains 10 steps; `cc12m-test` is 32 samples and repeats within the run, so the tables stop at step 10. Each cell gives the raw value and, beneath it, the change against its reference. Measured at `b61c686dc` on main `b663c1673` (`fb4079831` after it adds a type assertion only).

Step 1 is identical to the reference in every cell without TP or CP and within 0.005% with them, and the merge is exact on every rank of every cell.

None of the following is part of this PR: the cell configs [mm18_probe.py](https://github.com/QIU023/torchtitan_attention_residual/blob/673a0323/Raising_PRs/PR_K3_PARALLELISM/kit_lora_mm18_2026-10-12/mm18_probe.py), the training entry point with the merge check [mm18_train_export.py](https://github.com/QIU023/torchtitan_attention_residual/blob/673a0323/Raising_PRs/PR_K3_PARALLELISM/kit_lora_mm18_2026-10-12/mm18_train_export.py), and the runners [run_mm18.sh](https://github.com/QIU023/torchtitan_attention_residual/blob/673a0323/Raising_PRs/PR_K3_PARALLELISM/kit_lora_mm18_2026-10-12/run_mm18.sh) and [run_frozen.sh](https://github.com/QIU023/torchtitan_attention_residual/blob/673a0323/Raising_PRs/PR_K3_PARALLELISM/kit_lora_mm18_2026-10-12/run_frozen.sh).

```bash
ENTRYPOINT=mm18_train_export bash run_mm18.sh lora $TREE $OUT   # the seed checkpoint, then the 18 cells
ENTRYPOINT=mm18_train_export bash run_frozen.sh $TREE $OUT_MLA $OUT_MLA/seed "pp4_vpp2|pp=4,vpp=2|0,1,2,3" "pp8|pp=8|0,1,2,3,4,5,6,7"
```

Cells that read the single GPU sample stream.

| cell | loss, step 1 | step 5 | step 10 | grad norm, step 1 | step 5 | step 10 | merge check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dp1 [1] | `8.13623` | `7.71449` | `7.09263` | `0.4277` | `0.5820` | `0.9062` | exact on 1 rank |
| pp2 | `8.13623`<br>identical | `7.71372`<br>-0.01% | `7.09142`<br>-0.02% | `0.4277`<br>identical | `0.5859`<br>+0.67% | `0.9062`<br>identical | exact on 2 ranks |
| pp4 | `8.13623`<br>identical | `7.71539`<br>+0.01% | `7.09344`<br>+0.01% | `0.4277`<br>identical | `0.5859`<br>+0.67% | `0.9062`<br>identical | exact on 4 ranks |
| pp8 | `8.13623`<br>identical | `7.71372`<br>-0.01% | `7.09142`<br>-0.02% | `0.4297`<br>+0.47% | `0.5859`<br>+0.67% | `0.9062`<br>identical | exact on 8 ranks |
| tp2 x ep2 | `8.13642`<br>+0.00% | `7.71442`<br>-0.00% | `7.09488`<br>+0.03% | `0.4297`<br>+0.47% | `0.5859`<br>+0.67% | `0.9023`<br>-0.43% | exact on 2 ranks |
| tp4 x ep4 | `8.13640`<br>+0.00% | `7.71522`<br>+0.01% | `7.09517`<br>+0.04% | `0.4277`<br>identical | `0.5859`<br>+0.67% | `0.9023`<br>-0.43% | exact on 4 ranks |
| cp2 | `8.13626`<br>+0.00% | `7.71342`<br>-0.01% | `7.09520`<br>+0.04% | `0.4277`<br>identical | `0.5820`<br>identical | `0.9023`<br>-0.43% | exact on 2 ranks |
| ep2 x tp2 x pp2 x cp2 | `8.13628`<br>+0.00% | `7.71480`<br>+0.00% | `7.09160`<br>-0.01% | `0.4258`<br>-0.44% | `0.5859`<br>+0.67% | `0.9023`<br>-0.43% | exact on 8 ranks |

Two data parallel ranks.

| cell | loss, step 1 | step 5 | step 10 | grad norm, step 1 | step 5 | step 10 | merge check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fsdp2 [1] | `8.18727` | `7.76231` | `7.17045` | `0.4316` | `0.6406` | `0.8516` | exact on 2 ranks |
| ep2 x fsdp2 | `8.18727`<br>identical | `7.76248`<br>+0.00% | `7.17250`<br>+0.03% | `0.4316`<br>identical | `0.6445`<br>+0.61% | `0.8516`<br>identical | exact on 2 ranks |
| ep4 x fsdp2 x tp2 x pp2 | `8.18761`<br>+0.00% | `7.76153`<br>-0.01% | `7.17034`<br>-0.00% | `0.4297`<br>-0.44% | `0.6445`<br>+0.61% | `0.8516`<br>identical | exact on 8 ranks |
| ep2 x fsdp2 x tp2 x pp2 | `8.18761`<br>+0.00% | `7.76142`<br>-0.01% | `7.17220`<br>+0.02% | `0.4297`<br>-0.44% | `0.6445`<br>+0.61% | `0.8516`<br>identical | exact on 8 ranks |
| fsdp2 x pp2 x cp2 | `8.18745`<br>+0.00% | `7.76141`<br>-0.01% | `7.17252`<br>+0.03% | `0.4316`<br>identical | `0.6445`<br>+0.61% | `0.8516`<br>identical | exact on 8 ranks |
| ep2 x fsdp2 x pp2 x cp2 | `8.18745`<br>+0.00% | `7.76129`<br>-0.01% | `7.17073`<br>+0.00% | `0.4297`<br>-0.44% | `0.6445`<br>+0.61% | `0.8516`<br>identical | exact on 8 ranks |
| ep4 x fsdp2 x tp2 x cp2 | `8.18752`<br>+0.00% | `7.76255`<br>+0.00% | `7.17260`<br>+0.03% | `0.4297`<br>-0.44% | `0.6406`<br>identical | `0.8516`<br>identical | exact on 8 ranks |
| ep2 x fsdp2 x tp2 x cp2 | `8.18752`<br>+0.00% | `7.76157`<br>-0.01% | `7.17260`<br>+0.03% | `0.4297`<br>-0.44% | `0.6406`<br>identical | `0.8516`<br>identical | exact on 8 ranks |

Their own sample streams.

| cell | loss, step 1 | step 5 | step 10 | grad norm, step 1 | step 5 | step 10 | merge check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ep8 x fsdp8 [2] | `8.16220` | `7.73987` | `7.07305` | `0.4609` | `0.5664` | `0.8906` | exact on 8 ranks |
| cp4 [3] | `8.14816` | `7.80357` | `7.34267` | `0.4395` | `0.6523` | `0.8047` | exact on 4 ranks |

Frozen pipeline ranks: LoRA on the MLA projections only.

| cell | loss, step 1 | step 5 | step 10 | grad norm, step 1 | step 5 | step 10 | merge check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pp4 x vpp2 [4] | `8.10062` | `8.04503` | `7.97847` | `0.1309` | `0.2451` | `0.5039` | exact on 4 ranks |
| pp8 [4] | `8.10062`<br>identical | `8.04471`<br>-0.00% | `7.97848`<br>+0.00% | `0.1309`<br>identical | `0.2451`<br>identical | `0.5039`<br>identical | exact on 8 ranks |

- [1] reference: dp1 for the cells that read its sample stream, fsdp2 for the two data parallel ranks' stream
- [2] eight data parallel ranks, a sample stream of its own
- [3] 512 token sequences, since CP flex attention needs the sequence divisible by cp x 128; a sample stream of its own
- [4] `target_modules` = the five MLA projections (25 adapted), so the stages that hold only KDA layers have nothing to train: pp4 x vpp2 puts a frozen stage next to a trainable one on ranks 0, 1 and 3, pp8 freezes ranks 0, 3 and 5 entirely. Without the optimizer commit, pp4 x vpp2 raises `Optimizer pattern '.*' matched no parameters`; with only the part skip, pp8 raises `optimizer got an empty parameter list`, and with the optimizer and scheduler handled but not the grad norm, `No backend type associated with device type cpu`
- merge check: after step 10, on every rank, `merge_lora_state_dict` equals $W + \frac{\alpha}{r} B A$ computed on the gathered tensors, bit for bit, for every adapted projection, and `trainable_state_dict` is exactly the adapters
- the grad norm is main's, taken in bf16, where one unit near 0.43 is 0.0020 (0.4277 against 0.4297); the pipeline and TP cells reduce it in another order
- each GPU group (one, two, four and eight GPUs) compiled on its own cache, so step 1 is the comparison and the later steps are shown only

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

--- PASTE END ---
