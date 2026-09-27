# PR 4381 body v4（draft PR，`k3_pp_mm`），2026-09-27：4312 合并后 rebase 到 main

## 状态（不粘贴）

- **分支：** review 分支 `dep_review1` = `31f372593`，是 main `f35966713` 上的 4 个提交，已推到 fork。PR 分支 `k3_pp_mm` 还是 `232834a4d`（旧的 4312 base）。
- **rebase 里改了什么：**
  - `model.py` 冲突：main 的 #4777（修多模态 FSDP 卡死）让没有图像的 micro-batch 也造一张假图跑视觉塔，再用零依赖接回文本；DEP 让 forward 接收预先编码好的 `vision_embeds`。两边都保留：先用 `vision_embeds`，没有才就地编码，然后走 main 的假图分支。
  - **有没有语义冲突（已实测）：** 我先以为 DEP 的缓存跳过没有图像的 micro-batch，会让视觉塔的 FSDP all-gather 在有图、没图的 DP rank 上顺序错开，于是加过一次"在放置点编码假图"，后来整个撤掉了。
    - 用 #4777 自带的 `set_rank_conditional_image_presence`（偶数 DP rank 只有文本），在 dp2 × pp4 的 vit_dep 格子上跑 4 步：不加假图 rc=0，4 步都跑完；我加的版本反而在第 2 步崩了（缓存构造时记下的设备是 meta）。
    - 原因（代码核实）：开 PP 时 `MultimodalModel._apply_fsdp` 不单独包视觉塔，它的参数在第一个 stage 的根 FSDP 单元里，`tok_embeddings` 是另一个单元；PP 下默认 `reshard_after_forward=False`，两个单元每步各 all-gather 一次；提前编码按 `modules()` 顺序 unshard（先根、后 `tok_embeddings`），和没图的 rank 在 forward 里的顺序相同。所以不需要额外处理。
    - `fsdp_reshard_after_forward=always`（每个 micro-batch 都 all-gather）同样 4 步跑完，rc=0，没有超时。
  - #4617 把 `ParallelismConfig` 挪到了 `torchtitan.config.parallelism`，DEP 的测试改了导入。
  - 注释按"默认不加注释"规则修剪：14 处多行注释改成一行约束，或删掉（讲设计怎么选的那些）；starved/exhausted 的含义挪进字段 docstring。
- **标题：** 线上是 `[DO NOT review, pending K3 text PP merging] [Kimi K3] Add K3 MoonViT DEP support to schedule ViT stages into text LLM PP bubbles`。4312 已合并，前缀可以去掉，由你改。
- **没有重测的：** 测试计划最后一条（bubble 开和关的对比）还是 09-22 旧 base 的结论，要在新 head 上用同一份暖缓存重跑。
- **线上 body：** 之前用 API 读线上 body 被权限拦过，粘贴前请对照一下。

--- PR 4381 body v4: PASTE BEGIN ---

### Summary

Report sec 5.2.3: the vision tower takes a pipeline stage of its own ahead of the text stages, so its compute leaves the critical path of the stage that owns the embedding, and its encodes are placed around the schedule's own actions: ahead of the forward that reads them (`vision_dep.prefetch`), or in the schedule's idle intervals with the tower's backwards deferred to the intervals after backward actions (`vision_dep.bubble`).

### Design

- The tower stage is the split alone, no core change. `vit_dep_split` puts `[tok_embeddings, vision_encoder]` on the first stage and spreads the layers over the remaining stages through core's own `_generate_llm_fqn_per_model_part`, so the stage count the schedule sees is unchanged and the tower stage comes out of the text stages' budget. The embedding rides with the tower because the splice needs the token ids, which only the first stage receives. The block routing accepts a stage with no layers.
- The knobs are a `vision_dep` record on the model config: `enabled`, `prefetch`, `bubble`, `bubble_cost_ratio`, `bubble_max_pending`. They change the split every rank applies, so they belong to the model rather than to the command line, and the model config tree is off the CLI.
- `encode_images` is the tower's forward on one micro-batch's images, and `forward` takes `vision_embeds`, so the schedule's forward and the ahead-of-time encode cannot drift. An encode issued before the pipeline calls the stage gathers the FSDP parameters itself; the stage's own forward reshards them as its policy says. The first step encodes inline, because FSDP2 builds its state in the root module's first forward and an encode before that makes the tower a root of its own.
- `VisionDepPipelineStage` subclasses the AttnRes stage and hands the runtime each action as it completes, which is how a placement fires after the action it is anchored to. An anchor is an action's identity rather than a slot index, because the index does not survive lowering: the runtime walks `pipeline_order_with_comms`, which inserts sends and receives and holds no idle entries.
- The plan (`dep_bubble_plan.py`, pure) is read off `pp_schedule.pipeline_order[rank]`, so every rank derives the same placements with no collective and none reaches a vision collective the others do not. A micro-batch that carries no images is not encoded ahead; its forward takes the inline path, which runs the tower on the dummy image #4777 adds. With pipeline parallelism the tower's parameters sit in the first stage's root FSDP unit, which is gathered once per step, and the ahead-of-time encode unshards the stage's units in the order its forward does, so a data-parallel rank without images issues the same gathers in the same order. The first `pp` encodes run upfront, as the report prescribes; later ones go in the last idle slot whose accumulated budget covers `bubble_cost_ratio` text-stage actions, so the features stay resident as briefly as the budget allows.
- The tower's backward is cut at the splice (`dep_backward.py`): the features are spliced in through a detached stand-in whose gradient is captured and replayed at a later slot. Whatever the slots did not take is drained at step end, because a deferred backward that never runs leaves the tower without that micro-batch's gradient and raises nothing.
- `OptimizersContainer` skips a param-group pattern that matches nothing on one model part and refuses only a pattern that no model part on the rank matches. The tower stage holds none of the matrices the per-head DistMuon recipe (#4596) assigns to Muon, and before this change such a stage raised.

### Test plan

- `pytest tests/unit_tests/cpu -k "kimi_k3 or pipeline_parallel or cli or integration_test" -q` (120 passed, 1 skipped). The bubble tests cover the placement invariants, a backward-anchored placement firing, the run-ahead, and the deferred backward: a cut and replayed tower backward is bitwise the gradient the inline one produces, out-of-order replays accumulate like one pass, the pending bound changes when rather than whether a gradient runs, and nothing is lost when no slot ever comes.
- `pytest tests/unit_tests/cpu/test_optimizer_param_groups.py -q` (24 passed), including a pattern that matches nothing on one stage.
- The `kimi_k3_pp4_vp2_vit_dep` cell in the B200 suite, which derives its split from `vision_dep` rather than spelling one out: 10 steps on 4 x RTX 5060 Ti, loss 7.96843 to 6.13097 (no fixed seed, so only that it runs); every step places 2 of 2 planned encodes in a bubble, runs 4 upfront and 2 inline, and runs all 6 deferred tower backwards at planned slots.
- The same cell at dp_shard 2 with the even data-parallel ranks text only (`set_rank_conditional_image_presence`), 4 steps on 8 x RTX 5060 Ti: every step completes, under the default reshard policy and with `fsdp_reshard_after_forward=always`.
- pp4 x vp2 with the bubble on and off, same seed and batch on one warm compile cache: the tower's forward is identical either way; the losses separate from step 3 by the order the tower's gradients accumulate in. The hundred-step table on H100 goes here.

--- PASTE END ---
