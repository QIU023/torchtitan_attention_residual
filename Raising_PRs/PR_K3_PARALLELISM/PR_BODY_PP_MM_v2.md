# PR title: [DO NOT review, stack on PR 4312] [Kimi K3] DEP -- the vision tower takes a pipeline stage of its own

Draft stacked on PR 4312. Review branch `dep_review1` = `191b31bc0`, three commits on 4312's round-3 head `78be13c96`, which is five typed commits on upstream main `7349a2282`. The fork branch `k3_pp_mm` was force-pushed to this head on 2026-09-22; its previous head `384d576dc` could not be rebased at all, because four of its modules import `torchtitan.tools.logging`, which the new base does not have, and the split helper it called was removed with the per-model `parallelize.py` in #4810. It was rebuilt from the integration tree's ported version instead of merged.

What changed since `384d576dc`, from the 2026-09-22 review of the draft branches:

- Nothing installs itself by replacing a method on a live object any more. `VisionDepPipelineStage` subclasses the branch's own `AttnResPipelineStage` and overrides `forward_one_chunk`, `backward_one_chunk` and `backward_weight_one_chunk`; the marker attributes written onto torch's stages are gone. One instance wrap is left, on `pp_schedule.step`, which is where the per-micro-batch kwargs arrive and which core picks the class of; the previous head wrapped `step` three times with the ordering held together by a comment.
- The prefetch side stream is gone. `_vision_stream` returned None whenever gradients were enabled, so in training the overlap three docstrings described never existed; `model.py` now carries only `encode_images` and the `vision_embeds` argument.
- Placements anchored on a backward action fire. The runtime used to drop them while the plan still counted them, so a schedule whose idle slots follow backwards warned every step and never fired; an anchor the rank cannot run at all is counted and logged once instead.
- `build_plans` and `_FakeStage`, 61 production lines that existed only for tests, moved into the test that uses them.
- One split helper, built on core's `_get_pipeline_metadata` and `_generate_llm_fqn_per_model_part`, replaces the two copies in the module and the third in the recipe. The B200 cell now reaches it: the recipe sets no split, which is the code the cell exists to run.
- The knobs are a `vision_dep` record on `KimiK3Model.Config`, because after #4810 the model owns its pipelining and there is no `pipelining_fn` for a recipe to wrap. Nothing in the previous head could turn the feature on.
- Comment plus docstring is 142 of 677 added production non-blank lines, 21.0 percent, from 419 of 1084, 38.7 percent.

CPU on this head: 96 passed (`pytest tests/unit_tests/cpu -k "kimi_k3 or pipeline or cli or integration_test"`), pyrefly clean on `torchtitan/models/kimi_k3`.

GPU (2026-09-22, 4 x RTX 5060 Ti, torch 2.15.0.dev20260906+cu130, seed 42, five steps, one warm compile cache shared by the cells; the local SM120 guard lift on `kda.py` is not in the diff). The B200 cell's shape, pp4 x vp2 with eight micro-batches, runs to completion: the tower stage installs, the plan places what the upfront prefix leaves (`2/2 planned encode(s) ran in a bubble, 4 upfront, 2 left inline, 14 idle slot(s)`), and every deferred tower backward runs at a planned slot with none drained at step end. The tower's forward is bitwise identical whether the encode runs ahead or inline: the per-micro-batch feature norms match line for line across the two cells. The losses are identical at steps 1 and 2 and then separate, `5.66006` against `5.65858` at step 3 and `4.11665` against `4.11153` at step 5, which is the order the tower's bf16 parameter gradients accumulate in; each cell reproduces its own numbers exactly on a rerun, so that gap is arithmetic and not run-to-run noise. The hundred-step table on H100 comes before the draft leaves DO NOT review.

--- PASTE BEGIN ---

Draft, stacked on the text-side PP PR (#4312): the diff tab shows that PR's content too, so review the three commits on top. It will be rebased when the text PR lands.

### Summary

Report sec 5.2.3: the vision tower takes a pipeline stage of its own ahead of the text stages, so its compute leaves the critical path of the stage that owns the embedding, and its encodes are placed around the schedule's own actions: ahead of the forward that reads them (`vision_dep.prefetch`), or in the schedule's idle intervals with the tower's backwards deferred to the intervals after backward actions (`vision_dep.bubble`).

### Design

- The tower stage is the split alone, no core change. `vit_dep_split` puts `[tok_embeddings, vision_encoder]` on the first stage and spreads the layers over the remaining stages through core's own `_generate_llm_fqn_per_model_part`, so the stage count the schedule sees is unchanged and the tower stage comes out of the text stages' budget. The embedding rides with the tower because the splice needs the token ids, which only the first stage receives. The block routing accepts a stage with no layers.
- The knobs are a `vision_dep` record on the model config: `enabled`, `prefetch`, `bubble`, `bubble_cost_ratio`, `bubble_max_pending`. They change the split every rank applies, so they belong to the model rather than to the command line, and the model config tree is off the CLI.
- `encode_images` is the tower's forward on one micro-batch's images, and `forward` takes `vision_embeds`, so the schedule's forward and the ahead-of-time encode cannot drift. An encode issued before the pipeline calls the stage gathers the FSDP parameters itself; the stage's own forward reshards them as its policy says. The first step encodes inline, because FSDP2 builds its state in the root module's first forward and an encode before that makes the tower a root of its own.
- `VisionDepPipelineStage` subclasses the AttnRes stage and hands the runtime each action as it completes, which is how a placement fires after the action it is anchored to. An anchor is an action's identity rather than a slot index, because the index does not survive lowering: the runtime walks `pipeline_order_with_comms`, which inserts sends and receives and holds no idle entries.
- The plan (`dep_bubble_plan.py`, pure) is read off `pp_schedule.pipeline_order[rank]`, so every rank derives the same placements with no collective and none reaches a vision collective the others do not. The first `pp` encodes run upfront, as the report prescribes; later ones go in the last idle slot whose accumulated budget covers `bubble_cost_ratio` text-stage actions, so the features stay resident as briefly as the budget allows.
- The tower's backward is cut at the splice (`dep_backward.py`): the features are spliced in through a detached stand-in whose gradient is captured and replayed at a later slot. Whatever the slots did not take is drained at step end, because a deferred backward that never runs leaves the tower without that micro-batch's gradient and raises nothing.

### Test plan

- `pytest tests/unit_tests/cpu -k "kimi_k3 or pipeline or cli or integration_test" -q` (96 passed). The bubble tests cover the placement invariants, a backward-anchored placement firing, the run-ahead, and the deferred backward: a cut and replayed tower backward is bitwise the gradient the inline one produces, out-of-order replays accumulate like one pass, the pending bound changes when rather than whether a gradient runs, and nothing is lost when no slot ever comes.
- The `kimi_k3_pp4_vp2_vit_dep` cell in the B200 suite, which derives its split from `vision_dep` rather than spelling one out.
- pp4 x vp2 with the bubble on and off, same seed and batch on one warm compile cache: the tower's forward is identical either way; the losses separate from step 3 by the order the tower's gradients accumulate in. The hundred-step table on H100 goes here.

--- PASTE END ---
