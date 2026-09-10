# PR title: [DRAFT] DEP for Kimi K3 -- the vision tower takes a pipeline stage of its own

Branch `k3_pp_mm_v2` (= the published `k3_pp_text` head a3be242bf + two commits, `a0829b471` and the docstring commit on top). Replaces the old-tree head `k3_pp_mm` (f5e60f066) of PR 4381 once the GPU cells on this branch are in (see `phase13_k3like_48b_posttrain/DEP_NEW_TREE_2026-09-10.md`). File as DRAFT stacked on the text PP PR 4312.

--- PASTE BEGIN ---

Draft, stacked on the text-side PP PR (#4312): the diff tab shows that PR's content too, so review the two commits on top. It will be rebased when the text PR lands.

### Summary

Report sec 5.2.3: the vision tower gets a pipeline stage of its own ahead of the text stages, so its compute leaves the critical path of the stage that owns the embedding, and its encodes can be placed around the schedule's own actions: ahead of the consumer (`vit_prefetch`), or in the schedule's idle intervals with the tower's backwards deferred to idle intervals after backward actions (`vit_bubble`).

### Design

- The tower stage is the split alone, no core change: `kimi_k3_module_fqns_per_model_part(..., vit_dep=True)` puts `[tok_embeddings, vision_encoder]` on the first stage and spreads the layers over the remaining `num_virtual_stages - 1`; the stage count the schedule sees is unchanged, so the tower stage comes out of the text stages' budget. The embedding rides with the tower because the splice needs the token ids, which only the first stage receives. The pipeline stage's block routing accepts a stage without layers.
- `pipeline_kimi_k3` takes `vit_dep`, `vit_prefetch`, `vit_bubble`, `vit_bubble_cost_ratio`, `vit_bubble_max_pending`; a recipe sets them (they change the split every rank applies), no CLI flags.
- The model exposes `encode_images` (the tower's forward on one micro-batch's images) and takes `vision_embeds` in `forward`: the tower stage's `forward_one_chunk` is wrapped to take the cached features when the encode already ran, so the schedule's forward and the ahead-of-time encode cannot drift.
- `vit_prefetch` (`vit_prefetch.py`): the first step encodes inline (FSDP's lazy init has to see the tower on the schedule's own path first); from the second step the encode for micro-batch m+k is issued as m's forward begins. The run-ahead uses a side stream only outside autograd, so in training it is the cache path with inline encodes.
- `vit_bubble` (`dep_bubble_plan.py`, pure; `dep_bubble_runtime.py`; `dep_bubble_backward.py`): the placement plan is read off `pp_schedule.pipeline_order[rank]`, so every rank derives the same placements without a collective; the first `pp` encodes run upfront as the report prescribes, later ones in idle runs whose length covers `vit_bubble_cost_ratio` text-stage forwards; the tower's backwards are cut at the stage boundary (`cut_for_deferred_backward`) and drained from a bounded queue in the idle intervals after backward actions.
- Install order: the bubble runtime and the backward slots go in before `step` is wrapped with `begin_step`, so the prefetcher's per-step bookkeeping is outermost.

### Results

Integration tree (`k3_int_20260910`, 8x RTX 5060 Ti, 33-layer debug model, 4096 tokens per step in 256-token micro-batches, 8 pipeline micro-batches, spmd_types, seeded, `--debug.deterministic`):

| cell | loss 1 | loss 2 | loss 3 |
| --- | ---: | ---: | ---: |
| pp4 Interleaved1F1B x vp2, no DEP | 12.40087 | 10.55587 | 7.73265 |
| pp4 x vp2, DEP | 12.40087 | 10.43479 | 7.62074 |
| pp4 x vp2, DEP + bubble (cost ratio 0.5) | 12.40087 | 10.43479 | 7.62074 |
| pp4 x vp2, DEP + prefetch depth 1 | 12.40087 | 10.43479 | 7.62074 |
| pp8 x vp4 recipe, no DEP | 12.40087 | | 7.68959 |
| pp8 x vp4 recipe, DEP | 12.40087 | 10.44804 | 7.68082 |

Step 1 bitwise with dp1 in every cell; the three DEP placements are bitwise with each other (the placement moves when the encode runs, not what it computes); the later-step difference between splits is the reduction-order class this box shows for every parallelism. Bubble mode at pp4 x vp2: 4 upfront encodes, 2 of 2 planned encodes in bubbles, 2 synchronous, 14 idle slots per step; prefetch: 8 of 8 cache hits per step from the second step. pp2 with DEP runs out of memory on a 16 GB card (all 33 layers on one stage), the expected imbalance.

On this branch (`k3_pp_mm_v2`, its own seed checkpoint, 3 steps):

| cell | loss 1 | loss 3 |
| --- | ---: | ---: |
| dp1 | 12.41967 | 7.49054 |
| pp8 x vp4 recipe | 12.41967 | 7.61791 |
| pp8 x vp4 recipe with `vit_dep` | 12.41967 | 7.71360 |

Step 1 bitwise across the three; the later-step differences are the split's reduction order, as on the integration tree.

### Changed files

    torchtitan/models/kimi_k3/
      parallelize.py           the tower-stage split, the pipeline knobs, _install_vision_dep
      model.py                 encode_images, vision_embeds, the vision side stream
      pipeline_stage.py        routing for a stage without layers
      vit_prefetch.py          the run-ahead cache
      dep_bubble_plan.py       placement plans from the schedule's action order
      dep_bubble_runtime.py    encodes in the idle intervals
      dep_bubble_backward.py   deferred tower backwards
    torchtitan_recipes/tests/features.py   kimi_k3_debugmodel_pp8_vp4_vit_dep
    tests/unit_tests/cpu/test_kimi_k3_vit_dep_split.py, test_kimi_k3_dep_bubble.py (21 tests)

--- PASTE END ---
