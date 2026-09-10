# DEP on the new tree: the vision tower on a stage of its own (2026-09-10)

Port of the multimodal PP draft (PR 4381, `k3_pp_mm` f5e60f066, 2114 lines on the old adapter) onto `k3_int_20260910`'s stage design.

## What the old draft did, and what the new stage design already gives

- The tower stage: the old adapter's `_inject_kimi_k3_fqns` prepended a `[tok_embeddings, vision_encoder]` stage taken out of the text budget, with engagement assertions, a dedicated `KimiK3ViTStage` module, topology knobs in `knobs.py`. On the new tree the split is a function of the config (`kimi_k3_module_fqns_per_model_part`), the routing tables accept a layer-less stage (it commits nothing and sends an empty delta), and the model's forward on a stage without layers returns `(h, empty stack)` like any boundary: `vit_dep` is a keyword of `pipeline_kimi_k3`, set by a recipe, 20 lines.
- The placement of the encodes: `dep_bubble_plan.py` (pure), `dep_bubble_runtime.py` (fires planned encodes after the forward action before each idle run), `dep_bubble_backward.py` (`GradQueue`: the tower's backward cut at the splice and replayed in idle slots after backward actions, drained at step end), `vit_prefetch.py` (per-step feature cache) port verbatim; the glue moved from the adapter into `_install_vision_dep` in `parallelize.py`: the tower stage's `forward_one_chunk` serves the cache and injects the features as the model's new `vision_embeds` argument; `schedule.step` is wrapped to read the trainer's `kwarg_mbs`; the bubble plan reads the real schedule's `pipeline_order[rank]` instead of a fake schedule. Not ported: `vit_dep_stages > 1` (tower shares across stages), `VisionStepInputs`, the topology record.
- The old prefetch runs on a side stream only outside autograd (its own design note), so in training it is the cache path with inline encodes; the bubble mode is the placement that works.

## Results (33-layer debug model, 4096 tokens per step in 256-token micro-batches, 8 pipeline micro-batches, spmd_types, one seed checkpoint, `.mx3_seeds_main33`)

| cell | loss 1 | loss 2 | loss 3 | notes |
| --- | ---: | ---: | ---: | --- |
| pp2, no DEP | 12.40087 | | 7.65370 | |
| pp2, DEP | 12.40087 | | OOM | stage 1 holds all 33 layers on one 16 GB card: the expected imbalance at pp=2 |
| pp4 Interleaved1F1B x vp2, no DEP | 12.40087 | 10.55587 | 7.73265 | |
| pp4 x vp2, DEP | 12.40087 | 10.43479 | 7.62074 | step 1 bitwise; later steps differ from the other split by this box's usual step-2 amount |
| pp4 x vp2, DEP + bubble (cost ratio 0.5) | 12.40087 | 10.43479 | 7.62074 | bitwise with DEP alone; per step 4 upfront, 2/2 planned encodes in bubbles, 2 synchronous, 14 idle slots; 2 deferred tower backwards at planned slots, 0 drained |
| pp4 x vp2, DEP + prefetch depth 1 | 12.40087 | 10.43479 | 7.62074 | bitwise with DEP alone; 8/8 cache hits per step from the second step (the first step is inline) |
| pp8 x vp4 recipe, seeded, no DEP | 12.40087 | | 7.68959 | |
| pp8 x vp4 recipe, seeded, DEP | (below) | | | |

Unseeded recipe runs (each rank initialising its own parts) read different step-1 losses with and without DEP (12.48589 vs 12.59142): the trainer initialises weights per model part, so the split changes the random draws. Not a numerics difference; the seeded cells are the comparison.

Two things the port fixed on the way: the bubble runtime and the step wrapper must be installed so `begin_step` runs outermost, or the plan sees the previous step's micro-batch count; and an encode issued before the stage's first forward makes the tower an FSDP root of its own ("FSDP state has already been lazily initialized"), so the first step runs inline.

Commit on `k3_int_20260910`: see `git log` (the DEP commit). Local aliases for the cells (`kimi_k3_debugmodel_pp2_vit_dep`, `_pp4i`, `_pp4i_vit_dep`, `_pp4i_vit_dep_bubble`, `_pp4i_vit_dep_prefetch`) stay uncommitted in `/tmp/wt_k3int2`'s registry; the committed recipe is `kimi_k3_debugmodel_pp8_vp4_vit_dep` with its integration cell.
