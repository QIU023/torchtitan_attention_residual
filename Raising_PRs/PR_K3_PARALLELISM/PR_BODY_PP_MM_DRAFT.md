# PR title: [DRAFT] DEP for Kimi K3 -- the vision tower takes a pipeline stage of its own

Branch `k3_pp_mm` (base = the `k3_pp_text` review head + one commit). File as DRAFT stacked on the text PP PR; do not undraft before that PR lands and the tables are re-measured on this branch. Paste between the markers.

--- PASTE BEGIN ---

Draft, stacked on the text-side PP PR: the diff tab shows that PR's content too, so review only the last commit (`kimi_k3: DEP -- the vision tower takes a pipeline stage of its own`). It will be rebased when the text PR lands.

### Summary

Report sec 5.2.3: the vision tower gets a pipeline stage of its own ahead of the text stages, so its compute leaves the critical path of the stage that owns the embedding and can hide in pipeline bubbles. Opt-in via `vit_dep` because it changes the stage count, which the schedule and the checkpoint layout both see; with it off, this commit changes nothing.

### Design

- The tower stage is the FQN split alone, no core change: `_inject_kimi_k3_fqns` prepends a `[tok_embeddings, vision_encoder]` stage and takes it OUT of the text stage budget (the schedule asserts `num_stages % pp_degree == 0`, so appending would break pp=2 at the first vision stage).
  - The tower rides WITH the embedding: the splice needs token ids, which only stage 0 receives; the text stack's forward already treats missing `tok_embeddings` as "input IS the hidden state".
- Engagement is asserted, not inferred: each vision stage is wired with its micro-batch index, the wired count is checked against what the rank should own by stage index, and a mismatch raises -- an unwired share passes activations through unprocessed and reports no error otherwise.
- A stage owns the tower if it HOLDS one (walking wrapped modules), not if it is a particular class: FSDP2 rewrites the module class, and the folded layout never constructs a dedicated ViT stage class, so a type test wired nothing on a correct run.
- The topology record moves from an adapter-local dataclass to `knobs.py` and gains the vision fields (`vit_dep`, `vit_dep_stages`, `vit_prefetch`, `vit_bubble`): these decide the pipeline topology, so every rank must resolve them identically, once, at the pipelining entry.
- `vit_dep_stages > 1` raises: splitting the tower across stages needs the split to address `vision_encoder.layers`, which core's `_split_module` cannot reach on this layout; the share entry points (`forward_head/body/tail`, contiguous block ranges recomputing position tables per share) are in place for when it can.
- Two placement knobs for the encode, alternatives by construction: `vit_prefetch` issues the encode for micro-batch m+k on a side CUDA stream during m's text compute (cross-stream tensor lifetime handled by wait/record_stream on both edges), and `vit_bubble` runs planned encodes in the schedule's idle intervals on the main stream, with deferred tower backwards bounded by a `GradQueue`.

### Results

Draft placeholder: the tables below were measured on the integration tree this commit is extracted from (8x RTX 5060 Ti, seed 42, `--debug.deterministic`, steps 1/3/10 protocol), not on this branch; they will be re-measured on this branch before the draft is undrafted.

Tower-stage split: pp2/pp4/pp8 all reproduce dp1's step-1 loss bitwise, with step-2 drift 1.85e-3 (reduction order across the stage boundary); with the delta transport on, pp2xvp2 / pp2xvp4 / pp4xvp2 / pp8xvp4 keep step 1 bitwise and stay within 1.5e-3 at step 2. Timing at debug scale, and what it does and does not show: the bubble arm fills 8/8 planned slots with 0 fallbacks and the prefetch arm is bitwise on loss with noise-level step time, so the placement mechanism engages where it says it does. It does not yet show the report's result -- most of the encoder hidden -- and at this scale it cannot: the schedule hides the encodes of all but the first micro-batches, so the share that can be hidden is `(mb - O(pp)) / mb`, and with the debug flavor's micro-batch count and tower size the bubble arm costs +4.2% per step. Both knobs are off by default and land here as groundwork.

### Changed files

    torchtitan/models/kimi_k3/
      pipeline_adapter.py     +442/-86  the DEP stage budget and FQN placement, stage
                                  wiring + engagement assertion, prefetch/bubble install;
                                  the topology record moves out to knobs.py
      dep_vision_stage.py     +270  the ViT stage module for a split tower (share entry)
      dep_bubble_backward.py  +248  GradQueue: deferred tower backwards, bounded
      dep_bubble_plan.py      +244  bubble placement plans from the schedule's own shape
      dep_bubble_runtime.py   +176  runs planned encodes in schedule idle intervals
      vit_prefetch.py         +235  the run-ahead: per-step feature cache on a side stream
      knobs.py                +113  the topology record, moved and extended
      vit_cp_plan.py          +136  stage-boundary packing: config-level upper bounds,
                                  never batch-derived (P2P buffers are sized once)
      vision_encoder.py       +139  the share entry points: head / body / tail
      model.py                +107  the DEP config fields; encode_images and the
                                  vision-stream issue/join pair
    tests/unit_tests/cpu/test_kimi_k3_vit_stage_shares.py  +90

### CI/CD Coverage

CPU tests: the share split's block-bound invariants (`test_kimi_k3_vit_stage_shares`); the FQN-injection tests on the base branch already pin the DEP stage placement, including the text-model-with-None-tower shape that once produced a silently wrong pipeline.

--- PASTE END ---
