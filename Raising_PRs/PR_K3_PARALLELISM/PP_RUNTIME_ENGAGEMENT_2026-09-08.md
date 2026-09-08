# Pipeline runtime engagement (2026-09-08): what to post, where

Companion to `phase13_k3like_48b_posttrain/PP_RUNTIME_DESIGN_NOTE_2026-09-08.en.md` (the talk material). Three comments, none of which asks about the state of #4312 or #4313; the design question carries the conversation.

## 1. On #4486 (review comment, top level)

--- PASTE BEGIN ---

One design input from a second client of this runtime. Kimi K3's Block Attention Residuals need per-micro-batch ACTIVATIONS shared across stages: a block committed at stage S is read by every later stage, and its gradient returns from each of them to S -- a multi-consumer edge with micro-batch lifetime, where the MTP case is a single-owner parameter with step lifetime. #4312 implements it today as a `PipelineStage` subclass with a rank-local value store (validated step-1 bitwise against one GPU from 2 to 32 stages); the design note is [PP_RUNTIME_DESIGN_NOTE](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase13_k3like_48b_posttrain/PP_RUNTIME_DESIGN_NOTE_2026-09-08.en.md), section 2 maps its lifecycle onto these hooks.

What this runtime already gives that client: `PipelineResult.stage_indices` with the stage-to-rank map is the input of the routing table that decides what each hop carries, and `prepare_microbatch` runs per micro-batch while `kwarg_mbs` is built, the right place to hand every stage the micro-batch id (the key that has to survive P2P; tensor identity does not, NCCL hands out fresh receive buffers). One thing I hit bringing the client onto it (`pp_runtime_client` on the fork): the hook carries neither the micro-batch index nor a step boundary, and the metadata-inference pass calls it too, so a counter kept in the runtime drifts from the schedule's chunk id; passing the index into the hook would settle it.

What it does not: a micro-batch-end callback (the store releases a block after the rank's last virtual stage forward of that micro-batch), and a point INSIDE the schedule where a same-rank consumer's gradient is merged before the producer's backward; `finalize_gradients` after all backward passes is too late for that, and a placement on a parameter cannot describe an activation that has no owner before the forward that creates it. So my reading is: parameters can go either way (runtime or placement), activations need the runtime plus one ordering guarantee the schedules already provide (later virtual stages' backward first on a rank). If that matches your direction I will bring the AttnRes client onto `PipelineResult` as a stacked draft so the shape can be checked against two clients.

--- PASTE END ---

## 2. On #4313 (issue comment)

--- PASTE BEGIN ---

#4500 covers the same scope as this PR on the same kernel stack (KCP on KDA through Attention Gym, all-gather K/V and Ulysses on MLA, the multimodal inputs under cp), so I will re-scope this one to what it adds on top of #4500 once that lands: the MLA-specialised Ulysses and all-gather kernels (the packed exchange, one all-to-all for `(q | k_nope | v)` and the rope slice moved once as the headless vector it is, which @fegin noted is faster than the generic kernel on MLA), the TP/SP declarations of #4492 that #4500's `sharding.py` does not carry, and packed-document support once #4459 lands. One measurement that may help #4500's review: on one compile cache cp2 reads the same step-1 loss as dp1 bitwise; a 0.6 percent step-1 gap between CP=1 and CP=2 means the two cells read different tokens or the mask differs, and step 3/10 only compare once both cells share one inductor cache. The one-cache protocol is in this PR's results section.

--- PASTE END ---

## 3. On #4500 (review comment)

--- PASTE BEGIN ---

Same scope as #4313 on this stack; I will re-scope that PR to the deltas (the packed MLA kernels, the TP/SP declarations) once this lands. One ask for the results table: CP=1 and CP=2 at step 1 on the same global batch and one compile cache -- in #4313's matrix they are bitwise, so the 0.6 percent here most likely comes from different tokens per cell rather than from the kernels; and the grad-norm column will move by percents from the compile cache alone unless both cells share one.

--- PASTE END ---

## 4. The talk

Use the note's sections 1-6 in order (outline in its section 7, 25 minutes). The one sentence to land: parameters can be a placement or a runtime concern; cross-stage activations need the runtime plus one ordering guarantee the schedules already provide.

## 5. The client branch

`pp_runtime_client` on the fork (`464421e13`): #4312's 17 commits rebased onto #4486's head (`713a6bbb6` = main `2af775ea9` + its two commits) plus one commit -- `pipeline_kimi_k3` returns the `PipelineResult` with an `AttnResPipelineRuntime` (advisory micro-batch tag on the kwargs, drained-store check at step end), `pipeline_llm` keeps the `stage_class` hook next to `stage_args_factory` (9 lines in the core file). Unit tests 21 (ours and #4486's). Debug model, one seed checkpoint, `partial_dtensor`, 3 steps: dp1 12.41853 / 7.64136, pp2 (8 micro-batches, delta transport) 12.41853 / 7.63540 -- step 1 bitwise, the store drained every step. To move the PR head: `git push origin pp_runtime_client:k3_pp_text --force-with-lease` (the user's action) after #4486's shape settles; rebase again when #4486 or #4488 (`stage_metadata_fn`) moves.
