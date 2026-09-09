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

Five-step table in #4488's format on the same branch (`PP_NUMERICS_4488STYLE_2026-09-08.md`): bf16 flavor, dp1 vs pp2 (two 256-token micro-batches): step 1 loss bitwise, grad norm one bf16 ulp (3.9e-3), then max relative loss diff 8.6e-2 and grad-norm diff 8.3e-2 over steps 2-5; float32 masters with bf16 compute, torchtitan's default regime (9-layer alias, the 33-layer model does not fit fp32 states on 16 GB): step 1 loss bitwise, grad norm 1.1e-4, max 1.5e-2 / 1.8e-2. Not the pipeline's: dp1 twice and pp2 twice are bitwise, and the step-1 per-parameter profile (13 layers, fp32 masters) starts at ulp level in the last attention layer's query gradient inside stage 1 and grows 3-5x per layer down the backward with no jump at the boundary; Adam's first `lr * sign(g)` update turns the resulting sign flips into the step-2 spread. The DSV3 table's 1e-5 is a property of that model's backward gain, not a bar this model meets in any parallelism.

## 6. On #4312: replace the attachment with a rendered document (post this comment)

The 09-07 design note went out as a file attachment (`user-attachments/files/...md`): GitHub serves it as a raw text file, its relative image links do not resolve, so the reader saw no figures. Two remedies, use both: the rendered version in the public logbook (linked below), and the PDF with the figures embedded -- `phase13_k3like_48b_posttrain/PP_RUNTIME_DESIGN_NOTE_2026-09-08.en.pdf` (8 pages; rendered from the markdown with weasyprint, `PP_DESIGN_WORKFLOW_2026-09-07.en.pdf` likewise) -- attached to the comment. The figures below are absolute links that render inline.

--- PASTE BEGIN ---

@tianyu-l the design note I attached on 09-07 rendered as raw text without its figures; the rendered version is here: https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase13_k3like_48b_posttrain/PP_RUNTIME_DESIGN_NOTE_2026-09-08.en.md (rewritten against #4486's runtime, five figures, a 25-minute outline at the end for the talk). The three pictures that carry the argument:

![The stack grows across stages, partial blocks ride the wire, only the head stage aggregates](https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/main/phase13_k3like_48b_posttrain/figures/png/pp_attnres_dependencies.png)

![Two gradient routes: the pipeline's own backward P2P across ranks, a store slot within a rank](https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/main/phase13_k3like_48b_posttrain/figures/png/pp_dual_gradient_bridge_v2.png)

![torch's chain protocol against the multi-consumer edge, and where each of the five suggestions bites](https://raw.githubusercontent.com/QIU023/torchtitan_attention_residual/main/phase13_k3like_48b_posttrain/figures/png/pp_protocol_gap.png)

In #4486's terms: Block AttnRes is a second client of a model-owned pipeline runtime, with per-micro-batch ACTIVATIONS shared across stages where MTP shares a PARAMETER. `pp_runtime_client` on my fork is this PR rebased onto #4486 with the K3 entry returning a `PipelineResult` and an `AttnResPipelineRuntime` (drained-store check at step end); dp1 and pp2 read the same step-1 loss bitwise from one seed checkpoint. Section 2 of the note maps the lifecycle onto the hooks: the stage/rank map and the per-micro-batch hook are there, the store, the routing table, a micro-batch-end callback and the in-schedule gradient merge are not, and `prepare_microbatch` needs the micro-batch index (a counter in the runtime drifts, the metadata-inference pass calls it too). Happy to walk through it whenever suits.

--- PASTE END ---
