# Pipeline runtime engagement (2026-09-08): what to post, where

Companion to `phase13_k3like_48b_posttrain/PP_RUNTIME_DESIGN_NOTE_2026-09-08.en.md` (the talk material). Three comments, none of which asks about the state of #4312 or #4313; the design question carries the conversation.

## 1. On #4486 (review comment, top level)

Scope note for us: the comment reports what a second client measured against THEIR hooks and asks for one line at a call this PR already edits. The lifecycle gaps AttnRes still has are deferred to our own diff after this lands -- asking a maintainer to design for an unmerged client is how the CP line got overtaken. The `stage_class` ask has no in-tree user before #4312 and the comment says so, so it can be declined cleanly.

--- PASTE BEGIN ---

Two things from bringing a second client onto this runtime, both about the hooks rather than the client.

**`prepare_microbatch` cannot give a client the micro-batch it is preparing.** `PipelineRuntime.prepare_microbatch` (`pipeline_parallel.py:72`) receives the inputs and kwargs but no index, and `Trainer.pp_forward_backward_step` calls it while `kwarg_mbs` is built, so a client that needs the identity of a micro-batch has to count inside the runtime. That count is wrong in three ways:

- Nothing marks a step boundary, so the counter can only be reset from another hook -- `finalize_gradients` (`pipeline_parallel.py:84`), which runs in the optimizer block of a training step.
- The schedule's metadata inference re-executes the first micro-batch's forward (`_initialize_stage(arg_mbs[0], kwarg_mbs[0], ...)`, `torch/distributed/pipelining/schedules.py:843`), so anything counting per forward sees micro-batch 0 twice while the schedule's chunk ids stay unique.
- Validation builds its own micro-batches and drives `pp_schedule` from `components/validate.py`, reaching neither hook, so the runtime is bypassed there entirely.

Passing the schedule's chunk id into the hook settles all three: it is the one key that survives P2P, since tensor identity does not (NCCL hands out fresh receive buffers). For a shared parameter with step lifetime none of this matters, which is why it does not show up in the MTP case.

**One line at a call this PR already edits.** `stage_args_factory` lets a model supply a stage's static input and output metadata, and `_pipeline_module_split` now passes them into `PipelineStage(...)` (`pipeline_parallel.py:966`). A client that overrides the stage's forward and backward needs the class itself as well -- `stage_class: type[PipelineStage] = PipelineStage` threaded to the same call. There is no in-tree user for it before #4312, so this is only worth taking if the generalization looks right to you.

The client is Kimi K3's block attention residuals: activations shared across stages with micro-batch lifetime, where MTP shares a parameter with step lifetime. `pp_runtime_client` on my fork is #4312's stage on this PR's `PipelineResult`, dp1 and pp2 bitwise at step 1 from a shared seed checkpoint; today it uses `finalize_gradients` as a step-end check that the block store drained, and the design note is [here](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase13_k3like_48b_posttrain/PP_RUNTIME_DESIGN_NOTE_2026-09-08.en.md). The two lifecycle points it still lacks -- a micro-batch-end release, and a place inside the schedule where a same-rank consumer's gradient merges before the producer's backward -- I will raise as concrete diffs once this lands rather than argue them here.

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

## 7. Comments for the TP/SP re-scope (2026-09-09)

On #4492 (close it after posting):

> Closing: #4500 carries the CP-side declarations this PR made (the KDA local map, the tower over cp, the multimodal inputs) with the spmd_types backend for CP, so the remaining delta is TP/SP only. That now lives in #4499, re-pointed at a branch stacked on #4500's head; the `clip_grad_norm_` per-mesh grouping goes with it.

On #4499 (after the head moves to `tp_sp_on_4500` and the title loses the "DO NOT review" prefix):

> Re-based onto #4500's head (`2884d82a9`) as three commits, the TP/SP delta only; the PR stack is #4500 -> #4450 -> #4449 -> #4322, and the body follows #4500's format (same protocol, tp=1 on the parent as the reference, 100 steps). One note for #4500 from bringing TP onto it: at tp > 1 the tower's colwise / rowwise projector declaration leaves a Partial at its exit that nothing reduces when sequence parallel is off, so under spmd_types this branch declares the tower invariant on tp (it runs whole on every rank, as under partial_dtensor). Happy to fold that into #4500 instead if preferred.

On #4412 (with the head synced to `qb_review4`):

> Synced the head to main (`65ba8a697`); the CI run on the old head hit seven pyrefly type errors (fixed here: the bias is fetched once as a Tensor, `post_optimizer_build_fn` is set on an asserted model_spec) and the `torch.cuda._annotate_cuda_graph_trace` import that main's CI hit the same night (#4493 removed it). Locally on this head: the 14 CPU tests and the K3 GPU tests pass, pyrefly is clean on the touched files. Could you re-run CI?

## 8. On #4500: the multimodal encoder's own CP (2026-09-09)

No draft PR yet -- `k3_cp_mm`'s base predates the EP merge, so a PR from it would show hundreds of files. The comment carries the commit; the draft follows the re-cut on #4500's head.

--- PASTE BEGIN ---

The tower is replicated on the cp axis here (`_set_vision_encoder_sharding`), which leaves the other multimodal item of report section 5.2.3 open: partitioning one large image along the patch dimension across cp ranks with gather-KV inside the tower, and dividing a cp group into sub-groups so several large images balance across them.

That part is implemented and ran on 8 GPUs: https://github.com/QIU023/torchtitan/commit/a5339256e8eb6e18d6ca9a9a4aea26f3c2fefa8f -- the planners in `vit_cp_plan.py`, gather-KV attention and position-table slicing in `vision_encoder.py`, sub-group dispatch in `_encode_images`, about 750 lines inside the model folder with one line of pass-through in `common/vision_encoder.py`. The commit sits on a pre-EP base, so I will re-cut it on this PR's head and raise it as a stacked draft. It does not overlap this PR: the encoder stays replicated whenever no sample is large enough to partition.

--- PASTE END ---
