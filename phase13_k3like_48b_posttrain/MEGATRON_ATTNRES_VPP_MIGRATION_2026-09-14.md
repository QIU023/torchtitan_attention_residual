# Megatron side of Block AttnRes interleaved PP (rank-local cache): status, overlap, plan

2026-09-14. Research only. No NVIDIA PR, issue comment or review until torchtitan PR 4312 merges (memory `megatron-attnres-port-waits-for-4312`). Sources: forks as submodules `Megatron-LM/` (= NVIDIA main `54c62df`) and `Megatron-Bridge/` (= main `99702e1`); PR head clones under the session scratchpad `mgh/` (`ar6840` = PR 6840 head `64fbfe3`, `k3dev` = yuzhongw `kimi_k3_dev`, `mlm`, `mbr`).

## 1. Verdict

The design is already being implemented in Megatron-Core by NVIDIA: NVIDIA/Megatron-LM PR 6840 (jingqiny-99, draft, base `dev`, opened 2026-08-25 06:58Z, last push 2026-09-10) adds Block AttnRes with plain PP and interleaved VPP, and its VPP path is the same rank-local source cache plus delta payload plus deferred cross-chunk gradient that torchtitan PR 4312 carries (4312 opened 2026-08-25 01:46Z, five hours earlier; RFC pytorch/torchtitan#3029 opened 2026-04-20; the fork's PP8xVP4 validation is from July). VPP entered 6840 in commit `70855b4c` on 2026-08-28. An independent Megatron port would duplicate an NVIDIA-owned draft, so the plan below is not "port the adapter" but "measure and contribute the differences once 4312 merges".

## 2. State of each repo

| where | what exists | PP / VPP |
| --- | --- | --- |
| Megatron-LM `main` 54c62df | no AttnRes, no KDA; mHC hyper-connections folded into hidden (`[s,b,n*C]`) | mHC raises under PP>1 (`transformer_config.py:2318-2329`, "p2p buffers are sized from hidden_size") |
| Megatron-LM `dev` | KDA base (PR 6556); mHC gets `use_nstream` widening in `get_tensor_shapes` (fixed n x hidden, one tensor) | mHC PP via fixed widening |
| MLM PR 6840 (draft) | `megatron/core/transformer/attention_residual.py`, TransformerBlock and HybridStack wiring, MTP, FLA kernel; +4511/-74, 21 commits | PP full prefix; VPP rank-local cache, delta payload padded to one uniform width; rejects `variable_seq_lengths`, sequence packing, VPP with embedding split |
| MLM PR 6427 (draft, wuweiqiang24, base `core_r0.18.0`) | +24/-30 in the interleaved schedule (shape exchange, `set_current_microbatch` in backward, `module.vp_stage`); empty body | schedule plumbing only; the "cache" commit carries no cache code |
| MLM issue 6872 (K3 tracker, yuzhongw-nvidia) | AttnRes marked in progress, points at 6840; text still says non-interleaved PP | frozen preview branch `kimi_k3_preview_20260907` already contains the 6840 VPP cache |
| Megatron-Bridge `main` 99702e1 | K3 provider (PR 5130, merged 2026-07-30): own AttnRes layers, KDA via FLA `chunk_kda`, no MoonViT, no recipes | plain PP only: `[prefix | flattened bank]` along hidden with `variable_seq_lengths=True`, bank carried in the `context` slot; VPP raises (`kimi_k3_spec.py:110-111`); stage entry/exit ignore `vp_stage` (`kimi_k3_layers.py:475-480`); KDA with CP raises |
| Bridge issue 4910 (K3 roadmap) | every item unchecked, incl. "interleaved PP with per-microbatch cross-stage caching and backward gradients" | planned, no assignee |
| MLM PR 4398 (closed by its author a minute after opening) | Full/Block AttnRes for dense GPT | no PP state transfer; would be silently wrong under PP |

## 3. 6840 against 4312, piece by piece

| piece | torchtitan PR 4312 (`k3_pp_text`) | MLM PR 6840 |
| --- | --- | --- |
| cache | `RankLocalCache` per pp rank (`pipeline_adapter.py:140`), keyed by microbatch and block commit | `_AttnResSourceCache` process singleton (`attention_residual.py:311`), keyed by within-chunk microbatch id, holds detached `requires_grad` leaves |
| deferred gradient | `_LocalCacheCapture` autograd Function (365) plus an augment hook that adds the deposited grad | `_AttnResGradTap` (346): identity whose backward adds `leaf.grad` into the in-graph grad |
| payload layout | blocks as a separate `[T, K, D]` axis, width exact per hop from the static layout tables; `PipelineStage` shape inference returns the per-stage delta shape (`_forward_shape_inference`, 584) | sources plus partial concatenated along the sequence dim, padded to the maximum delta over all P*V-1 boundaries (`attn_res_uniform_payload_slices`, 234; `schedules.py:1305`) because the interleaved schedule has one `tensor_shape` |
| schedule changes | none in core; the adapter wraps the stage module | `tensor_shape[0] *= uniform_slices` in the interleaved schedule, cache reset at schedule entry |
| multi-rank test | real pipeline stages and schedule on CPU, exact block gradients against a single-stage reference, incl. a split with a block boundary inside a stage (`test_kimi_k3_pp_exact_block_grads.py`) | single-process emulation: stages run sequentially with a per-rank dict swapped in, one microbatch (`microbatch_id=0`), backward in reverse stage order (`test_attention_residual.py:1040-1116`); the real interleaved 1F1B order with several microbatches in flight is not exercised |
| GPU evidence | PP8xVP4 numerics and step time on the fork | none posted; GB200 PP2/EP2 proxy promised |

Scope note (user, 2026-09-14): Megatron drives its own P2P calls, unlike `torch.distributed.pipelining`, so payload layout, padding and stage splits are Megatron's business and are not compared or proposed; pp8 vp4 is only this project's own setup. The comparison that matters is the cache (section 3c). The padding numbers below are background only.

Padding cost of the uniform width, computed with 6840's own `attn_res_boundary_delta_slices` (slices of `[s,b,h]`; near-even layer split stubbed for `get_transformer_layer_offset`, since 93 layers need a custom layout in Megatron; script `scratchpad/slices6840.py` logic inlined in the session):

| layout | exact per-hop sum | padded (max x boundaries) | overhead |
| --- | ---: | ---: | ---: |
| 93 layers, pp8 vp4, block 12 | 81 | 3 x 31 = 93 | +14.8% |
| 93 layers, pp8 vp2, block 12 | 59 | 5 x 15 = 75 | +27.1% |
| 93 layers, pp4 vp4, block 12 | 37 | 3 x 15 = 45 | +21.6% |
| 96 layers, pp8 vp4, block 12 | 83 | 3 x 31 = 93 | +12.0% |

Caveat: under 6840 the 93-layer rows cannot actually be configured. It rejects `pipeline_model_parallel_layout` and `num_layers_in_first/last_pipeline_stage` with AttnRes, and the even split asserts `num_layers % pp == 0` and `(num_layers / pp) % vp == 0` (`transformer_block.py:175-197`); 93 = 3 x 31, so pp must be 3, 31 or 93. The rows show what the padding would cost once uneven splits are allowed. These counts use 6840's accounting (every slice is one hidden-sized tensor, the partial included) and are not comparable with the 167 -> 83 unit figure for 4312 (different split and unit). Not measured on GPU.

Also observed (read, not run): the `_UNIFORM_SLICES_MEMO` key omits `pipeline_model_parallel_layout` and the first/last stage layer counts, so two configs in one process that differ only there share a memo entry.

## 3b. How the 6840 VPP cache works (read from `ar6840`, head `64fbfe3`)

- Entry point: `AttnResStageSources` (`attention_residual.py:386`), created by `TransformerBlock.forward` at chunk entry (`transformer_block.py:981-996`) via `enter(...)`, fed each block start by `append_block_start`, closed by `exit_pack` (payload for the next chunk) or `exit_aggregate_values` (head stage).
- Microbatch id: read from `self.layers[0].current_microbatch`, the attribute `schedules.forward_step` already sets for TE CUDA-graph replay (`set_current_microbatch`, `schedules.py:494-495`); no new schedule plumbing.
- Cache: one process-wide dict `mb -> list of leaves` (`_AttnResSourceCache`, 311). At entry of chunk v>0 the rank takes `cache[mb]` (the full source list as of its previous chunk), appends the unpacked delta, and asserts the count equals `sources_formed_through(layers_before)`. At exit it overwrites `cache[mb]` with the new list, or pops it on the rank's last chunk (v = V-1); backward never reads the cache, only autograd-saved references.
- Gradient bridge: every source that first materializes in a chunk (each received delta slice and each locally formed block start) goes through `attn_res_tap_source`: the cache gets `tensor.detach().requires_grad_()`, the in-graph copy goes through `_AttnResGradTap`, an identity whose backward adds `leaf.grad` and clears it. Later chunks read the leaf, so their backward (which interleaved 1F1B runs first for the same microbatch) accumulates into `leaf.grad`; the producing chunk's backward drains it. autograd's own `.grad` accumulation is the slot, so there is no slot dict and no key.
- Payload: `[*delta_sources, partial]` concatenated along the sequence dim, zero-padded to `attn_res_uniform_payload_slices` (max delta over all P*V-1 boundaries, memoised), so the schedule's single `tensor_shape` becomes `seq * uniform_slices` (`schedules.py:1300-1307`); rotary embedding length is patched for the same reason (`rotary_pos_embedding.py:306`). The receiver slices off the real slices before chunking so pad rows never enter a source.
- Layer side: each `AttnResTransformerLayer` has two aggregations (`self_attention_attn_res`, `mlp_attn_res`, `transformer_layer.py:3003-3004`) and receives the stack as the `attn_res_sources` keyword; an assert checks its arity per layer.
- Reset: `attn_res_source_cache_reset()` at the schedule entries (`schedules.py:735-737`, `1184-1186`) is nested inside `if moe_paged_stash:`, so the "safety net" runs only with paged stash; normal runs rely on the last-chunk eviction and the stale-entry asserts.
- Refused with AttnRes (`transformer_config.py:1592-1724`): full recompute, CUDA graphs, `overlap_moe_expert_parallel_comm`, fused residual RMSNorm, fp32 residual, cpu offloading, custom pipeline layouts and first/last stage counts, variable seq lengths or packing with PP, VPP with embedding/loss split, heterogeneous block specs.

Against 4312's adapter:

| | 6840 | 4312 |
| --- | --- | --- |
| where it lives | core `TransformerBlock` / `HybridStack` plus one schedule line | model folder: `CrossStageCacheAdapter` wraps the stage module, no core change |
| mb id | existing `current_microbatch` attribute | thread-local per adapter |
| cached per mb | whole source list, replaced each visit | appended per commit with producer metadata `(rank, stage, idx)` |
| received blocks | tapped like local ones (leaf + tap) | kept attached: the P2P recv buffer is already a leaf, so the schedule's backward send drains their grads with no bridge |
| own-rank blocks | leaf `.grad` accumulation, drained by the tap | detached copy + `_LocalCacheCapture` depositing into a `(mb, stage, idx)` slot, producer-side grad hook sums it in |
| lost-gradient check | none (a consumer backward that never fires leaves `leaf.grad = None` silently) | hook compares deposit count with `expected_same_rank_captures` from the layout tables and raises |
| eviction | at forward of the last chunk | step-end sweep after backward, plus slot clearing |
| extras 4312 lacks | hybrid stacks, MTP, FLA aggregation kernel, memory-lean custom Function, attention-scope offload | |

## 3c. Cache only: the two designs side by side

Both rest on the same ordering fact: for one microbatch, every later same-rank chunk runs backward before the chunk that produced a cached source (Megatron: reverse chunk order in backward, `schedules.py:1244-1249`; torch pipelining: the interleaved schedules' backward order). Both key the cache by the within-chunk microbatch index. The differences:

1. What goes into the cache. 6840 treats every source the same: whether it arrived in the delta or was formed locally, the cache holds `detach().requires_grad_()` and the in-graph copy goes through `_AttnResGradTap`. 4312 splits by origin: blocks that arrived over P2P stay attached (slices of the recv buffer, itself a leaf, so a later chunk's backward accumulates straight into the buffer's `.grad` and the schedule sends it back), and only this rank's own commits are detached.
2. The gradient slot for own-rank sources. 6840 uses the leaf's `.grad` (autograd's AccumulateGrad sums the later chunks, the tap's backward adds it and sets `leaf.grad = None`). 4312 keeps an explicit dict keyed `(mb, producer_stage, idx)`: `_LocalCacheCapture.backward` deposits (first deposit cloned), a `register_hook` on the producer's block pops and adds. Same maths; 6840's is shorter.
3. Detecting a lost gradient. 6840 has none: if a consumer's backward never ran before the drain, the tap adds nothing and the late `.grad` lands on an orphaned leaf, silently. 4312 counts deposits per slot against `expected_same_rank_captures` from the layout tables and raises; commits with no grad path are marked so consumers do not deposit; a step-end sweep clears stray slots.
4. Granularity and eviction. 6840 stores the whole source list per microbatch, overwritten on each visit and popped at the forward of the rank's last chunk (backward only uses references held by the tap contexts and the consumer graphs). 4312 appends one entry per block with producer metadata and drops at microbatch or step end; tensor lifetimes are the same either way because the graphs hold them.
5. Where the cache boundary sits relative to recompute. 6840 taps inside the layer loop (`append_block_start` per block start), and its full-recompute path does not carry the stack, so full recompute is refused. 4312's cache reads and writes happen in the stage wrapper, outside the model, so any activation checkpointing inside the stage never re-runs a cache write.
6. Safety asserts both have: 6840 checks the reconstructed source count against the layer index and asserts no stale or missing entry per microbatch; 4312 checks delta sizes and commit counts against the layout tables.

What each side could take from the other (cache only):
- For 6840: the deposit-count check (item 3), and optionally not tapping received sources, since Megatron's `backward_step` also reads `input_tensor.grad` after `torch.autograd.backward` (`schedules.py:556-592`), so an attached recv slice would accumulate there the same way (not tried).
- For 4312 (only if review asks for less code): the leaf `.grad` slot could replace the captured-grad dict and `_LocalCacheCapture`, keeping a counter for the check.

## 4. What Megatron-Core needs for this design (from the MCore PP survey, 54c62df)

Ranked by difficulty; all paths under `megatron/core/`.

1. One tensor per hop: `GPTModel.set_input_tensor` asserts one tensor (`models/gpt/gpt_model.py:327`), `backward_step` backprops only `output_tensor[0]` (`pipeline_parallel/schedules.py:579`), and `deallocate_output_tensor` requires a non-view (171-201). Everything must be packed into one fresh 3-D tensor. MTP standalone (concat along dim 0, `transformer/multi_token_prediction.py:2418-2530`) is the template; 6840 follows it.
2. Variable width in the interleaved schedule: one `tensor_shape` for every hop (1184-1187); steady state receives forward and backward tensors in one `_communicate` call with one shape (`p2p_communication.py:650-671`). Options: pad to the maximum (6840), the blocking 3-int shape handshake of `variable_seq_lengths` (one extra `batch_isend_irecv` plus `.tolist()` host sync per call, 3-D only), or per-(chunk, direction) static shapes through `_communicate` and about a dozen receive sites across the three communication paths plus combined 1F1B.
3. Gradient routing for cached sources: backward has no microbatch id (`get_microbatch_id_in_model_chunk` asserts forward, 1251-1255) but chunk FIFO order gives it; reverse chunk order in backward (1244-1249) guarantees consumers run before the producer, which both designs rely on.
4. Recompute: `checkpointed_forward` passes a fixed positional tuple (`transformer/recompute.py:146-180`); the stack and the block commits must go through it, with units aligned to blocks. mHC's recompute plan (`transformer_block.py:655-672`) is the closest precedent.
5. The store and its lifecycle: nothing stores activations per microbatch across chunks today; `PipelineOffloadManager` (`fine_grained_activation_offload.py:443-461`) is the precedent for a per-rank singleton reset at schedule entry/exit, including the forward-only path.
6. CUDA graphs: per-layer graphs can take per-layer static shapes (`get_layer_static_inputs` override, as mHC does at `transformer_layer.py:1835-1851`); full-iteration graphs with a host cache or the shape handshake are likely incompatible (not tested).
7. Layout tables and the final aggregation: easy; `pipeline_model_parallel_layout` gives every rank the global layer ids locally, and `post_process` / final layernorm placement gate the head aggregation.

## 5. Plan

Nothing is filed before 4312 merges. After it merges, in order:

1. Re-read 6840 (it may have merged or changed by then) and rerun its unit tests on the box.
2. Real-schedule test: run 6840's cache under `forward_backward_pipelining_with_interleaving` with several microbatches (pp2 vp2 and pp4 vp2 on 2 to 4 GPUs, or gloo on CPU if the schedule allows), comparing exact source gradients against the unpipelined model, the way `test_kimi_k3_pp_exact_block_grads.py` does for titan. This is the gap in 6840's test plan and the most useful contribution.
3. Offer the lost-gradient count check (section 3c item 3) as the cache-side contribution, backed by a test where a consumer chunk's backward is skipped.
4. Payload layout, padding, stage splits and packing under PP are Megatron's P2P and schedule design and are not proposed from the titan side.
5. Bridge: the K3 provider's own AttnRes path refuses VPP. Once 6840 is in MCore, the useful Bridge change is to build the K3 provider on MCore's AttnRes instead of its own layers (retiring the hidden-dim bank packing), which unblocks VPP there; Bridge issue 4910's VPP item is where that would be raised.
6. Local prototype work, if any, goes on the fork submodules only and starts from the 6840 head, not from a fresh port.

## 6. Not verified

- 6840's VPP unit tests were read, not run; no GPU run of 6840.
- ilml/Attention-Residuals (named in MLM issue 4016 as heading for MBridge) and radixark/miles 1825 were not read.
- Where PR 6427's cache lives.
- Behaviour of full-iteration CUDA graphs with either design.
