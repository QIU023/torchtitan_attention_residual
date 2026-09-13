# Reply to Tianyu on PR 4312: activation save / release policy (comment 3976976576, `model.py` line 242)

For the user to post as the reply to 3976976576 (Tianyu's follow-up, 2026-09-10; nothing posted in that thread yet). Checked on 2026-09-13 against the PR head `k3_pp_text` = `pp_review4` = `dbc425403`: `KimiK3TransformerBlock.forward` and `_apply_attention_residual` in `kimi_k3/model.py`, `SelectiveAC` / `_get_default_save_ops` in `distributed/activation_checkpoint.py` (the PR base already has torch_remat, #4497; the per-op SAC is unchanged), `parallelize_kimi_k3` (`ac_policy.apply(model)`: one region per layer), `PPRankLocalCache` use in `kimi_k3/pipeline_stage.py`. Derived from the code, not measured; a memory trace of the 2 x 4 example can follow if asked. The attention-residual checkpoint wrapper on the integration branch is NOT in this PR and is not described.

--- PASTE BEGIN ---

Close, with one correction: the redundancy is per block, not per layer. Each block starts a new stack that holds every previous block's result, so the stacks of consecutive blocks overlap (that is the N(N+1)/2); but the layers inside a block share that one stack tensor, and no layer saves a copy of its own. The per-layer residual reads are recomputed in backward. This is the policy of this implementation (the default `SelectiveAC` of the debug flavors); Kimi has not published theirs.

What each layer does (`KimiK3TransformerBlock.forward`, `_apply_attention_residual`):

- The first layer of a block appends its input `x` to the stack: `S = torch.cat((S, x.unsqueeze(1)), dim=1)`, a new `[T, n, D]` tensor. The other layers of the block take and return that same tensor object.
- Every layer then reads the stack twice, before attention and before the FFN: `values = S` (first layer of a block, attention read) or `cat(S, partial)` (every other read), `.float()`, RMS-normalised, scored against the layer's projection, softmaxed over the entries, and a batched matmul gives the input. All of these are temporaries.
- `SelectiveAC` wraps each layer in one checkpoint region with a per-op policy: it saves the outputs of the linears (every second one recomputed), flex attention, topk / max and the communication ops, plus the region's inputs. The read's ops (`cat`, the bf16 -> fp32 copy, pow / mean / rsqrt, mul / sum, softmax, bmm) are not in that set, so their outputs are freed right after the read and recomputed in that layer's backward. The region input `S` is a reference to the block's stack, which exists once.

Your example, 2 blocks of 4 layers, one micro-batch of `T` tokens, hidden `D`:

| forward step | lives until backward (beyond any decoder's per-layer saves) | temporaries, freed after the read |
| --- | --- | --- |
| embedding | `e` `[T, D]` | |
| layer 0 (starts block 1) | `S1 = cat(empty, e)` `[T, 1, D]` | no attention read (its input is `e`); FFN read `cat(S1, h0)` `[T, 2, D]` + fp32 copy |
| layers 1-3 | nothing new (they pass `S1` along) | two reads each: `cat(S1, partial)` `[T, 2, D]` + fp32 copy |
| layer 4 (starts block 2) | `S2 = cat(S1, x4)` `[T, 2, D]`, `x4` = block 1's result | attention read `S2` + fp32 copy `[T, 2, D]`; FFN read `cat(S2, h4)` `[T, 3, D]` + fp32 copy |
| layers 5-7 | nothing new (they pass `S2` along) | two reads each: `cat(S2, partial)` `[T, 3, D]` + fp32 copy |
| output aggregation | | `cat(S2, x8)` `[T, 3, D]` + fp32 copy |

So the stacks add `S1` and `S2`, 1 + 2 = 3 rows of `[T, D]`, on top of what any decoder keeps per layer. For N blocks that is N(N+1)/2 rows -- 36 for the 93-layer model's 8 blocks -- independent of how many layers a block has.

Release: backward runs the aggregation, then layers 7 down to 0. Each layer's backward recomputes its two reads from the saved references (`S` and the layer input), uses them and frees them. `cat` saves nothing for its backward, so a stack is freed as soon as the last layer that took it as input has run backward: `S2` after layer 4's backward, `S1` after layer 0's.

Under PP, each rank additionally keeps (detached) views of the blocks a later stage on that rank will read, and drops them after the rank's last stage has run forward for that micro-batch; they point into the same stack storage, so they add no memory while the stack is alive anyway.

With activation checkpointing off, autograd would instead save every read's fp32 `values` and normalised keys, two reads per layer -- effectively a copy of all previous blocks per layer, the redundancy your question describes. Selective checkpointing is what keeps it at one stack per block.
