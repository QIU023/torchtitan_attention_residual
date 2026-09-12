# Reply to Tianyu on PR 4312: activation save / release policy (comment 3976976576, `model.py`, 2026-09-10)

For the user to post, in the thread under 3922719284. Checked against `pp_review4` = `k3_pp_text` = `dbc425403`: `KimiK3TransformerBlock.forward` and `_apply_attention_residual` in `kimi_k3/model.py`, `SelectiveAC` and `_get_default_save_ops` in `distributed/activation_checkpoint.py`. Derived from the code, not measured; a memory trace of the 2 x 4 example can follow if asked. We describe our implementation; Kimi's own training policy is not public.

--- PASTE BEGIN ---

Not quite: nothing is saved per layer. One stack tensor is created per block and shared by every layer of that block, and the per-layer attention-residual reads are recomputed in backward under activation checkpointing. So the redundancy is one stack per block, not one per layer.

What the code does (`KimiK3TransformerBlock.forward`, `_apply_attention_residual`):

- The first layer of a block concatenates its input `x` onto the stack: `stack = torch.cat((stack, x.unsqueeze(1)), dim=1)`, a new `[T, n, D]` tensor. The other layers of the block receive and return that same tensor object; no copy.
- Every layer reads the stack twice, before attention and before the FFN: `values = cat(stack, partial)` (`[T, n+1, D]`), cast to fp32, RMS-normalised, scored, softmaxed over the n+1 entries, and a weighted sum gives the input. These are temporaries.
- With the default `SelectiveAC`, each layer is one checkpoint region that saves only the outputs of the ops in its save set (linears -- every second one --, flex attention, topk, the MoE collectives) plus the region's inputs. The read's ops (`cat`, the fp32 cast, pow/mean/rsqrt, mul/sum, softmax, the batched matmul) are not in that set, so their tensors are freed right after the read and recomputed in the layer's backward. The region's `stack` input is a reference to the block's stack, which is alive anyway.

Your example, 2 blocks of 4 layers, per micro-batch, `T` tokens, hidden `D`:

| step of forward | new tensor that lives until backward | temporaries (freed after the read) |
| --- | --- | --- |
| embedding | `e` `[T, D]` | |
| layer 0 (block 1 starts) | `S1 = cat(empty, e)` `[T, 1, D]` | FFN read: `cat(S1, h)` `[T, 2, D]` + fp32 copy |
| layers 1-3 | none (they pass `S1` along) | two reads each: `cat(S1, partial)` `[T, 2, D]` + fp32 copy |
| layer 4 (block 2 starts) | `S2 = cat(S1, x4)` `[T, 2, D]` | reads `[T, 3, D]` + fp32 copy |
| layers 5-7 | none (they pass `S2` along) | two reads each, `[T, 3, D]` + fp32 copy |
| output aggregation | | `cat(S2, h8)` `[T, 3, D]` + fp32 copy |

On top of what any decoder keeps (each layer's input `h` and its saved linears / attention output), the stacks add `S1` and `S2`: 1 + 2 = 3 rows of `[T, D]`. For N blocks that is N(N+1)/2 rows -- 36 for the 93-layer model's 8 blocks -- and it does not grow with the layer count inside a block.

Release: backward runs layer 7 down to 0. Each layer's checkpoint recomputes its reads (rebuilding `cat(stack, partial)` from the saved stack reference), uses them and frees them. `S2` is last needed by layer 4's backward and is freed then; `S1` after layer 0's and the block-2 `cat`'s backward. Under PP the rank-local cache additionally holds a reference to each block a later stage on the rank will read, and drops it after the rank's last stage forward for that micro-batch; the autograd graph keeps what backward needs, as above.

With activation checkpointing off, the answer changes: autograd would save the fp32 `values` (and the normalised keys) of every read, two reads per layer, i.e. a copy of all previous blocks per layer -- the redundancy your question describes. That is what selective checkpointing avoids here.
