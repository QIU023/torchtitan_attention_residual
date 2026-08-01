# LoRA carries a TP gradient defect that step-0 testing cannot see

## The measurement

`kimi_linear_k3mini_qlora`, dp1, pure TP, shared checkpoint, only tp varies.
Ratio is tp1/tpN per parameter, so >1 means the TP gradient is too SMALL.

  parameter                                      /tp2      /tp4
  layers.20.self_attn.o_proj.lora_b            2.2726    3.2584
  layers.3.self_attn.o_proj.lora_b             1.3413    2.0086
  layers.11.self_attn.o_proj.lora_b            1.4444    1.9957
  layers.7.self_attn.o_proj.lora_b             1.2629    1.8370
  layers.15.self_attn.o_proj.lora_b            1.6625    1.8207
  layers.20.self_attn.kv_b_proj.lora_b         1.0043    1.8019

  max |ratio-1| over all params   tp2 1.27261   tp4 2.25842
  median                          tp2 0.02046

Monotonic growth with tp degree, concentrated on `o_proj.lora_b` -- the rowwise
case. That is the same signature as the two defects already fixed
(block_attn_res, exactly 1/tp; moe_sharding in_grad_placements, ~sqrt(tp)), and
the direction says a reduction is missing rather than doubled.

## Why the earlier check missed it

A first pass on this flavor from a cold seed showed lora_b within 1.5% and looked
clean. It was measuring nothing: LoRA initializes B to zero, so at step 0
grad_A = grad_out @ B^T is exactly zero and the adapter contributes nothing to the
loss. Every lora_a gradient was 0.0000 in that run, which should have been the
tell.

This run loads a checkpoint warmed for 3 steps first. All 139 lora_a parameters
then have nonzero gradients, and the defect appears.

Generalize: any adapter or gate that is zero-initialized is invisible to a
step-0 numerical check. The AttnRes pseudo-queries are zero-init too, and while
their multi-layer behaviour was checked, it was checked from cold seeds
throughout.

## Where to look

`lora.py` requests `to_local(grad_placements=(Partial(),))` for the replicated
operand of each branch -- lora_a under colwise, lora_b under rowwise -- and the
docstring justifies this by citing "the same trap as the attn_res pseudo-query
note". That note was later disproven: block_attn_res's Partial() request was
itself the bug, over-reducing by exactly tp. The justification is therefore
unsound even where the conclusion may be right, and the rowwise lora_b path
measures wrong.

Not yet diagnosed: whether `_tp_style` and `_tp_mesh` are set as the branch
assumes for o_proj, whether the Partial request fires at all, and why kv_b_proj
is affected at tp4 but not tp2. Measure before changing, as with the others.

## Diagnosed: the explicit TP path never runs

Runtime inspection under tp2:

  [LORA] KimiLoRALinear style=None mesh=False
         x=DTensor(Replicate(),) lora_a=DTensor(Replicate(),) lora_b=DTensor(Shard(0),)

`_tp_style` is None and `_tp_mesh` is unset, so the entire colwise/rowwise branch
in `KimiLoRALinear.forward` -- the one carrying the explicit
`to_local(grad_placements=(Partial(),))` calls and the docstring about making the
tp reductions happen -- is dead code in this configuration.

Both attributes are assigned in exactly one place (lora.py:287-288), inside the
function that distributes a **packed-MXFP4** base. So the hand-written TP handling
only activates when the base is packed. Plain LoRA and QAT LoRA under TP fall
through to generic DTensor dispatch on parameters that were distributed
elsewhere (lora_a Replicate, lora_b Shard(0)), and that fallback is what measures
wrong: o_proj.lora_b at 2.2726 (tp2) and 3.2584 (tp4).

So the earlier worry -- that lora.py's `Partial()` requests repeat the
block_attn_res over-reduction -- was aimed at code that never executes. The real
defect is the opposite: the intended reduction is absent because the path that
would perform it is gated behind packed-MXFP4.

Two things to establish before fixing, neither done yet:

1. Whether the packed-MXFP4 path (where `_tp_style` IS set) measures correct. If
   it does, the fix is to set the attributes for the non-packed case too. If it
   does not, the `Partial()` requests need re-deriving first -- and they cite a
   justification that was disproven.
2. Whether `lora_b` being Shard(0) is right for a rowwise layer. The probe caught
   one module; o_proj is rowwise and its B should shard on the input axis, not
   dim 0. If the distribution itself is wrong, no grad_placement fixes it.
