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

## The gating is deeper than the attribute -- first fix attempt was a no-op

Setting `_tp_style`/`_tp_mesh` from the plan's placements changed nothing: the
numbers came back bit-identical (o_proj.lora_b 2.2726 / 3.2584, all-param max
1.27261 / 2.25842, median 0.02046). Reverted rather than left in as a no-op.

The reason is in `KimiLoRALinear.forward`:

    if self._quantize_base == "nf4":
        ...
    elif self._quantize_base == "mxfp4":
        if getattr(self, "_tp_style", None) is not None:
            return self._forward_packed_tp(x)

`_forward_packed_tp` is reachable only when the base is **mxfp4**. The attribute
is a second gate inside that branch, not the gate. So the explicit TP handling --
the colwise/rowwise split and every `grad_placements` in it -- exists for exactly
one configuration: an mxfp4-packed base. nf4 and unquantized LoRA under TP have
no explicit handling at all, and that is the configuration measuring wrong.

The parameter distribution is not the problem. Read off at runtime it is exactly
right for every projection type:

  colwise (gate_proj, up_proj, q_b_proj, kv_b_proj, attn_gate_proj)
      lora_a Replicate       lora_b Shard(0)
  rowwise (down_proj, o_proj)
      lora_a Shard(1)        lora_b Replicate
  NoParallel (latent down/up, q_a_proj, kv_a_proj_with_mqa)
      both Replicate

Rowwise is where the defect shows, and the shape of it follows from that layout:
lora_b is Replicate while each rank computes its contribution from its own input
shard, so grad_b is Partial across tp and needs an all-reduce that nothing
performs. That is why o_proj.lora_b, and not the colwise adapters, is what moves.

So the fix is not to enable an existing path but to give the non-packed path the
same treatment the packed one already has. That is a real change to a forward
that three configurations share (nf4, mxfp4, unquantized), so it needs deriving
and then verifying on the warm-checkpoint instrument per configuration -- not a
one-line enable. Not attempted here.

## Second fix attempt also a no-op -- the DTensor branch is not taken at all

Adding `grad_placements=(Partial(),)` to the `full_tensor()` call at the end of
the adapter path changed nothing (max |r-1| identical at 1.27261 / 2.25842,
median 0.02046 -> 0.02064). Reverted.

That call sits inside `if isinstance(lora_out, DTensor) and not
isinstance(base_out, DTensor)`. For it to matter, `lora_out` has to be a DTensor,
which requires `x` to be one. The forward branches on exactly that:

    x_is_dt = isinstance(x, DTensor)
    ...
    if x_is_dt:      # keep adapters as DTensors
    else:            # la = la.to_local(); lb = lb.to_local()

If `x` arrives PLAIN at o_proj -- which is what `use_local_output=True` on the
producing module gives -- then both adapters are unwrapped to their local shards
and `F.linear(F.linear(x, la), lb)` runs entirely in plain-tensor land. For a
rowwise layer `la` is Shard(1), so that product is each rank's PARTIAL
contribution, and being a plain tensor there is nothing left to all-reduce it.
RowwiseParallel's all-reduce covers the base weight only; the adapter rides
outside it.

That is consistent with everything measured: the adapter contributes one rank's
share instead of the sum, so `grad_b` is short by roughly tp, monotonically, and
only on rowwise layers (o_proj, down_proj) -- colwise adapters are Shard on the
output axis and need no sum.

NOT yet confirmed: whether `x` is in fact plain at o_proj. That is one print and
should be the next thing measured -- both previous fixes were aimed at code paths
that turned out not to run, and the way to stop repeating that is to confirm
which branch executes before changing any of them.

If confirmed, the fix is not a grad_placements argument anywhere: it is that the
plain-tensor adapter path under rowwise TP needs its own all-reduce, or the
adapters must stay DTensors through the matmul so DTensor performs it.

## The plain-x hypothesis is also wrong; the gradient is mislabelled Replicate

`x` is a DTensor at every LoRA site, so the `x_is_dt` branch is taken and the
adapters stay DTensors through the matmul:

  gate_proj / up_proj (colwise)   x=R        lora_a=R      lora_b=S(0)
  down_proj / o_proj  (rowwise)   x=S(2)     lora_a=S(1)   lora_b=R
  latent down/up      (NoParallel) x=R       lora_a=R      lora_b=R

Hooking `o_proj.lora_b`'s gradient directly, over three successive steps from a
warm checkpoint:

  tp1   0.01480148  0.01612288  0.01783790   placement: plain
  tp2   0.01066847  0.01178489  0.01376823   placement: Replicate()

Ratio 1.387, against sqrt(2) = 1.414. That is the same signature as the
moe_sharding defect: the gradient is mathematically Partial across tp -- each
rank holds one near-orthogonal share -- but it is labelled Replicate, so DTensor
treats it as already complete and the sum never happens.

Three hypotheses have now been measured and killed:

  1. lora.py's Partial() requests over-reduce, like block_attn_res  -- that code
     never runs (gated on _quantize_base == "mxfp4").
  2. full_tensor() lacks grad_placements                            -- adding it
     changed nothing.
  3. x arrives plain so the adapters compute outside DTensor        -- x is a
     DTensor everywhere.

What is established: the value is right (loss matches), the parameter
distribution is right, the forward branch taken is the DTensor one, and the
gradient reaching lora_b under rowwise TP is short by ~sqrt(tp) while carrying a
Replicate label.

So the remaining question is narrow: on the path from `lora_out` back to
`lora_b`, where does a Partial gradient acquire a Replicate label? The two
candidates left are the `base_out + scaling * lora_out` add (if base_out is a
DTensor the full_tensor branch is skipped entirely and the add is Replicate +
Partial), and the parameter-gradient accumulation itself, which defaults a
Replicate parameter's grad to Replicate. Print the placements of `base_out` and
`lora_out` at the add before touching either.
