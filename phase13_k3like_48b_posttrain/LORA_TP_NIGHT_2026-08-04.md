# LoRA TP: baseline re-established, and one hypothesis killed cheaply

Working branch `lora_tp_overnight` off `303ce6a70`. **No code changed** --
`attention_residual_dev` is untouched.

## Baseline on the current tree

`kimi_k3_mini_diag_4l_mla_lora`, dp1 x tp2, 1 step, seed 42, deterministic:

    26 replicated-gradient parameters checked, 11 DISAGREE across ranks

    rel spread   |g|max        parameter
        2.9638   0.0120607     layers.3.mlp_res_proj.weight
        2.8230   0.0133937     layers.2.mlp_res_proj.weight
        2.2057   0.0117754     layers.3.attn_res_proj.weight
        1.3208   0.0327093     layers.0.mlp_res_proj.weight
        0.9347   0.0322153     final_attn_res_proj.weight
        0.9150   0.000487603   layers.1.self_attn.q_a_proj.lora_b
        0.7464   0.00834052    layers.1.mlp_res_proj.weight
        0.7272   0.000226735   layers.3.self_attn.q_a_proj.lora_b
        0.6133   0.00820403    layers.2.attn_res_proj.weight
        0.5561   0.0234914     layers.1.attn_res_proj.weight
        0.5089   0.000370578   layers.2.self_attn.q_a_proj.lora_b

Eight of the eleven are AttnRes projections, which are clean on the same dense
model with LoRA off and which aggregate across every layer -- downstream
victims. The root is `q_a_proj.lora_b`, on the three layers that have one.

Note the count differs from `LORA_TP_DIAGNOSIS_2026-08-04.md` (62 checked / 22
disagreeing). That measurement was on an older tree; this is the baseline for
the current one, and improvements have to be judged against this.

## Killed before spending the night on it: is the probe reporting Partial?

A Partial gradient legitimately differs per rank -- it is summed later -- so a
probe that compared local values without checking placement would report a
defect that is not one. Checked the probe rather than assuming:

    if not placements[axis].is_replicate():
        continue

It skips anything not Replicate on the TP axis. So the eleven have gradients
**labelled Replicate while holding different values per rank**, which is a
genuine defect: somewhere a Partial gradient is being relabelled Replicate
without the reduction.

## Also killed: "the adapters are lifted into the mesh by from_local"

`from_local` with a Replicate placement has `to_local` as its backward, which
does not reduce -- that would explain everything. It is not what happens.
`parallelize.py` distributes the adapters properly at setup:

* colwise: `lora_a` Replicate, `lora_b` Shard(0)
* rowwise: `lora_a` Shard(1), `lora_b` Replicate
* leftover (NoParallel, which is where `q_a_proj` lands): both Replicate

So `q_a_proj.lora_b` is already a Replicate DTensor before forward runs, and
the ad-hoc lifting in `LoRALinear.forward` never fires for it.

## Where this leaves the search

For `q_a_proj` everything stays DTensor end to end: `x` is Replicate,
`base_out = self.base(x)` is a DTensor, both adapters are DTensors, and the
final locality-matching block does not fire because `base_out` is not plain.
So the mislabel is not in the tensor-kind juggling that the previous two fix
attempts targeted -- which is consistent with both of those attempts having
failed.

The gradient arriving at `q_a_proj`'s output is Partial: its consumer
`q_b_proj` is colwise, and a colwise op's gradient with respect to a replicated
input is a per-rank contribution. The open question is which operation turns
that Partial into a Replicate-labelled gradient without reducing it.

**Next probe** (not a fix): dump `.grad.placements` at each stage of the adapter
chain -- after the inner `F.linear(x, la)`, after the outer one, and after the
add with `base_out` -- and find the step where Partial becomes Replicate. That
is a measurement, unlike the two previous attempts, both of which changed code
first and measured second.

## veRL

Not started. Recording that rather than leaving it to be inferred from silence.


---

# The localization was wrong, and the measurement that showed it

Everything above treats `q_a_proj.lora_b` as the root and looks for a
Partial-to-Replicate mislabel inside `LoRALinear.forward`. Both premises are
wrong.

## What the gradient chain actually does

Placement dump along the Q and KV compression paths, layer 0:

    q_b_proj          grad_out S(2)      grad_in P(sum)
    q_a_layernorm     grad_out P(sum)    grad_in R
    q_a_proj          grad_out R         grad_in R

The layernorm converting Partial to Replicate looked like the bug. It is not --
it is a real reduction, not a relabel. Values at that boundary:

    q_a_layernorm grad_out (Partial)   rank0 0.000193   rank1 0.000178   differ, correctly
    q_a_layernorm grad_in  (Replicate) rank0 0.000596   rank1 0.000596   identical
    q_a_proj      grad_out (Replicate) rank0 0.000596   rank1 0.000596   identical

So the gradient arriving at `q_a_proj` is correctly reduced. The same dump also
explains why `kv_a_proj_with_mqa` keeps a Partial gradient where `q_a_proj` does
not: its output is split, the rope half contributes a Partial gradient, and
`R + P(sum)` stays Partial. It is rescued by the split rather than handled.

## The measurement that broke the story

`q_a_proj.lora_b` gradients, per layer, both ranks:

    layer 0   rank0 0.01897978   rank1 0.01897978   identical
    layer 1   rank0 0.00397344   rank1 0.00267326   differ
    layer 2   rank0 0.00266132   rank1 0.00298768   differ
    layer 3   rank0 0.00168546   rank1 0.00153858   differ

**Layer 0 is clean.** Every placement measurement above was taken on layer 0 --
the one layer that is correct -- which is why it kept showing a healthy chain
while the rank-spread probe kept reporting a defect. Two measurements that
looked contradictory were describing different layers.

## Where this points instead

Layer 0 differs from layers 1+ in exactly one way: it has no incoming block
residuals. Layers 1+ take their input from the AttnRes cross-layer aggregation.
So the suspect is that aggregation path, not LoRA's TP handling, and the
disagreeing set fits: three `q_a_proj.lora_b` on layers 1-3 (not 0) plus eight
AttnRes projections.

It also explains both earlier fix attempts. Both edited
`LoRALinear.forward` -- declaring `grad_placements` on its `full_tensor`, and
making the packed TP path reachable. Neither could work, because the input
arriving at the module is already rank-dependent.

Note the constraint this has to satisfy: with LoRA off, the same architecture is
clean (40 replicated gradients checked, all agree). So the AttnRes aggregation
is not unconditionally wrong -- something about the LoRA-wrapped layers changes
what it carries. That is the next thing to measure, on layer 1, not layer 0.

---

# Retraction: the forward does not diverge

The section above concluded that the forward is rank-dependent, citing
`block_attn_res` outputs of 54.26 against 55.48 and an o_proj adapter delta of
-26.56 against -23.43. **Both readings are wrong, in the same way.**

`lora_b` is zero-initialised, which is standard LoRA:

    rank0  lb  ['R']  shape=[512, 8]  norm=0.000000
    rank1  lb  ['R']  shape=[512, 8]  norm=0.000000

So at step 0 the adapter contributes exactly nothing to the forward, and the
forward cannot differ from the LoRA-off case, which is clean. The traced chain
confirms it end to end: `inner` is `P(sum)` with a real norm, `lora_out` is
`P(sum)` with norm 0, `full_tensor()` gives norm 0.

The "delta" came from calling `self.base(x)` a second time inside the probe and
subtracting, which is not the adapter's contribution. And the differing
intermediate norms are **per-rank plain tensors** -- `o_proj`'s input is the
local attention-head shard, held plain by design -- so they are supposed to
differ. Comparing them across ranks measures nothing.

That is the same error as flagging `S(2)` tensors as divergent earlier in the
same session: a sharded or per-rank-local value differing across ranks is
correct behaviour, and a probe that reports it as a defect is measuring the
wrong thing. Twice in one session.

## What survives, and it is still useful

Three measurements stand:

1. `q_a_proj.lora_b`'s gradient is **identical across ranks on layer 0 and
   differs on layers 1, 2 and 3**. Direct per-parameter comparison, not
   inferred.
2. With LoRA off, the same architecture is clean: 40 replicated gradients
   checked, all agree.
3. The gradient chain through `q_a_layernorm` reduces correctly -- Partial
   output-grad differs per rank (0.000193 / 0.000178), Replicate input-grad is
   identical (0.000596 both).

So the defect is real, is in the backward, and is layer-dependent in a way that
tracks the presence of incoming block residuals. What is NOT established is any
claim about the forward.

## The next measurement, stated so it cannot repeat the error

Compare only quantities that are supposed to be equal across ranks: gradients
whose placement is Replicate on the TP axis, and nothing else. Specifically,
walk the backward of layer 1 and find the first Replicate-labelled gradient
whose values differ, then check whether the same point on layer 0 is clean. Do
not compare plain per-rank tensors or Shard-placed locals; they differ by
design.
