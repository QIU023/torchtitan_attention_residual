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
