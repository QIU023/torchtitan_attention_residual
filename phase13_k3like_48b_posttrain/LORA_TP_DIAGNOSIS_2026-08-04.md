# LoRA under TP: the defect localized, and one hypothesis ruled out

Refs: pytorch/torchtitan#3029

Supersedes the attribution in `LORA_TP_DEFECT_2026-07-31.md`, which was measured
on an MoE flavor and therefore could not separate the defect from routing.

## Established

Instrument: shared warm checkpoint, one varied dimension, and for the
cross-rank check `replicated_grad_rank_probe.py`, which compares each
tp-replicated gradient across the ranks that are supposed to hold identical
copies of it.

| model | replicated gradients checked | disagree across TP ranks |
|---|---|---|
| dense, no LoRA (`diag_4l_mla`) | 43 | **0** |
| dense, **+ LoRA** (`diag_4l_mla_lora`) | 62 | **22** |
| MoE, full-param (`k3mini_block_attn_res`) | 503 | 0 |

Relative spread on the 22 runs from 0.85 to 2.40. Direction, same flavor,
tp1 vs tpN:

| comparison | worst 1-cos |
|---|---|
| tp1 vs tp1 (control) | 4.77e-07 |
| tp1 vs tp2 | 1.06 |
| tp1 vs tp4 | 1.45 |

`1-cos > 1` means the gradients are anti-correlated, not merely different, and
the affected parameters carry healthy magnitudes (`ffn.down_proj.lora_b` at
4.9e-2 against a model median near 2.5e-3), so this is not the small-magnitude
cancellation that produced the earlier false positives.

**The forward is not implicated.** At tp2 all four ranks report an identical
loss and grad_norm (7.60776 / 4.5861). The divergence is entirely in backward.

**LoRA is the trigger, but not the only victim.** Of the 22, several are not
LoRA parameters at all: `final_attn_res_proj/norm`, `mlp_res_proj/norm`,
`attn_res_proj/norm`. They are clean on the same dense model with LoRA off. The
AttnRes projections consume the block-residual stream built from layer outputs,
so a partial gradient anywhere upstream reaches them.

This also explains why every earlier measurement was ambiguous: on an MoE
flavor, this defect and MoE's route flipping are superimposed, and neither a
norm ratio nor a cosine can separate them.

## The structural finding

`LoRALinear` has a TP-aware forward, `_forward_packed_tp`, whose whole purpose is
to get these gradients right -- it declares `grad_placements` on each replicated
operand, with a docstring explaining which reduction each one needs.

It is unreachable for plain LoRA.

`_tp_style` and `_tp_mesh` are assigned in exactly one place: the branch that
distributes `base_qdata` / `base_scale`, i.e. the packed-MXFP4 base. And
`forward` dispatches to it only under `self._quantize_base == "mxfp4"`. A LoRA
config that sets `lora_rank` without quantization -- which is the ordinary case,
and the one veRL will drive -- never sets `_tp_style`, never enters
`_forward_packed_tp`, and falls through to a generic path that was written to
fix tensor-kind mismatches (DTensor vs plain) rather than gradient placements.

So the TP gradient handling exists only for the quantized variant. That is the
shape of the defect, and it is consistent with every measurement above.

## Ruled out

`lora_out.full_tensor()` at the end of the generic path carries no
`grad_placements`, which violates this repo's own rule ("For any usage of
`DTensor.to_local`, specify explicitly the `grad_placements` ... This includes
the indirect calls from ... `full_tensor`"). It looked like the answer.

Declaring `grad_placements=(Partial(),)` there **does not fix it**: the count
stays at 22 and the magnitudes shift only slightly (`q_a_proj.lora_b` 7.50e-5 ->
6.00e-5). The change has been reverted rather than left in as a plausible-looking
non-fix.

So the missing reduction is not on that single boundary. The next candidates,
in order: whether the adapter product is Partial at all in the generic path
(if `la`/`lb` were unwrapped to plain tensors before the matmul, DTensor owns
nothing and there is no reduction to declare), and whether the `elif` branch
that lifts a plain `x` into the adapter mesh leaves `lora_b`'s gradient partial
with no owner.

## Two fix attempts, both reverted

**Attempt 1: declare `grad_placements` on the generic path's `full_tensor()`.**
Ruled out above -- count stays 22.

**Attempt 2: make the TP-aware forward reachable for a plain base.** Set
`_tp_style`/`_tp_mesh` in the plain-LoRA distribution loop and let `forward`
dispatch there. The path is reached and immediately fails on
`aten.mul.Tensor got mixed torch.Tensor and DTensor`, for two reasons that
compound:

1. **The style is three-valued, not two.** `lora_tp` carries a single
   `is_colwise` bool, and `_forward_packed_tp` branches on it. The probe shows
   three layouts in play:

   | projection | x | lora_a | lora_b | style |
   |---|---|---|---|---|
   | `q_a_proj`, `kv_a_proj_with_mqa` | Replicate | Replicate | **Replicate** | NoParallel |
   | `q_b_proj`, `kv_b_proj`, `attn_gate_proj` | Replicate | Replicate | Shard(0) | colwise |
   | `o_proj` | plain | Shard(1) | Replicate | rowwise |

   Forcing the replicated case into either of the other two mixes DTensor with
   plain tensors at the matmul -- and the replicated case is where two of the
   worst offenders live.

   Adding a third `replicated` style (classified in the loop that currently
   replicates leftover adapters) fixes that particular crash.

2. **Output locality is not the module's to decide.** With the third style in
   place the failure moves to `model.py:677`,
   `attn_out * self._attn_gate(x, ...)`: `attn_gate_proj` is colwise, so
   `_forward_tp` returns `DTensor(Shard)`, while the call site builds
   `attn_out` in plain-tensor land. The generic path never had this problem
   because it derives locality from `base_out = self.base(x)` -- the base goes
   through the TP plan's own hooks, which know whether that site wants
   `use_local_output`. Its comment says so explicitly: "which of the two ways
   depends on the style, and getting it backwards is a shape error, not a
   silent one".

So the correct shape of the fix is: `_forward_tp` must call `self.base(x)`
rather than doing its own local matmul, and match the base's locality on the
way out -- keeping the explicit `grad_placements` on the adapter operands,
which is the part the generic path lacks. That trades the packed path's local-
matmul optimization for correctness, and needs the packed callers re-verified.

Both attempts are reverted; the tree runs clean (dense LoRA tp2, 7.63126).

## Not done

The defect is localized and reproducible but **not fixed**. What exists is the
diagnosis, a dense control flavor (`kimi_k3_mini_diag_4l_mla_lora`) that isolates
it from MoE routing, and two probes that detect it in seconds.

Consequence for the parallelism claims: full-param TP/CP are verified at the
gradient level on dense (1.06e-4 and 1.87e-5 worst direction disagreement).
**LoRA under TP is not**, and should not be presented as ready.
