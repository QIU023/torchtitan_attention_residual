# CP/TP equivalence: the instrument exists, and sequence sharding is bit-exact

Item B of the overnight plan. The gap was never a known bug -- it was that no instrument
could attribute a difference, because every arm compared a sharded run against a reference
of a DIFFERENT SHAPE, and `chunk_kda` is a chunked scan whose blocking follows the shape.
That confound is what produced three wrong attributions (LoRA, AttnRes, KDA) of the same
band.

## The instrument, and it is fla's not mine

`fla.ops.kda.naive.naive_recurrent_kda` is token-by-token with no chunk size, so it is
shape-independent by construction. Using their reference rather than rederiving the
recurrence removes the obvious way to be wrong.

**Validated before use**, which was the precondition I set:

    seq=  16  rel 1.411e-03
    seq=  64  rel 1.748e-03
    seq= 128  rel 1.625e-03
    seq= 256  rel 1.663e-03

Flat in sequence length, so this is the kernel-vs-reference floor in float32, not a
divergence. Any CP/TP claim has to sit above that floor to mean anything -- and the
recorded CP deviation was 0.16, two orders above it.

**The first attempt was vacuous and is worth recording.** Driving it with `g` near zero
(no decay) and unnormalised q/k makes the recurrent state explode to ~1e17; both
implementations then return garbage that agrees to 1e-2, which reads like a passing test.
KDA l2-normalises q/k and its gate is `-exp(A_log) * softplus(...)`, strongly negative. An
instrument test needs the operating point, not just the API.

## Result: splitting the sequence is bit-exact

CP shards the sequence and carries the recurrent state across the boundary, which
`chunk_kda` expresses directly via `initial_state` / `output_final_state`. So this needs no
process group -- one GPU, no collectives, no confounds:

    reference magnitude        0.16412
    full        vs reference   rel 1.663e-03
    two shards  vs reference   rel 1.663e-03
    full        vs two shards  rel 0.000e+00

**Two shards agree with one call bit-for-bit**, and both sit at the same distance from the
shape-independent reference. So the scan itself does not care how the sequence is divided,
provided the state crosses.

## What this does and does not settle

**Settled.** Sequence division is not a source of error in KDA. That was the most plausible
mechanism behind the CP band, and it is now excluded on a single GPU with no confounds.
Combined with the already-verified per-head independence of `chunk_kda` (PR-D body), both
CP forms this repo has -- Ulysses, which gives each rank the full sequence for a head
subset, and KCP, which keeps the sequence sharded -- are exact at the KERNEL level.

**Not settled.** Our CP and TP PLUMBING around the kernel: `_cp_all_to_all_headseq`, the
conv halo, the TP head slice and the `o_proj` reduction. Those are our code, not fla's, and
this probe does not touch them. What it changes is that a future comparison can now use a
shape-independent reference for both arms, so a difference there is attributable instead of
confounded.

**Next.** Drive one KDA layer under cp2/tp2 through the real trainer, materialise the
gradients, and compare against `naive_recurrent_kda` on one rank. The instrument for that
comparison now exists and is validated; that is the part that was missing.

## Repro

    python phase13_k3like_48b_posttrain/matrix_scripts/kda_shape_independent_probe.py
