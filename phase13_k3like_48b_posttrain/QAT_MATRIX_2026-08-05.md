# MXFP4/MXFP8 QAT through the parallelism matrix

K3 is post-trained with QAT running the whole time (report sec 4.1.4: MXFP4
routed-expert weights, MXFP8 expert activations). Every matrix before this one
ran bf16, so nothing on record said the parallelism work holds in the precision
K3 is actually post-trained in. This runs it.

Two defects fell out. One would have invalidated this matrix; the other changes
what the numbers mean.

## Protocol

Flavor `kimi_k3_debugmodel_report_arch_qat`: `kimi_k3_debugmodel_report_arch`
with `mxfp4_qat = True` and nothing else changed, so any per-cell difference is
attributable to the fake-quant wrapper rather than to the model.

One step-0 seed checkpoint, written from the **bf16** flavor and loaded by both.
That is sound rather than convenient: QAT is fake-quant with a bf16 master, so it
adds no parameters -- verified, not assumed, by building both on meta and
comparing state-dict keys (406 each, symmetric difference empty). The same
property is why QAT needs no change in the checkpoint path, the HF adapter, or
veRL weight sync.

Seed load verified by value rather than by log line: the bf16 flavor from this
seed gives step-1 **12.07418**, bit-identical to the published seeded matrix, so
the seed is the same distribution as the one those numbers came from. QAT from
the same seed gives **12.06460**.

    torch 2.14.0.dev20260802+cu130, fla-core 0.5.1, 8 x RTX 5060 Ti
    FLAVOR=kimi_k3_debugmodel_report_arch_qat OUT=... STEPS=10 \
      EXTRA="--checkpoint.interval 100000 \
             --checkpoint.initial_load_path /workspace/qat_seed/checkpoint/step-0" \
      bash matrix_scripts/run13_flav.sh

## Defect 1: multimodal flavors silently dropped LoRA, Muon and QAT

`KimiK3MultimodalSpec` overrides `build` to construct the vision-bearing model
and returned it directly, so the block in `KimiK3Spec.build` that attaches LoRA,
tags Per-Head Muon and applies MXFP4 QAT never ran on any multimodal flavor. No
error, no warning: the config field was accepted and ignored.

Caught by asking for the QAT log line and not finding it. The first QAT run
looked fine -- exit 0, ten steps, plausible loss -- and was bf16.

Fixed by extracting `KimiK3Spec.apply_build_time_features` and routing the
multimodal `build` through it. No MoonViT module can match: the LoRA target names
(`q_a_proj`, `o_proj`, `gate_proj`, ...) do not exist in the tower, which uses
`fc0/fc1/wqkv/wo`, and the QAT scope is routed experts only.

Blast radius, enumerated rather than estimated: of six multimodal flavors, the
only one that sets any of the three fields is the QAT flavor added today. **No
published result is affected.** The defect was latent and would have invalidated
exactly this matrix and nothing before it.

Before the fix, step 1 = 12.05342 (QAT requested, not applied). After, 12.06357
cold / 12.06460 seeded, and the wrapper reports `12 routed-expert modules,
MXFP8 activations on`.

## The matrix: 13/13 at 10 steps

| cell | step 1 | step 10 |
|---|---|---|
| `dp1` | 12.06460 | 9.88588 |
| `fsdp2` | 12.06460 | 9.88077 |
| `pp2` | 12.06460 | 9.87834 |
| `ep2_fsdp2` | 12.06460 | 9.87670 |
| `cp2` | 12.06626 | 9.87795 |
| `fsdp2_pp2_cp2` | 12.06626 | 9.88267 |
| `ep2_fsdp2_pp2_cp2` | 12.06626 | 9.87995 |
| `tp2` | 12.08029 | 9.88432 |
| `fsdp2_tp2_pp2` | 12.08029 | 9.88015 |
| `fsdp2_tp2_cp2` | 12.08344 | 9.88071 |
| `tp2_pp2_cp2` | 12.08344 | 9.88301 |
| `ep2_fsdp2_tp2_pp2` | 12.07571 | 9.88082 |
| `ep2_fsdp2_tp2_cp2` | 12.06143 | 9.86768 |

All thirteen decrease monotonically and land in 9.867-9.886 at step 10.

`ep2_fsdp2_tp2_pp2` failed once on `EADDRINUSE` and passed on a fresh port. That
is the only transient this project has ever actually observed, and it is the only
condition the retry path fires on.

## Defect 2: TP silently narrows the quantization scope

The step-1 values cluster, but not the way bf16 does. In bf16 the seven cells
without TP and without CP are bit-identical and the rest split cleanly into
CP / TP / both. Here the four no-TP no-CP cells are still bit-identical
(12.06460) and CP still forms its own cluster (12.06626), but the TP cells sit
0.0157 away -- an order of magnitude more than bf16's TP delta of 0.0015 -- and
the two EP x TP cells do not join either TP cluster.

That is not TP amplifying rounding. `_fake_quant_mx` bails out when the last
dimension is not a multiple of the MX block size:

    if t.shape[-1] % block_size != 0:
        return t

For `w1_EFD` / `w3_EFD` the blocked last dim is `D`, which TP does not shard. For
`w2_EDF` the last dim is the intermediate size -- exactly what TP shards. The
debug flavor's `moe_intermediate_size` is 224, which is blockable (224 = 7 x 32);
under tp2 the shard is 112, which is not. So **`w2` is not quantized at all on
those ranks**, and the run trains a narrower scope than it was asked for.

Measured directly, not inferred:

    WARNING - MXFP4 QAT: tensor of shape (8, 256, 112) left UNQUANTIZED --
    last dim 112 is not a multiple of the MX block size 32.

`(8, 256, 112)` is `w2_EDF`: 8 local experts, D=256, F sharded 224 -> 112. The
run reproduces the matrix's `tp2` leg bit-for-bit (12.08029 / 11.99565), so the
warning is diagnostic and changes no numerics.

The old comment asserted the sharded dim "is never the sharded one for w1/w3" --
true, and silent about `w2`, which is the case that breaks.

**What was fixed and what was not.** Fixed: the silence. A skipped tensor now
warns once per shape, naming the sharded-w2 case and the divisibility rule.
Not changed: the numerics. Quantizing `w2` along a non-sharded dim, or padding
the shard, would make the QAT cells depend on layout in a different way; that is
a fidelity decision, not a bug fix.

At K3's real scale the condition does not arise -- `moe_intermediate_size` 3072
is 96 blocks, and 3072/8 = 384 is 12 blocks, so it stays blockable through tp8.
The defect is that a configuration where it *does* arise gets no signal, and the
debug flavor is such a configuration.

Consequence for how the table reads: the QAT cells are **not** all running the
same quantization scope, so the step-1 spread is not a pure reduction-order
measurement the way the bf16 matrix's is. Pass/fail across thirteen layouts
stands on its own. Anyone wanting a scope-clean QAT numerics comparison needs an
intermediate size divisible by `32 * tp_degree`.

## Still not covered

* Compiled QAT. Not attempted here.
* Bit-parity against Moonshot's MX kernels. Emulated rounding targets the OCP
  spec; "MX-deployable", not "K3-QAT bit parity". Unchanged from the module
  docstring.
* Continued QAT from K3's shipped packed MXFP4 starts from an already-degraded
  master, since the bf16 master is not released.
* Max-degree QAT cells (`ep8`, `tp4`, `cp4`, `pp4`, `pp8`).
