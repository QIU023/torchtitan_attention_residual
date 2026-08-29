Fake-quant QAT for Kimi K3's routed experts, per the MXFP4/MXFP8 QAT RFC.
Before this change the grouped-expert forward consumes the bf16 parameters
directly. After it the forward consumes dequant(quant(w)) -- MXFP4 weights,
MXFP8 activations, OCP microscaling block 32 -- with a straight-through
estimator, while the bf16 masters keep training underneath. Quantization is
emulated over torchao's MX primitives, so QAT runs on any GPU; FP4 hardware
only speeds deployment.

The scope is K3's released one, expressed as a type check rather than a name
list: the shipped quantization_config targets Linear with an ignore list that
removes attention, shared experts, dense FFN, lm_head and the vision tower,
which in this module tree is exactly the GroupedExperts 3-D parameters. So
MXFP4QATConverter matches on GroupedExperts.Config, and a model with no
grouped experts raises rather than silently training unquantized. The
fake-quantized weights shadow _parameters through the instance __dict__ for
the duration of forward only, because FSDP2's reset_sharded_param does
getattr outside forward and needs the DTensor parameter back, and because
renaming the masters would break the state-dict contract and the expert
TP/EP layout, which key off these names. Under expert TP the block scales
are per-block max-abs and w2's blocked dim is the sharded one, so a shard
that stops being block-divisible falls back to bf16 with a once-per-shape
warning instead of a silent skip. Out-of-range values fall back elementwise
to the high-precision value rather than emitting non-finites. Emulated MX
targets the OCP spec but is not verified bit-identical to vendor kernels:
MX-deployable, not bit-parity. This is the complement of QLoRA's
really-packed frozen bases; the two do not compose on the same weights.
CPU tests cover the STE identity gradient, unblockable passthrough,
converter swap, forward-window restore, and gradient flow to the masters;
they pass on this branch as it sits on current main.

Training loss on kimi_k3_debugmodel_mx_qat, one seed, warmed compile cache,
measured at this branch's pre-rebase tip 6d79d0d20 (the rebase onto current main is
conflict-free, but main's KDA kernel now requires SM100-class hardware the
measuring box does not have):

| config | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 12.57942 | 8.21028 | 3.91468 |
| dp2 | 12.56931 | 8.10061 | 3.44568 |

Changed files: torchtitan/components/quantization/mx_qat.py (new),
torchtitan/models/kimi_k3/tests/test_mx_qat.py (new), the mx_qat flavor in
torchtitan/models/kimi_k3/config_registry.py.
