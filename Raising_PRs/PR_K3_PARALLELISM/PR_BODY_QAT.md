### Summary

Adds MXFP4-weight / MXFP8-activation fake-quant QAT for the routed experts, per the K3 MXFP4/MXFP8 QAT RFC: bf16 master parameters train underneath, the forward sees dequant(quant(w)) with a straight-through estimator, OCP microscaling block 32, emulated over torchao's MX primitives so QAT runs on any GPU -- FP4 hardware only speeds deployment.

### Design

- Scope is K3's released one, expressed as a type: the shipped `quantization_config` targets `["Linear"]` with an ignore list that removes attention, shared experts, dense FFN, lm_head and the vision tower -- i.e. the routed experts and nothing else. In this module tree those are exactly the `GroupedExperts` 3-D parameters, so `MXFP4QATConverter`'s isinstance check on `GroupedExperts.Config` IS the official scope. A model with no grouped experts raises rather than silently training unquantized. (Name-based Linear target lists quantize precisely the complement -- measured before the release clarified the scope.)
- The fake-quantized weights shadow `_parameters` through `self.__dict__` for the duration of forward only. A class property is wrong here: FSDP2's `reset_sharded_param` does `getattr(module, name)` OUTSIDE forward and needs the DTensor parameter back. Renaming the masters is also wrong: the state-dict contract and the expert TP/EP layout key off these exact names, and QAT's masters must remain ordinary trainable params.
- Under expert TP the block scales are per-block max-abs, so w2's blocked dim is the sharded one: per-shard quantization narrows the effective scope, and a shard that stops being block-divisible is left in bf16 with a once-per-shape warning instead of a silent skip.
- Emulated MX targets the OCP spec but is NOT verified bit-identical to the model vendor's kernels: "MX-deployable", not bit-parity. Out-of-range values fall back elementwise to the high-precision value rather than emitting non-finites.
- This is fake-quant QAT (quantize at import/export only), the complement of QLoRA's really-packed frozen bases; the two do not compose on the same weights.

### Evidence

CPU: STE identity gradient, unblockable passthrough, converter swap, forward-window restore (the param object is back for FSDP after forward), gradient flow to the masters.

GPU, this branch's worktree (one seed, warm cache, steps 1/3/10; the branch bases on upstream main, where the K3 EP/TP gates are still on, so its cells are data-parallel -- the EP rows below come from the integration tree, where expert parallel is wired):

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.57942 | 8.21028 | 3.91468 |
| dp2 | 2 | TBD-MAKEUP | | |

Integration tree, same recipe over its own seed -- QAT composes with expert parallel out of the box (dp2 and dp2 x ep2 print the same step-1 loss):

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.42577 | 7.39508 | 3.96145 |
| dp2 | 2 | 12.50530 | 7.34477 | 3.50247 |
| dp2 x ep2 | 2 | 12.50530 | 7.30907 | 3.48970 |
| dp4 x ep4 | 4 | 12.42457 | 6.44649 | 3.54749 |

### Changed files

    torchtitan/components/quantization/mx_qat.py  +210
    torchtitan/models/kimi_k3/config_registry.py   +16  the mx_qat flavor
    torchtitan/models/kimi_k3/tests/test_mx_qat.py +80
