# QAT upstream audit -- 2026-09-02

Why this exists: `k3_qat` merges clean onto current main and its four CPU
tests pass, yet it should not be filed in its present form. MXFP4/MXFP8 is
not a K3 property, three groups are moving the same ground upstream, and the
maintainers have not adopted any fake-quant design. This records what was
found so the branch is re-based on a decision rather than on the code
compiling.

Decision: **QAT is deferred** behind QB and LoRA. It is not abandoned.

## Already in main (the baseline the branch must match)

| component | content |
|---|---|
| `quantization/mx.py` | `MXFP8LinearConverter`; `MXFP8GroupedExpertsConverter` overrides the `_grouped_mm` seam with torchao's `_quantize_then_scaled_grouped_mm` -- real MXFP8 compute, SM100 only |
| `quantization/nvfp4.py` | `NVFP4LinearConverter` (PR-3914, merged 2026-08-05) |
| `quantization/__init__.py` | `QuantizationConverter` base (carries `model_compile_enabled`); every quantization converter is re-exported from here |

## In flight, real low-precision compute (the expert-weight lifecycle is being redefined)

| PR | state | relation to `k3_qat` |
|---|---|---|
| PR-4368 grouped-mm seam takes `weight_ENK` in stored orientation instead of `B_t` | approved by tianyu-l 2026-09-02, about to merge | we do not override the seam; unaffected, but the neighbouring code moves |
| PR-4344 FSDP-managed MXFP8 grouped-expert training | draft "DO NOT REVIEW", stacked on PR-4368 | touches `kimi_k3` experts directly; FSDP quantizes in its post-all-gather hook and owns qdata/scale; titan owns the autograd function and the "model policy", torchao is kernel-only |
| PR-4203 MxFP8 overhaul (dense linear, same lifecycle) | "Do not review", waits on PR-4325 | explicitly leaves the grouped path alone |
| PR-4257 MXFP8 fused-MLP overrides (NVIDIA) | open, 8 CPU tests | `@override` registry entries for `FeedForward` and `GroupedExperts`; needs a torchao fused kernel |

## In flight, fake-quant (the same kind as `k3_qat`)

| PR | state | relation to `k3_qat` |
|---|---|---|
| PR-3265 QAT converter (mori360) | open, no activity since 2026-05-08 | torchao `QATConfig`, post-build `nn.Module` converter, seven schemes including `mx` and `nvfp4`, `finalize()` swaps in real quantized modules |
| PR-3889 NVFP4 quantization-aware distillation (andrewor14, torchao) | draft, no activity since 2026-07-08 | self-contained fake-quant primitives, STE, MoE experts and dense linears, weights and activations, a MoE fake-quant hook in `moe.py` |

`k3_qat` is the third fake-quant design: STE, bf16 masters, MXFP4 weights
and MXFP8 activations, routed experts only, a config-level converter keyed
on `isinstance(cfg, GroupedExperts.Config)`. It has the narrowest scope of
the three.

## Structural mismatches against main

- `MXFP4QATConverter` subclasses `ModelConfigConverter`, not the
  `QuantizationConverter` base main already has, and is not exported from
  `quantization/__init__.py`.
- It sits next to `mx.py` under the name `mx_qat.py` but uses a different
  mechanism: `mx.py` overrides the `_grouped_mm` seam; `mx_qat.py` installs
  fake-quantized weights into the instance `__dict__` for the duration of
  forward, shadowing `_parameters`.
- PR-4344 makes FSDP the owner of expert qdata/scale after all-gather. The
  forward-window shadowing then becomes a second mechanism on the same
  tensors. The maintainers' own phrasing is that titan owns the "model
  policy"; a reviewer will ask why MXFP4 fake-quant is not expressed as a
  policy in that framework.

What is actually K3-specific: the scope (the released `quantization_config`'s
ignore list reduces to "routed experts only", expressed as the isinstance
check) and the flavor. The mechanism belongs to the quantization component,
so the `[kimi_k3]` prefix on the draft title misplaces it.

## Options, in preference order

1. Fold into PR-3889. It already has the experts-and-dense fake-quant hook
   and the STE; what it lacks is the MXFP4 element dtype and an
   experts-only scope switch. `_fake_quant_mx` (block-32 non-divisible
   fallback, non-finite fallback) is the contributable piece. The author is
   in torchao and better placed to set the shape of fake-quant.
2. File as a draft only with a rewritten body: `QuantizationConverter`
   subclass, exported, no `[kimi_k3]` framing, and a section relating it to
   PR-3265 / PR-3889 / PR-4344. Without that section it sits like PR-3265.
3. Wait for PR-4344 and re-express MXFP4 fake-quant as a policy in its
   lifecycle.

Either way, re-run the CPU tests once PR-4368 merges: it does not touch the
branch, but the adjacent code in `moe.py` moves.

## Collateral found on the way

The `kimi_k3_debugmodel_mx_qat` flavor had leaked into `k3_lora_extras`
(`config_registry.py`, importing `mx_qat.py`, which does not exist on that
branch -- an ImportError for anyone selecting it). Removed from the LoRA
branch on 2026-09-02; the flavor belongs to the QAT branch.
