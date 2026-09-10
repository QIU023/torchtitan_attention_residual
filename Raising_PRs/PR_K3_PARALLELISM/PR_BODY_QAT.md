# PR title: [Kimi K3] MXFP4-weight / MXFP8-activation fake-quant QAT on the routed experts

Fork branch `k3_mx_qat` = `a8a850860` (two commits on main `ac10ca48f`, independent of the parallelism PRs; the first is the port, the second its scope test). Verified on this box: CPU tests below, one dp1 smoke on a single GPU.

--- PASTE BEGIN ---

## Summary

Kimi K3's quantization path: MXFP4 weights and MXFP8 activations, OCP microscaling with block 32, and bf16 master parameters training underneath. The forward sees `dequant(quant(w))` with a straight-through estimator, emulated over torchao's MX primitives, so QAT runs on any GPU; FP4 hardware only speeds deployment.

- Scope is the released one: the routed experts and nothing else. In this module tree those are the `GroupedExperts` 3-D parameters, so the converter's isinstance check on `GroupedExperts.Config` is the official scope (K3's released `quantization_config` targets `Linear` with an ignore list that removes attention, shared experts, dense FFN projections, the head and the vision tower; a name-based Linear target list would quantize precisely the set K3 keeps in high precision).
- The fake-quantized weights shadow `_parameters` through `self.__dict__` for the duration of `forward` only: a class property breaks FSDP2's `reset_sharded_param`, and renaming the masters breaks the state-dict contract and the expert TP/EP layout.
- Per-shard quantization under expert TP narrows the scope on `w2_EDF` (block scales are per-block max-abs); the non-blockable case warns once per shape instead of skipping silently.
- `kimi_k3_debugmodel_mx_qat` is the debug model under the converter.

Fidelity, stated plainly: the emulated MX rounding targets the OCP spec and is not verified bit-identical to Moonshot's kernels ("MX-deployable", not "K3-QAT-bit-parity"). This is fake-quant QAT (bf16 masters, quantize at import/export only), the complement of `components/lora.py`'s QLoRA (really-packed frozen bases, trainable adapters); the two do not compose on the same weights.

## Implementation

`torchtitan/components/quantization/mx_qat.py`: `_fake_quant_mx` (MX quantize/dequantize with the STE, passthrough when the last dim is not a multiple of the block), `_get_qat_experts_cls` (builds a subclass of the model's experts class whose `forward` swaps the fake-quantized `w1_EFD` / `w2_EDF` / `w3_EFD` in for the call and restores the masters after), `MXFP4QATConverter` (a `ModelConfigConverter` that replaces every `GroupedExperts.Config` in the model config with the QAT subclass's config and raises when it finds none). `torchtitan/models/kimi_k3/config_registry.py`: the `kimi_k3_debugmodel_mx_qat` flavor. Requires `torchao.prototype.mx_formats` (the tests skip without it).

## Limitations

- Emulated rounding, see above; no claim of bit parity with the released kernels.
- Under expert TP a `w2_EDF` shard whose last dim is not a multiple of 32 stays unquantized (warned once per shape).
- The cross-check of the converter's scope against the released checkpoint index (only the routed experts are packed there) lives with the released-format PR, which brings the `report_arch` flavor that matches the artifact; this PR checks the scope on the debug model.

## Tests

```text
pytest -q tests/unit_tests/cpu/test_kimi_k3_mx_qat.py tests/unit_tests/cpu/test_integration_test_definitions.py
```

Result: 18 passed (5 QAT: STE identity gradient, unblockable passthrough, the flavor swaps the experts while the masters stay parameters, the QAT forward differs from the parent's and the gradients flow to the masters, the converter reaches exactly the routed expert modules; 13 definitions). pre-commit passes on the touched files; `pyrefly check` reports the same error set as main `ac10ca48f`.

## Results

One dp1 smoke on a single GPU (RTX 5060 Ti, the KDA capability guard lifted locally for the run), the Kimi K3 debug model, bf16, `seed=42`, deterministic, one seed checkpoint shared by both cells (4096 tokens per step, 256 per micro-batch), 3 steps: the plain flavor against `kimi_k3_debugmodel_mx_qat` on this branch. Same init, so the step-1 gap is the fake-quant rounding of the routed experts alone.

| cell | step 1 loss / grad norm | step 2 | step 3 | peak memory |
| --- | ---: | ---: | ---: | ---: |
| `kimi_k3_debugmodel` | `12.37043` / `14.4375` | `10.23010` / `14.8125` | `7.74434` / `18.8750` | 12.64 GiB |
| `kimi_k3_debugmodel_mx_qat` | `12.37822` / `14.1250` | `10.17222` / `15.6250` | `7.36159` / `13.9375` | 12.64 GiB |

The fake-quant forward adds no resident memory (the quantized copies live only for the duration of the call). No same-configuration number exists from the earlier tree, so these are new.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
