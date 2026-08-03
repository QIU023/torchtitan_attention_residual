# PR #20 — Kimi K3: load the released packed-MXFP4 expert weights

**Target**: `pytorch/torchtitan`, on top of #4025 (Kimi K3 model addition)
**Why now**: #4025's author names MXFP4 direct loading as a follow-up. This is
that follow-up, ready and verified.
**Risk**: low and contained — a decoder for a format the release already
declares, plus a loader. No change to the model math, no change to any
parallelism path.

## What the release actually ships

`moonshotai/Kimi-K3`'s `config.json` declares:

    "format": "mxfp4-pack-quantized",
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    weights: {num_bits: 4, group_size: 32, scale_dtype: torch.uint8,
              strategy: "group", symmetric: true, type: "float"}
    ignore: ["re:.*self_attn.*", "re:.*shared_experts.*",
             "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "re:.*lm_head.*",
             "re:.*vision_tower.*", "re:.*mm_projector.*"]

So the routed experts — and only the routed experts — are stored as packed
E2M1 pairs with a per-group E8M0 byte scale. A loader that treats these as plain
tensors reads garbage; there is no in-band signal that it has gone wrong.

## What this adds

`packed_mxfp4.py`, 187 lines, four functions:

- `dequantize_mxfp4` / `quantize_mxfp4` — E2M1 nibble pairs plus an E8M0 byte
  exponent, group size 32.
- `load_packed_experts` — reads the released layout into the model's expert
  parameters.
- `_e2m1_table` — the 8-value codebook.

And `quant_scope.py`'s `OFFICIAL_IGNORE_PATTERNS`, transcribed verbatim from the
config above, with `is_quantizable` as the positive predicate. Worth having as
its own thing: the scope is easy to get inverted, and an inverted scope quantizes
attention and leaves the experts in bf16 — which trains, and is wrong.

## Correctness detail that is easy to get wrong

The shared exponent is `floor(log2(amax)) - 2`, not `floor(log2(amax / 6))`.
Both look plausible against the E2M1 max of 6.0, and they differ on exact
powers of two. OCP MX specifies the former. With it, `dequantize_mxfp4` matches
torchao's reference at relative error 0.0; with the latter it is off by a factor
of two on those groups only, which is the kind of error that survives a smoke
test.

## Test plan

- `tests/test_packed_mxfp4_load.py`, `tests/test_mxfp4_qat.py` — 12 tests,
  passing, CPU-only.
- Round-trip: quantize -> dequantize agrees with torchao at rel 0.0.
- The ignore-scope predicate is asserted against the config's regex list.

## What this does NOT claim

The full 2.8T checkpoint is 1.561 TB and has not been loaded here — this is
verified on the format and on generated tensors, not on the released weights
end to end. A maintainer with the disk should run one expert tensor through
`load_packed_experts` and compare against the HF forward before merge.
