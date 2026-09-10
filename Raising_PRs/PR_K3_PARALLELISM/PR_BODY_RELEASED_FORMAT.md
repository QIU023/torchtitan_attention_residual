# PR title: [Kimi K3] The released checkpoint format: two released-topology downscales, the released log-decay layout, packed weights refused, the quantized reader

Branch `k3_released_format` = `65badf428` (one commit on main `ac10ca48f`, independent of the parallelism PRs; nothing in the stack depends on it and it depends on nothing in the stack). Verified on 8 x RTX 5060 Ti (SM120) with the KDA capability guard lifted locally for the smoke (main refuses SM120); the released-layout debug artifact is `KIMI_K3_RELEASED_DEBUG_DIR`, the released index `KIMI_K3_RELEASED_INDEX`.

--- PASTE BEGIN ---

### Summary

Kimi K3 loads from and exports to the layout Moonshot released, checked against a released-layout artifact rather than against the adapter's own output.

- Two faithful downscales join the registry: `report_arch`, a 13-layer twin of the released topology at width 256 (full attention at 1-based 4, 8, 12, 13, a dense first layer, 8 latent-MoE experts with two shared, attention residual blocks of 7 so the depth ends in a partial block the way 93 = 7 x 12 + 9 does, a 4-layer tower at width 256 with every structural feature of the released one), which is the topology of the debug-size checkpoint shipped in the released layout; and `k3mini`, a 21-layer downscale at width 512 that keeps the 12-layer blocks and the released head dimensions so the KDA kernel runs at its real width.
- The state dict adapter carries the per-head log-decay `A_log` in the released `[1, 1, heads, 1]` layout on export and flattens it on import (`dt_bias` already did the equivalent).
- `from_hf` refuses packed or scaled tensors (`.weight_packed`, `.weight_scale`, fp8 / uint8 values) instead of mapping their bytes as weights, and `get_hf_storage_reader(path, from_quantized=True)` hands a quantized checkpoint to torch's `QuantizedHuggingFaceStorageReader`, so the released MXFP4 experts arrive dequantized through core's `--checkpoint.initial_load_in_hf_quantized`.

### Implementation

`__init__.py` adds `_small_vision_encoder_config`, `_report_arch` and `_k3mini` next to `_debugmodel` and `_kimi_k3`, on the same `_kimi_k3_config` assembly; the full-attention sets are spelled out like the existing flavors'. `state_dict_adapter.py` adds the two reshapes on the KDA `A_log` key, `_check_not_packed` (called at the top of `from_hf`) and the reader override; the existing key map is unchanged.

### Limitations

- The three new tests need artifacts that are not in the repository: the released-layout debug checkpoint (`KIMI_K3_RELEASED_DEBUG_DIR`, default `/workspace/k3qat_mm_hf`) and the released index (`KIMI_K3_RELEASED_INDEX`, no default). They skip without them.
- The debug artifact is unpacked (658 plain keys), so the tests exercise the packed-key refusal and the reader's plumbing, not a dequantizing load; that path is core's reader.
- Config export to the released `config.json` and video preprocessing are not part of this change.

### Tests

```text
KIMI_K3_RELEASED_INDEX=<released model.safetensors.index.json> pytest -q tests/unit_tests/cpu/test_kimi_k3_released_checkpoint.py tests/unit_tests/cpu/test_kimi_k3_quant_scope.py tests/unit_tests/cpu/test_kimi_k3_vision_preprocess_parity.py tests/unit_tests/cpu/test_integration_test_definitions.py
```

27 passed, 0 skipped: the released-layout checkpoint loads into `report_arch` with every key on a parameter of the right shape (412 parameters, 658 released keys, 0 missing / 0 extra / 0 shape mismatches) and exports back bitwise, the twin's dimensions match the artifact's `config.json`, packed keys are refused (3); the released index packs exactly the routed experts (1); the shared NaViT resize matches the released `media_utils.navit_resize_image` on ten image sizes at the released budgets, exact on resized size, padding and token count (10); the integration-test definitions (13). pre-commit passes on the touched files; `pyrefly check` reports the same error set as main `ac10ca48f` (64 pre-existing environment errors, none in the touched files).

### Results

`report_arch`, dp1, bf16, seed 42, deterministic, 256 tokens per step, the multimodal debug data, 2 steps; the loaded cell reads the released-layout artifact through `--checkpoint.initial_load_in_hf` (658 keys, 0.31 s):

| cell | step 1 loss | step 1 grad norm | step 2 loss | step 2 grad norm |
| --- | ---: | ---: | ---: | ---: |
| loaded from the released-layout artifact | `12.04915` | `22.5000` | `12.00766` | `44.7500` |
| seed-42 init, no load | `12.44157` | `6.9688` | `11.76225` | `10.6875` |

The artifact is a debug export at an untrained loss (ln 163840 = 12.0), so the row shows the load path end to end, not a trained model.

### Changed files

    tests/unit_tests/cpu/test_kimi_k3_quant_scope.py               +41
    tests/unit_tests/cpu/test_kimi_k3_released_checkpoint.py       +122
    tests/unit_tests/cpu/test_kimi_k3_vision_preprocess_parity.py  +96
    torchtitan/models/kimi_k3/__init__.py                          +85
    torchtitan/models/kimi_k3/state_dict_adapter.py                +55

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
