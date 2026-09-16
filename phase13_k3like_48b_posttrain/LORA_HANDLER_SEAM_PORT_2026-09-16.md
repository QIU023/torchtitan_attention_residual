# LoRA / QLoRA onto main's `LoRATransform` handler seam: survey and port plan (2026-09-16)

RFC 3029 deliverable: "#4576 on the core LoRA transform". Survey done read-only on `/tmp/wt_int0916` (`45ed0e3c1` = main `810e62786` + 72), by a read-only subagent, checked against the files named. Line numbers are the worktree's; main's `lora.py:25-207` is byte-identical to the worktree's `lora.py:692-874`, so the fork's `LoRAConverter` and everything above line 692 is purely additive.

## 1. Main's seam (`torchtitan/config/transform/lora.py`, `torchtitan/models/common/lora.py`)

- Handler protocol `_LoRAHandler` (`lora.py:718-730`): `config_type: type[Module.Config]` and `make_config(cfg, *, rank, alpha) -> Module.Config`. That is the whole extension surface: no quantization argument, no FQN, no parent config.
- `LinearLoRAHandler` (`lora.py:733-752`): `config_type = Linear.Config`; `make_config` calls `specialize_lora_linear(cfg._owner)` and rebuilds the config from `fields(cfg)` plus rank and alpha.
- `LoRATransform` (`lora.py:755-871`): `run_after = (ContextParallelTransform,)`; fields `handlers: tuple[_LoRAHandler, ...]` (instances), `rank`, `alpha`, `target_modules`; `__post_init__` rejects a handler whose `config_type` subclasses an earlier handler's (`796-802`); `transform` walks `model.traverse(Module.Config, recurse=True)` leaves to root, picks the first `isinstance` match in handler order (`833-840`), gates on `target_modules` by the FQN's last segment (`841-843`), wraps every other config in a frozen subclass (`854`, `_make_frozen_config` `692-715`). `conflicts_with = (LoRATransform,)` (`874`).
- Parameters come from `models/common/lora.py`: `_LoRALinearMixin` (`33-61`, `lora_a` kaiming, `lora_b` zeros, `scaling = alpha / rank`), `_adapter_sharding` (`63-87`, handles `S(0)` and `S(1)` only; line 83 asserts `S(1)`), `specialize_lora_linear` (`90-102`, cached, the nested `Config` adds exactly `rank` and `alpha`). `__all__ = ["specialize_lora_linear"]`, so no public marker class.
- `validate_converter_compatibility` (`config/transform/converter.py:37-54`) knows only `QuantizationConverter` and `LMHeadCastConverter`; transform conflicts are `_reject_conflicts` (`apply.py:49-58`), which sees transforms only. Converters run inside `model_registry` before any transform (`README.md:33-37`).
- Export, merge, adapter-only state dict: not present on main (`git grep` on `810e62786` for `merge_lora_state_dict`, `trainable_state_dict`, `quantize_lora_bases`, `LoRAConverter`, `MXFP4Experts` is empty). Main's only LoRA wiring is `llama3_debugmodel_float8_emulate_lora` (`llama3/config_registry.py:225-250`): quantization converter inside `model_registry`, then `apply_transforms(config, [LoRATransform(handlers=(LinearLoRAHandler(),), rank=8, alpha=16.0, target_modules=[...])])`.

## 2. The fork's pieces (commits `5e905d7b8` and nine follow-ups; flavors `87dbb698b`, `a591f288b`; `to_hf` skip `7d56552fe`)

- `LoRAConverter` (`lora.py:877-1013`): the legacy-converter twin of the transform, with `quantize_experts` and `quantize_base` in its `Config` (`902-914`), subclass hooks `_supports_lora` / `_make_lora_config` (`941-958`), and a third branch in `convert` (`981-994`) that turns every `GroupedExperts.Config` with `dim % 32 == 0 and hidden_dim % 32 == 0` into the packed MXFP4 experts class, not gated on `target_modules`.
- `_get_lora_cls` (`lora.py:108-495`): a parallel factory to `specialize_lora_linear` (different MRO, its own dict cache) whose `Config` adds `quantize_base` and overrides `build()` to promote `_packed_sharding` into `_sharding_config` (`127-132`), because `Module.Config.build` installs `_param_init` and `_sharding_config` after `__init__` (`protocols/module.py:90-99`).
- `LoRALinearBase` marker (`75-102`), `_lora_adapter_sharding` (`32-72`) with the extra branch (`53-61`, `5f0c5af94`) that gives adapters of an `R` or `I` declared base the base's own placement (main's line 83 asserts `S(1)` and fails on Kimi's declarations).
- NF4 frozen base (`347-375`, `1183-1196`, `quantize_lora_bases` post-load hook; `init_states` refuses a pack on a DTensor base, `329-344`).
- Packed MXFP4 frozen base, pack then shard (`164-172`, `176-246`: `weight` replaced by `base_qdata` uint8 `[out, in/2]` and `base_scale` e8m0 `[out, in/32]`, works on meta, rewrites a private copy of the sharding config, `2b19b0880`); from-scratch init on this rank's rows (`248-277`); local-shard dequant with the MX-block alignment error (`279-306`).
- Packed-base TP forward `_forward_packed_tp` (`377-452`): local dequant, `Partial` grad placements, colwise `Shard(-1)`, rowwise `Partial()` with bias divided by the TP size. No test covers it.
- Packed MXFP4 grouped experts (`498-689`): `MXFP4ExpertsBase` marker, `w1_EFD` / `w2_EDF` / `w3_EFD` packed to `(E*A, B/2)` qdata + `(E*A, B/32)` scale, class properties that dequantize on read, `to_local()` under EP, the declared 3-D entry carried onto the packed pair (`3f1c2ffe9`).
- `trainable_state_dict` (`1016-1023`); `merge_lora_state_dict` (`1059-1180`): folds `W + (alpha/rank) B A` in the modules so serialization hooks fire (fused `wqkv` to split keys), NF4 and packed bases dequantized, experts gathered with `full_tensor()`, everything restored in a `finally`, adapter and packed keys stripped; `_state_dict_prefix` raises on an unknown wrapper.
- Export key spelling: `kimi_k3/state_dict_adapter.py:28-30` (`_LORA_ADAPTER_SUFFIXES`) and the `to_hf` skip at `171-176`; the released packed spelling exists on the read side only (`QuantizedHuggingFaceStorageReader`, `311-323`). No writer emits the released packed spelling.
- `scripts/quantize_lora_dcp.py`: builds the packed flavor on meta to derive the key map, one-pass DCP repack. No test.
- Flavors: `kimi_k3/config_registry.py:116-172` (`kimi_k3_debugmodel_lora`, `_kimi_k3_lora_converter`, `kimi_k3_debugmodel_qlora_mxfp4`), converter-only; `kimi_k3/__init__.py` has no LoRA wiring and `model_registry` has no `transforms=` parameter.
- `quantization/mx_qat.py` (`MXFP4QATConverter`) rewrites every `GroupedExperts.Config` too and states it does not compose with the QLoRA path; `_EXPERT_WEIGHT_NAMES` is duplicated there.

## 3. Mapping and what main lacks

| fork capability | main seam | verdict |
| --- | --- | --- |
| rank / alpha / target_modules | `LoRATransform` fields | 1:1 |
| `quantize_base` on a linear | a handler subclass holding the option as instance state (`QLoRALinearHandler(quantize_base=...)`, `config_type = Linear.Config`); handlers are instances, so no protocol change | fits |
| the QLoRA linear class (packed storage, NF4 forward, packed TP forward) | a second mixin in `models/common/lora.py` next to `_LoRALinearMixin`, a `specialize_qlora_linear`, the handler picks the factory | fits, needs 3(a) |
| adapters of an R / I base | the branch of `5f0c5af94` into `_LoRALinearMixin._adapter_sharding` (`models/common/lora.py:77-87`) | upstream fix |
| packed MXFP4 experts | a quantization transform composed before LoRA (the `MXFP4QATConverter` shape), not a LoRA handler: `target_modules` gates every handler (`841-843`) and the experts' FQN segment is not in Kimi's target list | separate transform |
| `trainable_state_dict`, `merge_lora_state_dict` | none | new upstream API, needs a public marker |
| adapter keys out of `to_hf` | `StateDictAdapter.to_hf` override, already the seam Kimi uses | fits |
| released packed spelling on export | absent on both sides | out of scope |

No seam on main, to name upstream: (a) `Module.Config.build` installs `_param_init` and `_sharding_config` after `__init__` (`protocols/module.py:90-99`), so a module that changes its own parameter set at `__init__` cannot see the declarations it must rewrite; the fork works around it in three places (`lora.py:127-132`, `571-576`, `308-323`, `639-651`); the ask is a post-build hook or a way to declare replacement parameter names and placements from `__init__`. (b) `target_modules` gates every handler uniformly; a per-handler `matches(fqn, cfg)` or the separate-transform route above. (c) No merge or export API and no exported marker (`__all__` at `models/common/lora.py:27`). (d) Neither `validate_converter_compatibility` nor `_reject_conflicts` sees the other kind, so `LoRAConverter` next to `LoRATransform`, or `MXFP4QATConverter` next to `LoRAConverter(quantize_experts="mxfp4")`, is accepted silently.

## 4. Conflicts while both implementations stay

- Two dynamically built `LoRA{parent}` classes from separate caches, unrelated types.
- `merge_lora_state_dict` checks `LoRALinearBase` only (`1081-1085`): a model adapted by `LinearLoRAHandler` is skipped, the adapters are never folded, and `to_hf` then drops `.lora_a.weight` / `.lora_b.weight` (`state_dict_adapter.py:171-176`), so the export is a base-only checkpoint that looks successful. `quantize_lora_bases` is a no-op on main-built modules the same way. Today no flavor mixes the two, so nothing is wrong on the tree; it is the trap of a half port.
- A flavor using both would double-adapt: `LinearLoRAHandler` matches the fork's `LoRALinear.Config` (a `Linear.Config` subclass), and neither validator sees the other kind; both also wrap non-targets in frozen subclasses, so `Frozen{Frozen{Config}}` and order dependence.

## 5. Order of work

1. Upstream, smallest: the R / I branch into `_LoRALinearMixin._adapter_sharding` (without it main's LoRA cannot build Kimi K3 under its TP declarations).
2. Upstream: export a marker from `models/common/lora.py`, then `trainable_state_dict` and `merge_lora_state_dict` (minus the quantized branches, keeping `_state_dict_prefix`), and the Kimi `to_hf` skip as it is.
3. Upstream: the `Module.Config.build` post-hook (3a); everything packed depends on it and it removes the three workarounds.
4. NF4 base: `specialize_qlora_linear`, `NF4LoRALinearHandler(quantize_base="nf4")`, the post-load hook, the DTensor refusal kept verbatim.
5. Packed MXFP4 base and its from-scratch init.
6. Packed-base TP forward, with the distributed test it never had.
7. Packed grouped experts as a standalone quantization transform composed before `LoRATransform`, with the merge's expert branch.
8. Rewire the Kimi flavors from `model_registry(converters=[LoRAConverter.Config(...)])` to `apply_transforms(config, [ExpertsMXFP4Transform(...), LoRATransform(handlers=(...))])` as llama3 does; update `scripts/quantize_lora_dcp.py`'s marker names.
9. Delete `LoRAConverter` and `_get_lora_cls` in the same change as 8, not before (section 4).

## 6. Tests

Fork coverage today: `tests/unit_tests/cpu/test_lora.py:533-757` (adapter-only dict, merge keys and zero-init identity, delta fold, serialization hooks, wrappers, NF4 pack / forward / merge, packing at init, MXFP4 pack at build and merge), `test_kimi_k3_qlora_experts.py` (pack at build, dequant property, merge restores keys, pack under the declared sharding), `test_state_dict_adapter_stages.py:48-61`. Gaps: `_forward_packed_tp` untested; `kimi_k3_debugmodel_lora()` built by no test and absent from `test_debug_config_defaults.py` and `tests/integration_tests/b200.py`; `quantize_lora_dcp.py` untested.

Main's tests to keep green after the port: `test_lora.py:30-532` (handler selection, the shadowing rule at `lora.py:796-802`, so a `Linear.Config`-subclass handler is declared before `LinearLoRAHandler`; freezing on composite and root modules), `test_transforms.py:221, 279` (`test_lora_runs_after_context_parallelism`, every trainable parameter is `lora_a` / `lora_b`), `test_torchft_trainer.py:23-36`, `test_quantization.py:94-118` (quantization before LoRA, the exact adapted FQN set), `test_debug_config_defaults.py` (add the Kimi flavors), `tests/integration_tests/features.py:313-318` (`float8_emulate_lora_tp2_pp2`, the only distributed LoRA job; a Kimi packed-TP analogue belongs in `b200.py`, with `test_integration_test_definitions.py:129-140` updated in step), `test_mx_qat.py:35, 58`.
