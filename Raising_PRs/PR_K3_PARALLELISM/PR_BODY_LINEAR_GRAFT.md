# PR title (fork-internal until the A/B decision): [Kimi K3] The Kimi-Linear-48B graft: the released shapes, alpha-gated residual reads, the text-only checkpoint spelling

Fork branch `k3_linear_graft` = `3b6f03347` (three commits on main `ac10ca48f`, independent of the parallelism PRs). Option A of `GRAFT_48B_DESIGN_2026-09-10.md`: the old tree's Kimi-Linear generality ported into upstream's K3 folder as alternative module forms; not for upstream until the A/B decision. Verified on this box with the CPU tests below and one RTX 5060 Ti (SM120, the KDA capability guard lifted locally for the runs, never committed).

--- PASTE BEGIN ---

## Summary

Grafting the attention-residual post-train stack onto the released Kimi-Linear-48B-A3B needs three things the K3 folder did not have: the released model's shapes, a way for the grafted model to start as the backbone it loads, and the text-only checkpoint spelling.

- **The Kimi-Linear shapes.** Kimi Linear differs from K3 in every module: SiLU feed-forwards (not SiTU), an ungated MLA that projects the query straight to the heads (no `q_lora_rank`), a KDA whose output gate is factored through `head_dim` and whose decay is unbounded, and a routed MoE that reads the full-width token with one shared expert and a routed scale of 2.446 (not the latent MoE with two). The modules take these as alternative forms (`wq` or `wq_a`/`q_norm`/`wq_b`, an optional MLA `gate`; `output_gate` or `output_gate_a`/`output_gate_b`, `lower_bound=None`), the block accepts the common `FeedForward` / `MoE` configs, and K3's own configs build exactly what they built before (bitwise below). Two flavors register: `kimi_linear_48b` (the released 27-layer topology with three-layer residual blocks) and `kimi_linear_debugmodel` (the same pattern at 9 layers).
- **The graft gate.** Each residual read is mixed into the plain residual stream through a scalar alpha, `h = plain + alpha * (read - plain)`, one per read (attention and feed-forward read of every layer, and the output read), zero-initialised so step 0 is the plain backbone exactly; the plain stream is a third carrier accumulated in the backbone's op order so alpha = 0 is bit-identical. The `_gated` flavor suffix turns it on (`kimi_linear_debugmodel_gated`, `kimi_linear_48b_gated`).
- **The text-only spelling.** A config without a vision tower exports `model.*` / `lm_head.weight` instead of the multimodal `language_model.model.*` wrapper, plus the Kimi-Linear leaves (`self_attn.q_proj` on an MLA layer, `self_attn.g_a_proj` / `g_b_proj`) and the alphas; the plain routed MoE reuses the `block_sparse_moe` spelling.

## Implementation

Ported from the old tree's `model_configs.py` (`is_k3` shape switch), `model.py` (`mla_gated`, `q_lora_rank=None`, the plain MoE path built from core `TokenChoiceTopKRouter` / `GroupedExperts` / `FeedForward`), `kda.py` (`g_a_proj` / `g_b_proj`, `fused_kda_gate(..., lower_bound=None)`), `attn_res_model.py` (the alphas, `plain_stream`), `__init__.py` (the `_gated` suffix table) and `hf_key_map.py` / `state_dict_adapter.py` (the `model.` prefix, `_PASSTHROUGH_LAYER_TAGS`, `model.output_res_alpha`), onto upstream's config-tree modules: `_mla_config(q_lora_rank=None, gated=False)`, `_kda_config(full_rank_output_gate=False, gate_lower_bound=None)`, `_swiglu_feed_forward_config`, `_moe_config` (core `MoE.Config` with `route_scale`), `_kimi_linear_config`, `_apply_graft_gate`, and a prefix-aware `KimiK3StateDictAdapter`. Upstream pieces reused: `FeedForward`, `MoE`, `GroupedExperts`, `TokenChoiceTopKRouter`, `make_token_dispatcher_config`.

The unbounded decay `-exp(A_log) * softplus(raw + dt_bias)` is outside what Attention Gym's fused chunk kernels take (per-token decays in about `[-5.914, 0]`, their FP32 exponent budget over a 15-step span): at initialisation the first KDA layer already returns non-finite values on the fused path. The unbounded gate therefore runs on Attention Gym's eager reference path (`impl="reference"`, exact, no range limit); the bounded K3 gate stays fused.

## Limitations

- The graft over a LoRA base (`_gated_lora`) is not here; it stacks on the LoRA branch.
- The released 48B weights are not on this box, so nothing loads them; the spelling is checked by exporting a Kimi-Linear-shaped model and reading it back, with the key set written out from the released layout's rules.
- The unbounded decay runs on the eager reference KDA path: exact, but not a training kernel. Training the 48B shape at scale needs a chunk kernel that accepts the released decay range (the old tree ran on fla's), or the bounded gate at the cost of the released parameterisation.
- Tensor, pipeline and context parallelism are not exercised (main's K3 has none of them).
- The old tree's own Kimi-Linear flavors could not be run for a same-shape comparison: they need `fla-core`, which is not installed here.

## Tests

```text
pytest -q tests/unit_tests/cpu/test_kimi_k3_graft_gate.py tests/unit_tests/cpu/test_kimi_k3_linear_checkpoint_spelling.py tests/unit_tests/cpu/test_integration_test_definitions.py
pytest -q tests/unit_tests/gpu/test_kimi_k3.py tests/unit_tests/gpu/test_kda_attention.py
```

Result: CPU 4 + 2 + 13 passed (the gate test's model-gradient case runs on CUDA when available and did here); GPU 2 passed, 2 skipped (the same split as main on this card). pre-commit passes on every touched file; `pyrefly check` (the pinned 0.45.1) reports the same 21 errors as main `ac10ca48f`, none in the touched files.

## Results

dp1, seed 42, deterministic, 4096 tokens per step in 256-token micro-batches, 3 steps, one seed checkpoint per flavor; loss / grad norm.

| cell | step 1 | step 2 | step 3 |
| --- | --- | --- | --- |
| `kimi_k3_debugmodel`, main `ac10ca48f` | `12.53584` / `14.6250` | `9.49238` / `16.2500` | `6.60885` / `9.7500` |
| `kimi_k3_debugmodel`, this branch | bitwise | bitwise | bitwise |
| `kimi_linear_debugmodel` (ungated reads) | `12.53577` / `2.8281` | `9.96061` / `11.9375` | `8.32351` / `6.4062` |
| `kimi_linear_debugmodel_gated` (alpha = 0, own seed) | `12.54028` / `2.5312` | `10.76466` / `2.5625` | `8.50130` / `2.8750` |

The two Kimi-Linear rows are different functions by construction: the ungated flavor runs the attention-residual reads from step 0, the gated one at alpha = 0 is the plain backbone (standard residual stream). The identity that matters for a graft is checked directly: on this debug shape with its KDA layers (bf16, 256 tokens, GPU), the gated model at alpha = 0 equals the plain backbone computed through the same modules bitwise, and alpha = 0.25 on one read moves the output; the CPU test does the same on an MLA-only shape and checks every alpha receives a gradient. Loading the ungated flavor's seed checkpoint into the gated flavor is refused by DCP (`Missing key ... layers.0.attention_res_alpha`): a native checkpoint carries every parameter, and that is the strict native path, not the graft's HF path, where the reads and alphas are left out of the key space and keep their initialisation (the third adapter test).

The Kimi-Linear cells run the KDA decay on Attention Gym's eager reference path (461 tokens per second on this card at the debug shape, 458 gated, against 261 for the K3 debug model on the fused kernels at twice the width and 24 layers); the fused path returns non-finite values for the unbounded decay from the first KDA layer.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
