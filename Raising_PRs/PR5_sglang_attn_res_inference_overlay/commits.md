# Backing commits — PR #5 Block AttnRes inference overlay

## Discovered in

**Algorithm root**: Phases **2-4** (`experiments/attn_res/` + `experiments/kimi_linear/` —
torchtitan training-side AttnRes, paper-Table-1 reproduction).

**Inference path**: Phase **11** — SGLang carrier work for the AttnRes
ckpt (`phase11_rlhf_grpo_infra/eval_*.py` + the full inference benchmark sweep). The
overlay's modular split into algorithm-core / per-arch carriers
emerged during Phase 11 as we added Qwen3 after Kimi-Linear to validate
the algorithm-core's model-agnosticity.

## Fork source

| Commit | Title | Files |
|---|---|---|
| `4a27b32e1` | `phase 10 Stage C: KimiBlockAttnResForCausalLM model class` | new model class — `attn_res_overlay.py` seed |
| `2f2e917d8` | `phase 11: Block AttnRes generic inference overlay + two-phase + seq-shard` | core algorithm `layers/attn_res.py` + overlay |
| `d89f5be75` | `phase 11: static cleanup for upstream-PR readiness` | cleanup pass on overlay surface |
| `61c83cb30` | `phase 11: bench-only env toggles for AttnRes overlay` | bench toggles (likely strip for upstream) |
| `b8bd81a19` | `phase 11: flatten leading dims for sgl_kernel rmsnorm (cuda-graph compat)` | RMSNorm shim |
| `0ddd84617` | `phase 11: fix shard-mode AR in fallback path (correctness regression)` | correctness fix |
| `63325b2b4` | `phase 11: fused Triton Phase-2 merge + RMSNorm + logit kernel` | two-phase fusion |
| `a61c5c79f` | `[VLM] AttnRes carrier: SigLIP + projector + KimiBlockAttnResForCausalLM` | VL carrier seed |
| `63ea2ab75` | `[VLM] processor + overlay refinements (host-side image-token splice)` | VL processor |
| `ac56bcbc0` | `[VLM][fix] HybridLinearAttnBackend dispatch for VLM-wrapped Kimi Linear` | model_runner hook |
| `d6fb3bbd7` | `[VLM][fix] AttnRes VLM inference: set_forward_context + MLA layer wiring + eps` | VL fix-ups |

All commits on `QIU023/sglang@attention_residual_inference` (and
`main` after merge to `dc154e785`).

## Status

**Code complete, runs end-to-end on hf_step3100 ckpt at 44.6 tok/s
bf16 baseline + verified Qwen3 carrier independent of Kimi-Linear.**
NOT cherry-pick-ready: the commits accumulated organically over Phase
11 and contain bench-only env toggles + AttnRes-specific shapes that
need a refactor pass before upstreaming (see PR.md "Suggested staging"
for the 3-PR breakdown).

## Filing recipe

```bash
# This is NOT a single cherry-pick. The full path:

# 1. File the RFC issue first using PR.md as the body.
# 2. Wait for upstream RFC discussion to conclude + Kimi K-series
#    public release for algorithm legitimacy.
# 3. Open PR #5a (algorithm-only):
#    - Refactor `layers/attn_res.py` to be model-agnostic.
#    - Strip bench-only env toggles (commit 61c83cb30).
#    - Add standalone unit tests.
#    - Cherry-pick distilled change from 2f2e917d8 + 63325b2b4 +
#      b8bd81a19 + 0ddd84617.
# 4. After 5a lands, open 5b (Kimi-Linear carrier):
#    - Cherry-pick attn_res_overlay.py from 2f2e917d8 + 4a27b32e1.
#    - Cherry-pick model_runner hook from ac56bcbc0.
# 5. After 5b lands, open 5c (VL carrier):
#    - Cherry-pick a61c5c79f + 63ea2ab75 + d6fb3bbd7.
#    - Fold PR #2 (base64 data-URL) as a day-1 processor feature.
```

## Conflict surface

High. The overlay touches `model_runner.py` and the model registry —
both are hot paths on upstream main. Expect rebase work at each PR
stage.

## Notes for the PR opener

- **Do not file before Kimi K-series public release.** The RFC will
  bounce as "one fork's experiment" without the production anchor.
- The algorithm-only PR (#5a) is the cleanest first contribution if
  the RFC closes positively. ~250 LOC, has its own tests, no model
  wiring.
- Plan for at least 3 rounds of review per PR. Total timeline:
  multi-month.
