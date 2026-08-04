# CLAUDE.md — Kimi K3 / AttnRes reference stack (logbook repo)

Operational context for any Claude instance working in this repo. Full narrative:
[phase13_k3like_48b_posttrain/HANDOFF_2026-07-17.md](phase13_k3like_48b_posttrain/HANDOFF_2026-07-17.md)
(authoritative strategy, from the planning session) and
[phase13_k3like_48b_posttrain/PLAN.md](phase13_k3like_48b_posttrain/PLAN.md)
(repo-side execution detail; §0 reconciles the two).

## Commit-message rule: no cross-repo reference forms (both repos, incl. the fork)

Never put `owner/repo#N` or a full `github.com/.../pull|issues/N` URL for a
THIRD-PARTY issue/PR in a commit message (title or body) — on push, GitHub
creates a permanent "referenced this PR" event in THEIR timeline; ~14 such
events already spammed pytorch/torchtitan#4025 (irrevocable; history rewrite
does not remove them and can double-fire). Write `PR-4025` / "the upstream K3
PR" instead. Bare `#N` resolves repo-locally (harmless) but avoid it too for
consistency. Deliberate cross-links belong ONLY in issue/PR comments we
intentionally post. Reference forms in FILE CONTENT (docs) are fine — files
never fire timeline events. Our own repo's issues/PRs are exempt.

## What this project is

IC (Yiqiao / QIU023) **reference implementation** of Kimi K3's training-side
infra — NOT a product. Kimi K3 (2.8T total / 104.2B activated, released 2026-07-16, weights+report
due 2026-07-27) confirmed Block Attention Residuals (AttnRes) + KDA in
production. This repo owns the earliest torchtitan AttnRes implementation +
PP cross-stage adapter (backward-correct, validated to PP8×VP4 on 8 GPUs).

Three adoption surfaces, in priority order:
1. **torchtitan upstream**: `experiments/kimi_k3/` folder to inclusion
   standard, structured to the qwen3_5 template (the hybrid linear-attention
   precedent) so core promotion is a `git mv` — but target experiments first;
   move to `models/` only if the maintainer proactively suggests it in review.
   New SHORT RFC "Kimi K3 support" before 7.27: cite the original AttnRes RFC
   (pytorch/torchtitan#3029, whose adoption gate K3 now satisfies) and offer
   issue consolidation. Maintainer history: Tianyu rejected upstreaming the
   generic cross-stage PP mechanism (~2026-04); the adapter lives as private
   impl inside the model folder's parallelize. Do not re-propose it as a
   generic mechanism.
2. **veRL recipe**: 48B+AttnRes SFT/GRPO one-command (LoRA + full-param configs).
3. **This repo**: integration hub — KD scripts, provisional 2.8T flavor,
   EP@896 scaled smoke, version pins, ≤8-GPU quickstart.

## Honesty rules (non-negotiable, from the user)

- Never claim 2.8T was personally validated. Claim: "validated on 48B real
  weights and K3-faithful topology; scale-out is config-level."
- Never present under-trained model benchmarks as competitive results.
- K3's "<2% overhead" (algorithm FLOPs) ≠ our "+2.7% step-time" (PP-adapter
  comms on PCIe). Never conflate.
- Structure details pending tech report → interfaces hold placeholders; say so.
- Inline comments: one line max; move WHY to PR body/docs.

## Repo map

- Submodule `torchtitan/` (fork QIU023/torchtitan, branch
  `attention_residual_dev`): the real implementation —
  `torchtitan/experiments/kimi_k3/` (attn_res.py, model.py,
  pipeline_adapter.py ~1143 lines, kimi_linear/ with parallelize.py ~1077
  lines: FSDP2/HSDP+TP+EP complete, CP blocked on fla-core, PP via adapter).
- Submodule `sglang/` (fork QIU023/sglang, branch
  `attention_residual_inference`): Block AttnRes two-phase inference overlay +
  VLM serving + PR branches (pr1/pr7/pr8/pr15 pushed).
- `phase2..phase12_*/`: logbook of past phases (pretrain evidence, PP pressure
  tests, VLM SFT, GRPO/OPD attempts, AD/VLA research docs).
- `phase13_k3like_48b_posttrain/`: current phase. PLAN.md + HANDOFF doc.
- `Raising_PRs/`: upstream PR filing kits (sglang/fla/pytorch/torchstore).
- `K3_RELEASE_IMPACT_2026-07-16.md`: verified K3 facts + reconciliation
  checklist for 7.27.

## Current state (2026-07-17)

- Validated: AttnRes+PP adapter numerics (|Δloss| ≤ 0.011 @ PP8×VP4, 48B-shape
  downscale, −11.4% peak mem / +2.7% tps); 447M full pipeline
  (pretrain→SFT→GRPO end-to-end, model too weak to gain — infra correct);
  PR15 loaded official Kimi-Linear-48B in SGLang.
- Upstream torchtitan merge: **DONE on the fork** (2026-07-17, `469577cdf`).
  Dev branch diff vs upstream/main is now exactly `experiments/kimi_k3/`
  (renamed from attention_residual) + 1 registry line. `experiments/rl/` =
  upstream's rebuilt version; our SGLang/Monarch RL is preserved on branch
  `experiments_rl_unmerged` (phase11 replay must check that branch out).
  All torch-2.9 compat shims dropped (fleet is torch 2.11). compileall
  clean; **pytest + debug-flavor smoke still pending on the GPU box.**
- HF↔DCP converters exist as phase11 scripts
  (`hf_to_dcp_kimi_attn_res.py`, 424/424 keys @ meta-49.12B) — the handoff's
  "state_dict_adapter ❌" is really "promote script → titan
  state_dict_adapter.py" ⚠️.
- veRL-torchtitan backend: handoff says veRL native = FSDP/Megatron only
  (titan backend = major integration work); an earlier web search suggested
  titan engine-workers exist. **Unverified conflict — check veRL source on the
  GPU box before planning around either.**

## Near-term order (pre-7.27, from HANDOFF §8)

① titan `kimi_k3/` folder to inclusion standard (debugmodel CI, parity,
smoke curves+MFU, state_dict_adapter) → ② veRL recipe one-command
(LoRA-only weight sync first) → ③ 48B POC (LoRA + full-param small GRPO;
α trainable-vs-frozen; cross-engine logprob parity) → ④ docs/extension
points → ⑤ provisional 2.8T flavor + EP@896 scaled smoke → ⑥ new RFC.
KD/downscale line is deprioritized behind ①②. Flavor configs are a
**parameterized generator** (layer ratio / experts / latent dims as variables).

On 7.27: watch vLLM PR queue (config truth may land before the report) →
artifact-discovery checklist (K3_RELEASE_IMPACT §4; incl. packed-MXFP4
quantized-weight import — never treat packed weights as plain tensors) →
regenerate flavors → official weight mapping → freeze weight-sync tensor
naming against official vLLM K3 class → rerun smoke+parity → K3 model PR.
Competitive context: Megatron-Bridge#4910 tracks KDA/AttnRes/Block-AttnRes-PP
as open/planned; our PP adapter row is already validated (PLAN §0b).

## Key technical positions (details in HANDOFF §5)

- AttnRes is likely position-wise → inference needs NO persistent cross-stage
  cache, only per-token stage payload (vLLM IntermediateTensors); keep the
  interface open until 7.27.
- LoRA does NOT exempt cross-stage backward — skip-edge gradients are real
  autograd paths; adapter must route them or gradients are silently wrong.
  Optimization hook: `first_trainable_stage` partial-backward truncation.
- CP for KDA = state-passing/scan (LASP-family), not ring attention; declared
  non-goal in the RFC (also blank in titan's qwen3_5 — a future opportunity).
- 48B graft anchor: Kimi-Linear-48B + AttnRes(α=0) [+ Gated MLA near-identity]
  is numerically identical to the original checkpoint at step 0 — the cleanest
  adapter-correctness anchor.
