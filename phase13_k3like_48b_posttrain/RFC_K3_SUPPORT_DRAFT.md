# RFC draft — file on pytorch/torchtitan before 2026-07-27

> Post as a new issue. Deliberately short: the long-form context lives in
> #3029 and the linked evidence. Feel free to trim further before posting.

---

**Title:** [RFC] Kimi K3 architecture support under `experiments/` (follow-up to #3029)

## Summary

#3029 proposed Block Attention Residuals (AttnRes) and was deferred pending
adoption by a production model. **Kimi K3 (released 2026-07-16) is that
model**: the [official blog](https://www.kimi.com/blog/kimi-k3) confirms
AttnRes + Kimi Delta Attention (KDA) as core architecture components (~25%
training-efficiency gain, <2% compute overhead). Open weights and the tech
report are due **2026-07-27**.

I propose adding **`torchtitan/experiments/kimi_k3/`** — the K3 model family
(KDA + MLA + MoE + AttnRes) in the standard experiment layout (`model.py` /
`config_registry.py` / `parallelize.py` / `state_dict_adapter.py`, following
the `qwen3_5` structure as the hybrid linear-attention precedent).

Maintainers: happy to have this **consolidated with #3029** into a single
tracking issue if you prefer — this supersedes it.

## Finished work as the K3-support continuation point

- AttnRes primitive + a Kimi-Linear port (KDA via `fla-core`, MLA, MoE) with
  FSDP2 / TP / EP parallelization —
  [implementation](https://github.com/QIU023/torchtitan/tree/attention_residual_dev/torchtitan/experiments/attention_residual).
- PP support via a cross-stage adapter kept **private to the model folder's
  `parallelize`** — no core changes, per earlier feedback on the
  generic-mechanism proposal:
  [adapter](https://github.com/QIU023/torchtitan/blob/attention_residual_dev/torchtitan/experiments/attention_residual/pipeline_adapter.py),
  [design notes + pressure-test launchers](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase3_attnres_pp_integration).
- Numerics: naive-vs-adapter loss within the bf16 nondeterminism band
  (|Δloss| ≤ 0.011) across PP×VP shapes up to **PP=8 × VP=4 (32 virtual
  stages)**, including a Kimi-Linear 48B-layout carrier —
  [pressure-test report](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md).
- 12.5K-step training runs on the 436M/447M Kimi-Linear shapes —
  [phase-4 pretrain log](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase4_kimi_attnres_lm_pretrain/README.md).
- CPU unit tests for the primitive, model, and pipeline adapter —
  [tests](https://github.com/QIU023/torchtitan/tree/attention_residual_dev/torchtitan/experiments/attention_residual/tests).

## Plan

**Before the release (this issue is the placeholder — no PR yet):** build and
smoke the post-training stack on the **open Kimi-Linear-48B-A3B weights** (the
K3-family carrier available today): AttnRes graft (zero-init, step-0
numerically identical to the original checkpoint), SFT/GRPO with LoRA and
full-param configs.

**After 2026-07-27 (weights + report + official vLLM/SGLang support):** drop
the official architecture/config into the same infra — flavor configs are
parametrically generated, so reconciliation (AttnRes block count, KDA:MLA
ratio, gated-MLA details) is config-level — then downscale pretraining +
post-training. Target: scale-up stays config-only, so **2.8T LoRA
post-training runs on the same stack** given the hardware.

**Out of scope:** CP for the KDA layers (`fla-core`'s `chunk_kda` has no
cross-rank state passing; same limitation exists for `qwen3_5`) and components
K3's report has not yet specified (Stable LatentMoE internals, SiTU, Per-Head
Muon) — interfaces hold placeholders until the report.

I've maintained the fork through upstream refactors since April and will
continue to own this experiment.
