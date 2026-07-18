# RFC draft — file on pytorch/torchtitan before 2026-07-27

> Post as a new issue. Deliberately short: the long-form context lives in
> #3029 and the fork. Feel free to trim further before posting.

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

## What already exists (maintained fork, tracking main since April)

- AttnRes primitive + model, and a Kimi-Linear port (KDA via `fla-core`, MLA,
  MoE) with FSDP2 / TP / EP parallelization.
- PP support via a cross-stage adapter kept **private to the model folder's
  `parallelize`** — no core changes, per earlier feedback on the
  generic-mechanism proposal.
- Numerics: naive-vs-adapter loss within the bf16 nondeterminism band
  (|Δloss| ≤ 0.011) across PP×VP shapes up to **PP=8 × VP=4 (32 virtual
  stages)**, including a Kimi-Linear 48B-layout carrier; 12.5K-step training
  runs on the 436M/447M shapes.
- CPU unit tests for the primitive, model, and pipeline adapter.

Fork: https://github.com/QIU023/torchtitan/tree/attention_residual_dev

## Staged plan

1. **Now:** `experiments/kimi_k3/` PR — debugmodel CI, parallelism parity
   tests, smoke-flavor convergence curve + MFU in the README.
2. **On weights/report (7.27):** reconcile configs against the official
   release (AttnRes block count, KDA:MLA ratio, gated-MLA details) and finish
   the `state_dict_adapter` for official weights. Flavor configs are
   parametrically generated, so this is config-level.
3. **Out of scope:** CP for the KDA layers (`fla-core`'s `chunk_kda` has no
   cross-rank state passing; same limitation exists for `qwen3_5`) and
   components K3's report has not yet specified (Stable LatentMoE internals,
   SiTU, Per-Head Muon) — interfaces hold placeholders until the report.

I've maintained the fork through upstream refactors since April and will
continue to own this experiment.
