**Title:** [RFC] Kimi K3 Architecture Support

## Summary

Currently, my proposed plan is to add **`torchtitan/experiments/kimi_k3/`** — the K3 model family (KDA + MLA + MoE + AttnRes) in the standard experiment layout (`model.py` / `config_registry.py` / `parallelize.py` / `state_dict_adapter.py`, following the `qwen3_5` structure as the hybrid linear-attention precedent). K3 architecture model and its pre-training/post-training implementation will be aligned with the official Kimi technical report, which is tracked to release on 07-27. I will verify through a downscaled model given GPU resource constraint, but config-only scalable to official 2.8T version from Kimi released weight checkpoints.

And I would suggest having the **RFC #3029** for Block Attention Residual support merged into this overall Kimi K3 RFC.

## Finished work as the K3-support continuation point

- AttnRes primitive + a Kimi-Linear port (KDA via `fla-core`, MLA, MoE) with FSDP2 / TP / EP / CP parallelization — [implementation](https://github.com/QIU023/torchtitan/tree/f76b3ae9a/torchtitan/experiments/kimi_k3).
- PP support via a cross-stage adapter kept **private to the model folder's `parallelize`** — no core changes, per earlier feedback on the generic-mechanism proposal. Non-Interleaved1F1B schedules fall back to the plain pipeline path (correct, without the cache saving): [adapter](https://github.com/QIU023/torchtitan/blob/f76b3ae9a/torchtitan/experiments/kimi_k3/pipeline_adapter.py), [design notes + pressure-test launchers](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase3_attnres_pp_integration).
- Numerics: the cross-stage adapter tracks the naive path within the naive-vs-naive nondeterminism band across PP×VP shapes up to **PP=8 × VP=4 (32 virtual stages)**, including a Kimi-Linear 48B-layout carrier — every adapter |Δloss| is at or below its config's own two-naive-run spread. Reproduced on the pinned current fork ([pressure-test report, 2026-07-22](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-07-22.md)); original torch-2.9-era runs at [2026-05-12](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md).
- 12.5K-step training runs on the 436M/447M Kimi-Linear shapes — [phase-4 pretrain log](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase4_kimi_attnres_lm_pretrain/README.md).
- Multimodal (vision-native) precedent: SigLIP-splice scaffold in the experiment ([model](https://github.com/QIU023/torchtitan/blob/f76b3ae9a/torchtitan/experiments/kimi_k3/multimodal_model.py), [CPU test](https://github.com/QIU023/torchtitan/blob/f76b3ae9a/torchtitan/experiments/kimi_k3/tests/test_kimi_multimodal_model.py)); LLaVA-1.5-style pretraining + SFT + GRPO exercised end-to-end on the 447M carrier — [phase-5 VLM training](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase5_vlm_multimodal_sft), [phase-11 post-training](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase11_rlhf_grpo_infra).
- CPU unit tests for the primitive, model, and pipeline adapter — [tests](https://github.com/QIU023/torchtitan/tree/f76b3ae9a/torchtitan/experiments/kimi_k3/tests).

## Plan

**Before the release (this issue is the placeholder — no PR yet):** build and smoke the post-training stack on the **open Kimi-Linear-48B-A3B weights** (the K3-family carrier available today): AttnRes graft (zero-init), SFT/GRPO with LoRA and full-param configs.

**After 2026-07-27 (weights + report + official vLLM/SGLang support):** drop the official architecture/config into the same infra — flavor configs are parametrically generated, so reconciliation (AttnRes block count, KDA:MLA ratio, gated-MLA details) is config-level — then downscale pretraining + post-training. Target: scale-up stays config-only, so **2.8T LoRA post-training runs on the same stack** given the hardware.

**Out of scope for the first landing:**

- Inference/serving — covered by Moonshot's official vLLM contribution; torchtitan scope here is training-side only.
- Vision path — the blog confirms native vision but publishes no architecture details; text-only until the tech report.

## Advance preparation (pre-7.27, exploratory — not part of the first landing)

Ahead of the release, the other K3-family deltas are prototyped as opt-in extension points (off by default), each ready to reconcile against the official config: Gated MLA (near-identity graft init), Per-Head Muon, MXFP4/MXFP8 QAT (torchao MX, emulated), Quantile Balancing routing, a provisional 2.8T-A50B flavor validated at EP@896, and LoRA / full-param post-training on the open 48B weights via the veRL torchtitan engine. Free exploration to de-risk 7.27, not proposed for landing — component-to-evidence map: [K3 reproduction stack status](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase13_k3like_48b_posttrain/K3_REPRODUCTION_STACK_STATUS.md).
