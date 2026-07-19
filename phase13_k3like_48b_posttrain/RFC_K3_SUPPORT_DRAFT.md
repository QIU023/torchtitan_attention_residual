**Title:** [RFC] Kimi K3 Architecture Support

## Summary

Hi TorchTitan maintainers! I have previously raised [RFC #3029](https://github.com/pytorch/torchtitan/issues/3029) proposing Block Attention Residuals (AttnRes) with a personal torchtitan fork containing its distributed parallelism infra implementations. Per maintainer feedback, the PR was gated on a production model adopting Attention Residuals, which has now arrived: **Kimi K3** — the [official blog](https://www.kimi.com/blog/kimi-k3) confirms AttnRes + Kimi Delta Attention (KDA) as core architecture components (~25% training-efficiency gain, <2% compute overhead). Open weights and the tech report are due to be released by the Kimi team by **2026-07-27**.

Therefore, I would sincerely request to own and contribute the Kimi K3 full support in torchtitan, scoped mainly to Kimi K3 architecture model (text-only and vision-native) pre-training and (full-param & LoRA) post-training (using torchtitan as the distributed framework backend of veRL).

Currently, my proposed plan is to add **`torchtitan/experiments/kimi_k3/`** — the K3 model family (KDA + MLA + MoE + AttnRes) in the standard experiment layout (`model.py` / `config_registry.py` / `parallelize.py` / `state_dict_adapter.py`, following the `qwen3_5` structure as the hybrid linear-attention precedent). K3 architecture model and its pre-training/post-training implementation will be aligned with the official Kimi technical report, which is tracked to release on 07-27. I will verify through a downscaled model given GPU resource constraint, but config-only scalable to official 2.8T version from Kimi released weight checkpoints.

And I would suggest having the **RFC #3029** for Block Attention Residual support merged into this overall Kimi K3 RFC.

## Finished work as the K3-support continuation point

- AttnRes primitive + a Kimi-Linear port (KDA via `fla-core`, MLA, MoE) with FSDP2 / TP / EP parallelization — [implementation](https://github.com/QIU023/torchtitan/tree/a3b3c74b3/torchtitan/experiments/kimi_k3).
- PP support via a cross-stage adapter kept **private to the model folder's `parallelize`** — no core changes, per earlier feedback on the generic-mechanism proposal: [adapter](https://github.com/QIU023/torchtitan/blob/a3b3c74b3/torchtitan/experiments/kimi_k3/pipeline_adapter.py), [design notes + pressure-test launchers](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase3_attnres_pp_integration).
- Numerics: naive-vs-adapter loss within the bf16 nondeterminism band (|Δloss| ≤ 0.011) across PP×VP shapes up to **PP=8 × VP=4 (32 virtual stages)**, including a Kimi-Linear 48B-layout carrier — [pressure-test report](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-05-12.md).
- 12.5K-step training runs on the 436M/447M Kimi-Linear shapes — [phase-4 pretrain log](https://github.com/QIU023/torchtitan_attention_residual/blob/main/phase4_kimi_attnres_lm_pretrain/README.md).
- Multimodal (vision-native) precedent: SigLIP-splice scaffold in the experiment ([model](https://github.com/QIU023/torchtitan/blob/a3b3c74b3/torchtitan/experiments/kimi_k3/multimodal_model.py), [CPU test](https://github.com/QIU023/torchtitan/blob/a3b3c74b3/torchtitan/experiments/kimi_k3/tests/test_kimi_multimodal_model.py)); LLaVA-1.5-style pretraining + SFT + GRPO exercised end-to-end on the 447M carrier — [phase-5 VLM training](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase5_vlm_multimodal_sft), [phase-11 post-training](https://github.com/QIU023/torchtitan_attention_residual/tree/main/phase11_rlhf_grpo_infra).
- CPU unit tests for the primitive, model, and pipeline adapter — [tests](https://github.com/QIU023/torchtitan/tree/a3b3c74b3/torchtitan/experiments/kimi_k3/tests).

## Plan

**Before the release (this issue is the placeholder — no PR yet):** build and smoke the post-training stack on the **open Kimi-Linear-48B-A3B weights** (the K3-family carrier available today): AttnRes graft (zero-init, step-0 numerically identical to the original checkpoint), SFT/GRPO with LoRA and full-param configs.

**After 2026-07-27 (weights + report + official vLLM/SGLang support):** drop the official architecture/config into the same infra — flavor configs are parametrically generated, so reconciliation (AttnRes block count, KDA:MLA ratio, gated-MLA details) is config-level — then downscale pretraining + post-training. Target: scale-up stays config-only, so **2.8T LoRA post-training runs on the same stack** given the hardware.

**Out of scope for the first landing:**

- CP / 1M-context training — in the hybrid stack, ring/zigzag applies only to the full-attention (MLA) layers; KDA layers need Ulysses-style head sharding or LASP-style cross-rank state passing, neither of which `fla-core`'s `chunk_kda` supports today (the same blank exists for `qwen3_5`). Future work.
- Inference/serving — covered by Moonshot's official vLLM contribution; torchtitan scope here is training-side only.
- Vision path — the blog confirms native vision but publishes no architecture details; text-only until the tech report.
