# verl upstream PRs from the sync work (2026-09-08)

Branch `sync_fixes_upstream` on the fork (QIU023/verl), two commits on verl-project/verl main `7cb65014`. Upstream main already gathers sharded expert stacks whole before the HF split (#7324), so the expert-stack fix from the fork is not needed there; the pipeline-stage gather and the TP padding/vocab gather have no home upstream yet (its engine raises NotImplementedError under PP and has no TP path), they wait for the engine's PP/CP/TP support to be upstreamed.

## PR 1: [debug] rollout-vs-actor log-prob difference next to the probability one

--- PASTE BEGIN ---

### Summary

`calculate_debug_metrics` reports `training/rollout_probs_diff_*` on the probability scale, `|exp(rollout_logp) - exp(actor_logp)|`. For a policy whose entropy is high that number is bounded by `exp(-entropy)` whatever the two engines compute: on a random-init 12-layer MoE with a 163,840-token vocabulary (entropy 11.5 nats) every value was around 1e-5 while the rollout and the actor were 0.78 nats apart per token, half the model's experts missing from the rollout. This adds `training/rollout_logprobs_diff_max` and `training/rollout_logprobs_diff_mean`, the same difference on the log-prob scale, next to the existing metrics; nothing else changes.

### Changed files

    verl/utils/debug/metrics.py   +6/-0   the two log-prob-scale metrics

--- PASTE END ---

## PR 2: [engine, torchtitan] ship fake-quantized expert stacks for QAT models in the weight sync

--- PASTE BEGIN ---

### Summary

A torchtitan fake-quant QAT experts module (the MXFP4-weight / MXFP8-activation converter, `torchtitan.components.quantization.mx_qat`) computes its forward with `dequant(quant(w))` while the masters stay bf16. Before this change the sync shipped the bf16 masters, so the rollout sampled a policy the actor never scores. After it the gathered expert stack is fake-quantized before the HF split when its owning module is an `MXQATExpertsBase`; every other parameter is untouched, and a torchtitan without the converter takes the guarded import's early return.

### Results

Kimi K3 debug export (12 layers, 32 routed experts, top-4), GRPO on two GPUs, `training/rollout_logprobs_diff_mean` at steps 1-3: 0.502 / 0.502 / 0.503 before, 0.260 / 0.258 / 0.255 after. Offline on the same sequences: the weights-only fake-quant forward sits 0.47 nats from the bf16 forward, the full fake-quant forward 0.24 from the weights-only one, so the remaining 0.26 is the activation fake-quant a bf16 rollout does not apply.

### Changed files

    verl/workers/engine/torchtitan/transformer_impl.py   +30/-1   `_as_the_actor_computes`, `_owning_module`, the call in the stack loop
    tests/workers/test_torchtitan_engine_qat_sync.py     +63/-0   (new)

--- PASTE END ---
