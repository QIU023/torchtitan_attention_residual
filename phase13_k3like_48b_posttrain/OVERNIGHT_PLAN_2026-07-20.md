# Overnight execution plan (2026-07-20, ~10h autonomous)

Goal: push the K3 torchtitan + veRL post-training demo (LoRA + full-param)
as far as possible. Discipline: timebox each; record data on success,
root cause on failure; incremental commit+push; no fabrication; never
delete verified work. Standard-framework demos (A/B) before more
bespoke machinery (C).

## Answered up front: MXFP4-QAT on titan (user question)

K3 ships packed MXFP4 (post-QAT-from-SFT), NOT a bf16 master. To QAT on
titan: dequant MXFP4->bf16 master (already-degraded; K3's original
master is not released), apply torchao MX fake-quant in forward. QAT is
fake-quant on bf16 compute -> runs on ANY GPU incl. H200 (FP4 hardware
only speeds deployment, not QAT). Result is MX-DEPLOYABLE but "continued
QAT from a quantized checkpoint", not bit-identical to K3's QAT (torchao
MX rounding vs Moonshot kernels unverified; full fidelity also needs
MXFP8 activation fake-quant). Matches PLAN 3b's "strict parity only with
B200-class + exact recipe".

NF4 (what our current QLoRA uses) is NEITHER NVFP4 NOR MXFP4 -- it is a
quantile-codebook software format (QLoRA/torchao), block 64, no hardware
FP4. Not bit-convertible to MXFP4 (E2M1 + E8M0 per-32). Kept as a
labelled memory/comms stopgap that titan customers may also want; the
K3-faithful MXFP4 path is a separate line (torchao mx).

## A. Post-training demos (standard composition)
- [A1] 194m full-param SFT via veRL (GSM8K).            [running]
- [A2] 194m LoRA SFT via veRL (base-vs-graft A/B material).
- [A3] NF4 QLoRA post-init trainer hook (build-time meta-quant is
       wrong; quantize after init) + small QLoRA SFT end-to-end.

## B. GRPO loop (the missing post-training piece)
- [B1] 194m GRPO via veRL main_ppo + rollout.name=hf, GSM8K
       exact-match reward. Iterate integration; capture a few real
       RL steps.
- [B2] if B1 works: GRPO base vs +AttnRes A/B.

## C. MXFP4 K3-faithful line (follow-up, clearly labelled)
- [C1] torchao mx API check + emulated MXFP4 linear prototype.
- [C2] MXFP4 fake-quant QAT wrapper, debug-scale correctness.

## D. Consolidation
- [D1] demo runbook (all working commands), RFC update, INVENTED_PARTS
       review doc (the ~30% non-standard: alpha gate / module-LoRA /
       NF4-experts subclass -- precedent + upstream risk each),
       placeholder-N note for 7.27.
- [D2] re-pin SHAs, push both repos.

Results appended to this file as each item resolves.

---
## Results log

### A1 full-param SFT 194m -- PASS
40 steps GSM8K, loss 11.49 -> 11.26, 2.85 s/step, rc=0. Random-init
fixture so descent is slow by design; demonstrates the full-param SFT
PATH end-to-end on the titan engine (dp_shard=8). 194m full-param fits
5090 comfortably; config-scalable to 48B full-param on H200 (PLAN 3c,
fp32 masters don't fit 5090).

### A2 LoRA graft SFT 194m -- PASS
Graft flavor (block_attn_res gated + LoRA r=16) loading the baseline
194m checkpoint (shared backbone, zero-init AttnRes), 40 steps, loss
10.96 -> 10.77, 3.19 s/step, rc=0. Exercises: HF->titan load into a
graft flavor, adapter-only training, the alpha-fullparam exception,
all through the veRL SFT pipeline.

### B1 GRPO 194m -- BLOCKED (structural, documented)
Progressed through: hydra config -> ray init -> worker init -> tokenizer
(custom-code, needs data.trust_remote_code=true) -> engine_workers
init_model -> rollout-class registration. WALL: veRL v1 GRPO's
`_ROLLOUT_REGISTRY` (workers/rollout/base.py) contains ONLY
(vllm|sglang|trtllm, "async") server adapters. There is NO `hf` rollout
in the RL path -- HFRollout is SFT/eval-only. So GRPO needs a real
inference server:
  - vllm: no K3 support until Moonshot's 7.27 contribution;
  - sglang: our fork's AttnRes overlay (sglang submodule, branch
    attention_residual_inference) -- the intended path, heavy setup,
    intentionally not installed in /venv/verl (protects torch 2.12);
  - trtllm: n/a.
This is exactly the "rollout-side AttnRes serving" gap in PLAN 2. Not a
one-flag fix; not fabricating a GRPO number. GRPO loop stands up to the
rollout-server boundary; closing it is the sglang-overlay leg (or
7.27 vllm). Fixes captured en route (deps: cachetools; config:
log_prob_micro_batch_size_per_gpu on rollout+ref, data.trust_remote_code)
are recorded for when a server is available.

### C MXFP4/MXFP8 QAT -- PASS (K3-faithful quant path)
torchao MX primitives confirmed (MXTensor.to_mx float4_e2m1fn_x2 /
float8_e4m3fn, block 32; dequant rel-err ~0.11 for FP4, expected).
apply_mxfp4_qat straight-through fake-quant wrapper: 2 CUDA tests pass
(wrap+forward+STE grad; quantization measurably perturbs logits).
Committed 1058f837. This is the K3-faithful line the user chose
(direction 2), distinct from the NF4 QLoRA stopgap.

### D consolidation -- DONE
- K3_REPRODUCTION_STACK_STATUS.md: full component->evidence map.
- INVENTED_PARTS_REVIEW.md: the ~30% non-standard, with risk.
- DEMO_RUNBOOK.md: every working command.
- Gated MLA (provisional near-identity graft) + Per-Head Muon (base
  algorithm faithful) added as K3 architecture/optimizer deltas, tested.
- Certification batch PASS: kimi_k3 67, dense_carrier 70, debugmodel
  train, TP=2 447m, EP@896 -- all green after ~28 overnight commits.
- Final RFC re-pin -> 7f3289a43; submodule bumped.

### Overnight scorecard
DONE: full-param SFT, LoRA SFT (194m+48B), QLoRA SFT loop + NF4 experts
+ FSDP composition, MXFP4/MXFP8 QAT (K3-faithful), Gated MLA, Per-Head
Muon, 2.8T provisional flavor, EP@896 mesh, stack-status + review +
runbook docs, full certification.
BLOCKED (documented, not faked): GRPO (needs inference server), 48B
step-time (all-gather bound, H200), QLoRA-in-trainer (meta-first
ordering), full-5D + 48B full-param (H200).
