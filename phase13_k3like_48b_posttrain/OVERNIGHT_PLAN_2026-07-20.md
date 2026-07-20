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
