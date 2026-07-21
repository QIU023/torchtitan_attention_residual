# MXFP4-base LoRA + parallelism-matrix verification (2026-07-21)

Option A: LoRA on a frozen MXFP4 base (K3's native FP4 weight format), and
the adapter-LoRA x parallelism matrix that prior SFT never covered (it ran
FSDP-only). All on the 8x5090 box, debug graft flavor
(`kimi_linear_debugmodel_gated_lora`: AttnRes alpha-graft + LoRA rank-8,
4 layers, 8 experts). torchtitan fork @1ca1bd39.

## Phase 0 -- MXTensor under FSDP2 (decision gate): PASS via split-storage

FSDP2 rejects an MXTensor param directly ("non-contiguous parameters not
supported" -- the packed qdata is half-width so the logical view has a
mismatched stride; NF4Tensor shards directly but MXTensor does not).
Fix (`lora.py quantize_base_mxfp4`): store `qdata` (uint8) + `scale` (E8M0
bytes viewed as uint8, since FSDP2 all-gather has no e8m0 copy kernel) as
plain contiguous frozen params + the flatten ctx, reconstruct the MXTensor
via `__tensor_unflatten__` after all-gather. Shards + trains on 2 GPU.

## Phase 1 -- lora.py MXFP4 branch: DONE, suite 79 green

`quantize_base_mxfp4` / forward dequant-matmul (no weight-only MX linear in
torchao 0.17 yet) / merge dequant / `quantize_lora_bases(mode='mxfp4')`
post-load hook / adapter-dtype. block-32 needs in%32==0 (all K3 dims).
Unit tests: mxfp4 merge+export, post-load hook. GroupedExperts MXFP4 = TODO.

## Phase 2 -- adapter-LoRA x parallelism matrix (8 cards)

Closes the flagged gap: prior LoRA SFT was pure FSDP; PP/EP were only ever
run full-param (the "q_lora_rank" in those logs is MLA latent, not LoRA).

**bf16-LoRA, real `torchtitan.train` path** (`run_lora_parallelism_matrix.sh`):

| cell | loss@3 | grad_norm |
|---|---|---|
| FSDP (dp8) | 7.617 | 0.018 |
| FSDP+EP (dp8,ep2) | 7.585 | 0.024 |
| FSDP+PP (dp4,pp2) | 7.676 | 0.047 |
| FSDP+PP+EP (dp4,pp2,ep2) | 7.646 | 0.046 |
| wide-EP (dp8,ep8) | 7.612 | 0.018 |

All losses in the random-init CE band (~ln(2016)=7.6), grads finite/nonzero.
**AttnRes-LoRA x PP** (the correctness-critical cell): last-stage loss 7.68
matches FSDP 7.62 and grad_norm>0 -> the cross-stage adapter routes the
LoRA/graft skip-edge gradients (no silent-wrong-gradient). FSDP+PP+EP is the
3-axis simultaneous composition, never run before. (PP logs a negative
placeholder loss on the first stage; the real loss is on the last stage.)

**MXFP4-LoRA, real KimiLinear model under FSDP2, 8 cards**
(`mxfp4_lora_fsdp_real.py`): 15 bases packed to MXFP4, loss finite, gnorm
0.013, 57/57 trainable (LoRA+graft) get grads, base frozen. The real lora.py
MXFP4 path shards + all-gathers + dequant-matmuls + trains.

MXFP4 x {EP, PP} directly: NOT run (torchtitan.train has no post-load
quantize hook). Inferred from the composition -- bf16-LoRA composes with
EP/PP (matrix above) and the MXFP4 base shards under FSDP2 (Phase 0 + real
harness); MXFP4 is a base-storage swap on the same wrappers. Direct
verification is a follow-up (wire quantize_lora_bases into the train/
parallelize path).

## Phase 3 -- MXFP4 caveats (measured, not hidden)

- **Per-weight rel-err 11.4%** (FP4 E2M1 coarseness).
- **step-0 logit rel-err 36%** on the debug model -- per-layer errors
  compound. A frozen POST-HOC MXFP4 base (no QAT) deviates substantially;
  the LoRA adapters must compensate. NF4 is finer for post-hoc quant.
- **Faithful path**: K3's released weights are MXFP4-QAT (trained robust to
  MXFP4). Post-hoc quantizing bf16 weights is lossy by nature -> the real
  "customer takes K3 MXFP4 weights + LoRA" test wants those QAT weights
  (7.27), plus optional MXFP4-QAT continued training (mxfp4_qat.py).
- **Memory win**: base packs 3.76x; not visible at debug scale (base tiny),
  manifests at 48B where the base dominates -- same as NF4.
- **48B sharded-load**: MXFP4 base through titan shard-then-load DCP is
  unverified on the 5090 (same open question as NF4) -> H200 item.
