# 48B step-time investigation (8x5090, 2026-07-20)

Symptom: 48B gated+LoRA trains at ~2 tps (~4-5 min/step) regardless of
pipeline (veRL or titan-train), offload on/off, fixed or dynamic
shapes, NCCL_PROTO, or allocator mode.

## Root cause (measured)

Single-step kernel profile (torch.profiler, rank0):
`ncclDevKernel_AllGather_RING_LL` = **305.3s of 306.2s (99.7%)**,
60 calls x 5.09s. Compute (all mms) < 1s. The step IS the FSDP
per-layer weight all-gather.

Box facts (microbench, `nccl_bench.py`):
- GPU peer-to-peer: **False** (consumer PCIe rig, host-staged NCCL).
- Raw all-gather algo-bw: **3.84 GB/s**, flat from 8 MB to 5.4 GB
  payloads; unaffected by expandable_segments.

Arithmetic: 48B bf16 = 96 GB of weights; fwd+bwd re-gather moves
~213 GB/step -> **~55 s/step is the physical floor** on this fabric
even at raw speed. Historical d1280/e16-e32 carriers hit ~1000 tps
because they are ~20x smaller -- same fabric, sub-critical traffic.

## Open sub-mystery (documented, not blocking)

FSDP-effective gather bandwidth is ~0.7 GB/s -- 5x below the measured
raw ceiling. Ruled out: NCCL_PROTO=Simple (no change; OOMs at seq 512),
expandable_segments (raw bench unaffected; training without it equally
slow), shape churn, offload. Remaining suspects: per-gather output
allocation pattern inside FSDP vs the bench's reused buffer; comm/
compute stream interaction. Parked with evidence in
/workspace/smoke_runs/{prof48b,titan_48b_*}.log (box-local).

## Consequences / directions

1. This box cannot host serious 48B FSDP-8 training loops as-is; it
   remains excellent for <=1B carriers, PP-shape validation, and
   mechanism smokes (everything the checklist used it for).
2. Amortize: gather cost is per-step -- larger micro-batch tokens
   (seq 2048) buys ~4x tps at the same comm cost.
3. **QLoRA (4-bit frozen base) is a comms fix here, not just memory**:
   packed 4-bit shards cut gather traffic ~4x -> floor ~14 s/step.
   Correction of the earlier "quantization is orthogonal to step time"
   statement: on comms-bound fabrics they are the same problem.
4. H200 leg (PLAN 3c) unaffected: NVLink fabrics do not have this
   bottleneck.
