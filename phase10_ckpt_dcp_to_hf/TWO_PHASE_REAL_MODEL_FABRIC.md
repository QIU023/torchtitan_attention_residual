# Phase 10 Stage K — Two-Phase TP Fabric in Real-Model Context

Stage I demonstrated the RS+AG vs AllReduce pattern with a synthetic
12 MB tensor. **Stage K does the same in actual kimi_linear AttnRes
inference** — runs the real model forward (firing real FSDP/EP/TP
fabric) AND injects the RS+AG ops at real-model attention output
shape (4 × 512 × 1168 ≈ 4.6 MB) per layer per step.

The injected ops match what production-grade Block AttnRes inference
would generate when its post-attention TP path uses `ReduceScatter →
local merge → AllGather` instead of standard `AllReduce` (per the
project blog).

## Configuration
* Model: kimi_linear_436m_block_attn_res_n4 (16 layers, hidden 1168)
* Ckpt: phase4 step-8000 (1.39B params)
* Mesh: FSDP=4 × TP=2 × EP=2 (PP=1) on 8 GPUs
* Steps: 50, seq=512, micro_bs=4
* Wall: 12.9 s

## Captured fabric (Stage K trace)

```
ReduceScatter   1-16 MB  nranks=2  count=6400   <- INJECTED two-phase
AllGather       1-16 MB  nranks=2  count=6400   <- INJECTED two-phase
AllGather       1-16 MB  nranks=4  count=6400   (regular FSDP)
AllGather      16-256 MB nranks=2  count=6000   (regular FSDP)
AllGather       256 MB+  nranks=2  count=400    (init unshard)
AllGather       1-16 MB  nranks=2  count=6000   (regular FSDP)
AllGather       <1 KB    nranks=8  count=24     (barrier)
AllGather      16-256 MB nranks=4  count=8      (FSDP layer-wise)
AllGather       1-64 KB  nranks=4  count=8      (small param FSDP)
```

## Pattern recognition for IXIA

Direct comparison with **Stage D inference baseline** (same model,
same mesh, no injection):

| Pattern | Stage D | Stage K | Delta |
|---|---|---|---|
| `ReduceScatter 1-16 MB nranks=2` | 0 | **6400** | +6400 (INJECTED) |
| `AllGather 1-16 MB nranks=2` (extra) | 0 | **6400** | +6400 (INJECTED) |
| `AllGather 1-16 MB nranks=4` (FSDP) | 6400 | 6400 | unchanged |
| `AllGather 16-256 MB nranks=2` (FSDP) | 6000 | 6000 | unchanged |
| `AllGather 256 MB+ nranks=2` (init) | 400 | 400 | unchanged |

50 steps × 16 layers = 800 logical RS+AG events × ~8 NCCL ring
sub-events per call = 6400 traced rows. Math checks out.

The pair signature **`ReduceScatter @ 1-16 MB nranks=2` immediately
followed by `AllGather @ 1-16 MB nranks=2`** is the diagnostic
that distinguishes two-phase Block AttnRes inference from standard
AllReduce inference. IXIA pattern detection: `RS → AG sequential at
nranks=2 with matching message size`.

## Implications

1. **Two-phase fabric is detectable**: the RS+AG injection produces a
   distinct collective sequence that does not appear in any other
   regime (training, inference baseline, naive autoregressive, real
   PPO). It's a unique Block AttnRes-inference fingerprint.
2. **Volume-equivalent**: total bytes sent through fabric for the
   RS+AG pair == bytes sent through the equivalent AllReduce. The
   distinction is purely in **collective shape and sequencing**.
3. **Production deployment can now be modeled**: an IXIA test that
   alternates RS / AG bursts at 1-16 MB nranks=2 reproduces the
   two-phase Block AttnRes fabric pattern at the per-attention-layer
   granularity. With our captured trace this can be tuned to v11
   model dimensions exactly.

## Production integration path

To replace the **injected** RS+AG with **actual** post-attention
TP-collective replacement (true two-phase, no extra fabric on top of
the standard collective):

1. Modify `apply_tp_kimi_linear` in
   `torchtitan/torchtitan/experiments/kimi_linear/parallelize.py` to
   replace `o_proj`'s `RowwiseParallel` with a custom plan that
   produces `Shard(seq_dim)` output (i.e., effective ReduceScatter
   instead of AllReduce).
2. Modify `KimiAttnResDecoderLayer.forward` to accept the
   sharded-output, apply Phase 2's online softmax merge locally on
   the shard, then explicitly AllGather before the next layer.
3. Numerical validation: shard-wise RS-then-merge-then-AG must match
   AR-then-merge bitwise (or bf16-tolerant) under the same input.

Estimated effort: 1 day. Stage K's fabric injection is the validator
— once integration is done, the trace should show only the injected-
style 6400 RS + 6400 AG without the standard AllReduce that
currently fires inside o_proj.

## Files

* `phase10_ckpt_dcp_to_hf/inference_two_phase_real.py` — Stage D inference + per-step
  injected RS + AG ops at real-model attention shape.
* `phase10_ckpt_dcp_to_hf/run_two_phase_real.sh` — runner.
* `phase5_vlm_multimodal_sft/runs/inference_two_phase_real/tier_b_trace/` — trace
  artifacts (ixia_config.json + summary CSVs).
