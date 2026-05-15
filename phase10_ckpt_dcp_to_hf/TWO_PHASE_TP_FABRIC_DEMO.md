# Phase 10 Stage I — Two-Phase TP Fabric Pattern Demo

This is a **standalone fabric-pattern smoke** demonstrating the unique
collective signature that arises when Block AttnRes inference fuses its
Phase 2 (online softmax merge) into the post-attention TP collective
path, replacing a single `AllReduce` with `ReduceScatter -> local merge ->
AllGather`.

The smoke is intentionally minimal: it does not implement two-phase
inside the actual `KimiAttnResDecoderLayer`, only fires the
**collective patterns** that two-phase would generate. The intent is
to make IXIA-level pattern recognition possible without committing
to the full integration (which is ~1 day of code, beyond Phase 10's
21h budget).

## Setup
* World: 8 ranks, all in single TP group
* Per-step work: 16 collective events (matching v11's 16 attention layers)
* Message size: 12 MB (matching v11's per-attention TP AllReduce)
* Steps: 50

## Captured signatures

Both runs ran at TP=8 with a `bf16` 12 MB tensor per attention layer,
50 steps × 16 layers each.

### AllReduce baseline (`phase5_vlm_multimodal_sft/runs/two_phase_tp_allreduce/tier_b_trace/`)

```
AllReduce         12 MB  nranks=8  count=6400  (per-rank histogram)
AllReduce        <1 KB   nranks=8  count=8     (final barrier)
```

Per-step pattern: 16 × `AllReduce(12 MB, nranks=8)` events. Standard
ring algorithm fires `2*(world-1) = 14` Send/Recv hops per logical
AllReduce, producing the ~6400 sub-events in the trace.

### RS+AG two-phase (`phase5_vlm_multimodal_sft/runs/two_phase_tp_rs_ag/tier_b_trace/`)

```
ReduceScatter    12 MB  nranks=8  count=6400
AllGather        12 MB  nranks=8  count=6400
AllReduce        <1 KB  nranks=8  count=8    (final barrier)
```

Per-step pattern: 16 × (`ReduceScatter` + `AllGather`) pairs = **32
collective events vs 16** in the AllReduce baseline.

## IXIA differentiator

The two patterns have **identical total fabric bytes** (RS reduces
into per-rank shards; AG re-assembles; net bytes equivalent to one
AllReduce per call). What differs is the **fabric pattern shape**:

* **AllReduce mode**: single contiguous burst per attention layer.
  Time axis: `[AR][AR][AR]...` 16 times.
* **RS+AG mode**: paired bursts per attention layer with a "merge gap"
  between them where local compute happens (online softmax merge).
  Time axis: `[RS][gap][AG][RS][gap][AG]...` 32 events with
  inter-event gaps.

For IXIA modeling this means:
1. **Pattern detection**: RS-followed-by-AG (with non-zero inter-event
   spacing) is the unique signature. AllReduce-followed-by-AllReduce
   is the baseline.
2. **Buffering implications**: AR can pipeline naturally; RS+AG
   requires the merge step to complete before AG fires, creating a
   serialization point that affects link utilization patterns.
3. **Per-link load**: AR uses ring algorithm (each link sees
   ~2*(N-1)/N of the message size in either direction); RS+AG uses
   directed reduce-scatter then all-gather, with each link seeing
   roughly the message size in each direction sequentially. Net
   bytes equivalent, instantaneous link utilization different.

## Production-grade integration (future work)

To get the *real* RS+AG pattern in actual inference, modify
`KimiAttnResDecoderLayer.forward` to:

1. Hold off the post-attention TP collective until after the
   AttnRes pre-FFN aggregation has been computed.
2. Replace `tensor_model_parallel_all_reduce(attn_out)` with a
   manual `reduce_scatter -> online_softmax_merge_with_partial ->
   all_gather` sequence.

Estimated integration effort: 1 day (model.py + parallelize.py +
numerical equivalence test). Captured in
`phase7_nccl_traffic_catalog/FINAL_CATALOG.md` future-work item 5.

## Files

* `phase10_ckpt_dcp_to_hf/two_phase_tp_smoke.py` — both modes in one file (mode
  controlled by `--mode {allreduce,rs_ag}`)
* `phase10_ckpt_dcp_to_hf/run_two_phase_smoke.sh` — runs both back-to-back, captures
  separate tier_b traces, runs the standard pipeline → 2 ixia_config
  files for side-by-side IXIA loading.
* `phase5_vlm_multimodal_sft/runs/two_phase_tp_{allreduce,rs_ag}/tier_b_trace/` —
  trace artifacts (ixia_config.json + collective_summary.csv.gz +
  flows.csv.gz).
