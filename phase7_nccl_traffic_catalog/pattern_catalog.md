# Phase 7 NCCL Collective Pattern Catalog

Auto-generated from `phase7_nccl_traffic_catalog/extract_collectives.py` outputs across all `phase5_vlm_multimodal_sft/runs/8gpu_*/tier_{a,b,c}_trace/collective_summary.csv` files. **PCIe wallclock is uninterpretable on this hardware; pattern data (op, size, participants, count) is independent of physical interconnect and is the deliverable.**

## Replay priority (most realistic first)

| Priority | Config | Tier | GBS | Steps | Trace dir | Total collectives |
|---|---|---|---|---|---|---|
| 1 | `a2_fsdp2_pp4` | tier_a | 384 | 100 | `phase5_vlm_multimodal_sft/runs/8gpu_a2_fsdp2_pp4_tier_a/tier_a_trace` | 1202 |
| 2 | `b0_fsdp8` | tier_a | 384 | 100 | `phase5_vlm_multimodal_sft/runs/8gpu_b0_fsdp8_tier_a/tier_a_trace` | 263378 |
| 3 | `a2_fsdp2_pp4` | tier_b | 120 | 50 | `phase5_vlm_multimodal_sft/runs/8gpu_a2_fsdp2_pp4_tier_b/tier_b_trace` | 1070 |
| 4 | `b0_fsdp8` | tier_b | 120 | 50 | `phase5_vlm_multimodal_sft/runs/8gpu_b0_fsdp8_tier_b/tier_b_trace` | 23778 |
| 5 | `a2_fsdp2_pp4` | tier_c | 16 | 500 | `phase5_vlm_multimodal_sft/runs/8gpu_a2_fsdp2_pp4_seed42/tier_c_trace` | 792306 |
| 6 | `a3_fsdp2_pp2_tp2` | tier_c | 16 | 500 | `phase5_vlm_multimodal_sft/runs/8gpu_a3_fsdp2_pp2_tp2_seed42/tier_c_trace` | 2140462 |
| 7 | `b0_fsdp8` | tier_c | 16 | 500 | `phase5_vlm_multimodal_sft/runs/8gpu_b0_fsdp8_seed42/tier_c_trace` | 236178 |

## Collective histograms per (config, tier)

### `a2_fsdp2_pp4`

| Collective | Size bucket | nranks | Tier A count | Tier B count | Tier C count |
|---|---|---|---:|---:|---:|
| AllGather | <1KB | 8 | 24 | 24 | 40 |
| AllGather | 1-64KB | 2 | 12 | 10 | 8000 |
| AllGather | 1-16MB | 2 | 8 | 8 | 16000 |
| AllGather | 16-256MB | 2 | 168 | 166 | 136016 |
| AllReduce | <1KB | 2 | 8 | 8 | 20000 |
| AllReduce | <1KB | 4 | 0 | 0 | 4000 |
| AllReduce | <1KB | 8 | 0 | 0 | 8 |
| Broadcast | <1KB | 8 | 8 | 8 | 24 |
| Broadcast | 1-16MB | 8 | 0 | 0 | 8 |
| Recv | <1KB | 4 | 56 | 56 | 56 |
| Recv | <1KB | 8 | 7 | 7 | 14 |
| Recv | 1-64KB | 8 | 14 | 14 | 14 |
| Recv | 64KB-1MB | 4 | 258 | 204 | 176000 |
| Recv | 64KB-1MB | 8 | 0 | 0 | 21 |
| Recv | 1-16MB | 4 | 74 | 60 | 48000 |
| ReduceScatter | 1-64KB | 2 | 12 | 10 | 8000 |
| ReduceScatter | 1-16MB | 2 | 6 | 6 | 16000 |
| ReduceScatter | 16-256MB | 2 | 120 | 132 | 120000 |
| ReduceScatter | 256MB+ | 2 | 14 | 12 | 16000 |
| Send | <1KB | 4 | 56 | 56 | 56 |
| Send | <1KB | 8 | 7 | 7 | 14 |
| Send | 1-64KB | 8 | 14 | 14 | 14 |
| Send | 64KB-1MB | 4 | 262 | 208 | 176000 |
| Send | 64KB-1MB | 8 | 0 | 0 | 21 |
| Send | 1-16MB | 4 | 74 | 60 | 48000 |

### `a3_fsdp2_pp2_tp2`

| Collective | Size bucket | nranks | Tier A count | Tier B count | Tier C count |
|---|---|---|---:|---:|---:|
| AllGather | <1KB | 8 | 0 | 0 | 40 |
| AllGather | 1-64KB | 2 | 0 | 0 | 16000 |
| AllGather | 1-16MB | 2 | 0 | 0 | 32000 |
| AllGather | 16-256MB | 2 | 0 | 0 | 288036 |
| AllReduce | <1KB | 2 | 0 | 0 | 28000 |
| AllReduce | <1KB | 8 | 0 | 0 | 8 |
| AllReduce | 64KB-1MB | 2 | 0 | 0 | 496024 |
| AllReduce | 1-16MB | 2 | 0 | 0 | 64000 |
| Broadcast | <1KB | 8 | 0 | 0 | 24 |
| Broadcast | 1-16MB | 8 | 0 | 0 | 8 |
| Recv | 0 | 2 | 0 | 0 | 96000 |
| Recv | <1KB | 2 | 0 | 0 | 112 |
| Recv | <1KB | 8 | 0 | 0 | 14 |
| Recv | 1-64KB | 8 | 0 | 0 | 14 |
| Recv | 64KB-1MB | 2 | 0 | 0 | 352000 |
| Recv | 64KB-1MB | 8 | 0 | 0 | 21 |
| ReduceScatter | 1-64KB | 2 | 0 | 0 | 16000 |
| ReduceScatter | 1-16MB | 2 | 0 | 0 | 32000 |
| ReduceScatter | 16-256MB | 2 | 0 | 0 | 272000 |
| Send | 0 | 2 | 0 | 0 | 96000 |
| Send | <1KB | 2 | 0 | 0 | 112 |
| Send | <1KB | 8 | 0 | 0 | 14 |
| Send | 1-64KB | 8 | 0 | 0 | 14 |
| Send | 64KB-1MB | 2 | 0 | 0 | 352000 |
| Send | 64KB-1MB | 8 | 0 | 0 | 21 |

### `b0_fsdp8`

| Collective | Size bucket | nranks | Tier A count | Tier B count | Tier C count |
|---|---|---|---:|---:|---:|
| AllGather | <1KB | 8 | 40 | 40 | 40 |
| AllGather | 1-64KB | 8 | 4800 | 400 | 4000 |
| AllGather | 64KB-1MB | 8 | 4800 | 400 | 4000 |
| AllGather | 1-16MB | 8 | 153600 | 12800 | 128000 |
| AllGather | 16-256MB | 8 | 4800 | 400 | 4000 |
| AllReduce | <1KB | 8 | 3208 | 1608 | 16008 |
| AllReduce | 1-64KB | 8 | 800 | 400 | 4000 |
| Broadcast | <1KB | 8 | 24 | 24 | 24 |
| Broadcast | 1-16MB | 8 | 8 | 8 | 8 |
| Recv | <1KB | 8 | 14 | 14 | 14 |
| Recv | 64KB-1MB | 8 | 28 | 28 | 28 |
| Recv | 1-16MB | 8 | 7 | 7 | 7 |
| ReduceScatter | 1-64KB | 8 | 4800 | 400 | 4000 |
| ReduceScatter | 64KB-1MB | 8 | 4800 | 400 | 4000 |
| ReduceScatter | 1-16MB | 8 | 4800 | 400 | 4000 |
| ReduceScatter | 16-256MB | 8 | 76800 | 6400 | 64000 |
| Send | <1KB | 8 | 14 | 14 | 14 |
| Send | 64KB-1MB | 8 | 28 | 28 | 28 |
| Send | 1-16MB | 8 | 7 | 7 | 7 |

## Cross-config comparison at Tier A (production-standardized)

Which collectives fire under each 3D config, at production tensor sizes. A blank cell means that collective never appeared in that config's Tier A trace.

| op | size | nranks | `a2_fsdp2_pp4` | `b0_fsdp8` |
|---|---|---|---|---|
| AllGather | <1KB | 8 | 24 | 40 |
| AllGather | 1-64KB | 2 | 12 |  |
| AllGather | 1-64KB | 8 |  | 4800 |
| AllGather | 64KB-1MB | 8 |  | 4800 |
| AllGather | 1-16MB | 2 | 8 |  |
| AllGather | 1-16MB | 8 |  | 153600 |
| AllGather | 16-256MB | 2 | 168 |  |
| AllGather | 16-256MB | 8 |  | 4800 |
| AllReduce | <1KB | 2 | 8 |  |
| AllReduce | <1KB | 8 |  | 3208 |
| AllReduce | 1-64KB | 8 |  | 800 |
| Broadcast | <1KB | 8 | 8 | 24 |
| Broadcast | 1-16MB | 8 |  | 8 |
| Recv | <1KB | 4 | 56 |  |
| Recv | <1KB | 8 | 7 | 14 |
| Recv | 1-64KB | 8 | 14 |  |
| Recv | 64KB-1MB | 4 | 258 |  |
| Recv | 64KB-1MB | 8 |  | 28 |
| Recv | 1-16MB | 4 | 74 |  |
| Recv | 1-16MB | 8 |  | 7 |
| ReduceScatter | 1-64KB | 2 | 12 |  |
| ReduceScatter | 1-64KB | 8 |  | 4800 |
| ReduceScatter | 64KB-1MB | 8 |  | 4800 |
| ReduceScatter | 1-16MB | 2 | 6 |  |
| ReduceScatter | 1-16MB | 8 |  | 4800 |
| ReduceScatter | 16-256MB | 2 | 120 |  |
| ReduceScatter | 16-256MB | 8 |  | 76800 |
| ReduceScatter | 256MB+ | 2 | 14 |  |
| Send | <1KB | 4 | 56 |  |
| Send | <1KB | 8 | 7 | 14 |
| Send | 1-64KB | 8 | 14 |  |
| Send | 64KB-1MB | 4 | 262 |  |
| Send | 64KB-1MB | 8 |  | 28 |
| Send | 1-16MB | 4 | 74 |  |
| Send | 1-16MB | 8 |  | 7 |

## Caveats

* **PCIe wallclock is unrepresentative.** Counts and tensor sizes are framework-determined and portable; latency / throughput would shift ~10× on NVLink-class interconnects.
* **Tier C is alignment-load (GBS=12).** Counts here are low-batch and don't reflect production overlap behavior. Use Tier A for replay decisions.
* **CP=2 traces missing.** kimi_linear's KDA layers' fla-core kernel doesn't support ring-recurrence over seq-sharded inputs. See `parallelize.py` CP branch.
* **TP plan is conservative.** Dense MLP only; KDA/MLA/AttnRes stay replicated. See `apply_tp_kimi_linear` docstring.
