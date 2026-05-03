# Phase 7 NCCL Collective Pattern Catalog

Auto-generated from `phase7/extract_collectives.py` outputs across all `phase5/runs/8gpu_*/tier_{a,b,c}_trace/collective_summary.csv` files. **PCIe wallclock is uninterpretable on this hardware; pattern data (op, size, participants, count) is independent of physical interconnect and is the deliverable.**

## Replay priority (most realistic first)

| Priority | Config | Tier | GBS | Steps | Trace dir | Total collectives |
|---|---|---|---|---|---|---|
| 1 | `a2_fsdp2_pp4` | tier_a | 384 | 100 | `phase5/runs/8gpu_a2_fsdp2_pp4_tier_a/tier_a_trace` | 1202 |
| 2 | `a3_fsdp2_pp2_tp2` | tier_a | 384 | 100 | `phase5/runs/8gpu_a3_fsdp2_pp2_tp2_tier_a/tier_a_trace` | 1490 |
| 3 | `all4d_fsdp2_pp2_tp2_ep2` | tier_a | 384 | 100 | `phase5/runs/8gpu_all4d_fsdp2_pp2_tp2_ep2_tier_a/tier_a_trace` | 1338 |
| 4 | `b0_fsdp8` | tier_a | 384 | 100 | `phase5/runs/8gpu_b0_fsdp8_tier_a/tier_a_trace` | 263378 |
| 5 | `noppc_fsdp4_tp2_ep2` | tier_a | 384 | 100 | `phase5/runs/8gpu_noppc_fsdp4_tp2_ep2_tier_a/tier_a_trace` | 226 |
| 6 | `a2_fsdp2_pp4` | tier_b | 120 | 50 | `phase5/runs/8gpu_a2_fsdp2_pp4_tier_b/tier_b_trace` | 1070 |
| 7 | `b0_fsdp8` | tier_b | 120 | 50 | `phase5/runs/8gpu_b0_fsdp8_tier_b/tier_b_trace` | 23778 |
| 8 | `noppc_fsdp4_tp2_ep2` | tier_b | 120 | 50 | `phase5/runs/8gpu_noppc_fsdp4_tp2_ep2_tier_b/tier_b_trace` | 226 |
| 9 | `a2_fsdp2_pp4` | tier_c | 16 | 500 | `phase5/runs/8gpu_a2_fsdp2_pp4_seed42/tier_c_trace` | 792306 |
| 10 | `a3_fsdp2_pp2_tp2` | tier_c | 16 | 500 | `phase5/runs/8gpu_a3_fsdp2_pp2_tp2_seed42/tier_c_trace` | 3518 |
| 11 | `all4d_fsdp2_pp2_tp2_ep2` | tier_c | 16 | 500 | `phase5/runs/8gpu_all4d_fsdp2_pp2_tp2_ep2_seed42/tier_c_trace` | 2694 |
| 12 | `b0_fsdp8` | tier_c | 16 | 500 | `phase5/runs/8gpu_b0_fsdp8_seed42/tier_c_trace` | 236178 |
| 13 | `noppc_fsdp4_tp2_ep2` | tier_c | 16 | 500 | `phase5/runs/8gpu_noppc_fsdp4_tp2_ep2_seed42/tier_c_trace` | 226 |

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
| AllGather | <1KB | 8 | 24 | 0 | 24 |
| AllGather | 1-64KB | 2 | 16 | 0 | 32 |
| AllGather | 1-16MB | 2 | 16 | 0 | 64 |
| AllGather | 16-256MB | 2 | 220 | 0 | 576 |
| AllReduce | <1KB | 2 | 8 | 0 | 16 |
| AllReduce | 64KB-1MB | 2 | 36 | 0 | 100 |
| Broadcast | <1KB | 8 | 8 | 0 | 8 |
| Recv | 0 | 2 | 76 | 0 | 192 |
| Recv | <1KB | 2 | 112 | 0 | 112 |
| Recv | <1KB | 8 | 7 | 0 | 7 |
| Recv | 1-64KB | 8 | 14 | 0 | 14 |
| Recv | 64KB-1MB | 2 | 284 | 0 | 704 |
| ReduceScatter | 1-64KB | 2 | 16 | 0 | 32 |
| ReduceScatter | 1-16MB | 2 | 12 | 0 | 64 |
| ReduceScatter | 16-256MB | 2 | 136 | 0 | 480 |
| ReduceScatter | 256MB+ | 2 | 20 | 0 | 64 |
| Send | 0 | 2 | 72 | 0 | 192 |
| Send | <1KB | 2 | 112 | 0 | 112 |
| Send | <1KB | 8 | 7 | 0 | 7 |
| Send | 1-64KB | 8 | 14 | 0 | 14 |
| Send | 64KB-1MB | 2 | 280 | 0 | 704 |

### `all4d_fsdp2_pp2_tp2_ep2`

| Collective | Size bucket | nranks | Tier A count | Tier B count | Tier C count |
|---|---|---|---:|---:|---:|
| AllGather | <1KB | 8 | 24 | 0 | 24 |
| AllGather | 1-64KB | 2 | 4 | 0 | 4 |
| AllGather | 1-16MB | 2 | 8 | 0 | 16 |
| AllGather | 16-256MB | 2 | 68 | 0 | 112 |
| AllReduce | <1KB | 2 | 8 | 0 | 8 |
| AllReduce | 64KB-1MB | 2 | 12 | 0 | 20 |
| Broadcast | <1KB | 8 | 8 | 0 | 8 |
| Recv | 0 | 2 | 118 | 0 | 136 |
| Recv | <1KB | 2 | 262 | 0 | 354 |
| Recv | <1KB | 8 | 7 | 0 | 7 |
| Recv | 1-64KB | 8 | 14 | 0 | 14 |
| Recv | 64KB-1MB | 2 | 22 | 0 | 128 |
| Recv | 1-16MB | 2 | 184 | 0 | 576 |
| ReduceScatter | 1-64KB | 2 | 0 | 0 | 4 |
| ReduceScatter | 1-16MB | 2 | 0 | 0 | 8 |
| ReduceScatter | 16-256MB | 2 | 0 | 0 | 60 |
| ReduceScatter | 256MB+ | 2 | 0 | 0 | 8 |
| Send | 0 | 2 | 116 | 0 | 136 |
| Send | <1KB | 2 | 262 | 0 | 354 |
| Send | <1KB | 8 | 7 | 0 | 7 |
| Send | 1-64KB | 8 | 14 | 0 | 14 |
| Send | 64KB-1MB | 2 | 16 | 0 | 120 |
| Send | 1-16MB | 2 | 184 | 0 | 576 |

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

### `noppc_fsdp4_tp2_ep2`

| Collective | Size bucket | nranks | Tier A count | Tier B count | Tier C count |
|---|---|---|---:|---:|---:|
| AllGather | <1KB | 8 | 24 | 24 | 24 |
| AllGather | 1-16MB | 4 | 16 | 16 | 16 |
| AllGather | 16-256MB | 4 | 24 | 24 | 24 |
| AllReduce | <1KB | 4 | 8 | 8 | 8 |
| AllReduce | 64KB-1MB | 2 | 0 | 0 | 8 |
| AllReduce | 1-16MB | 2 | 8 | 8 | 0 |
| Broadcast | <1KB | 8 | 8 | 8 | 8 |
| Recv | <1KB | 2 | 16 | 16 | 16 |
| Recv | <1KB | 8 | 7 | 7 | 7 |
| Recv | 64KB-1MB | 8 | 14 | 14 | 14 |
| Recv | 1-16MB | 2 | 16 | 0 | 32 |
| Recv | 16-256MB | 2 | 16 | 32 | 0 |
| Send | <1KB | 2 | 16 | 16 | 16 |
| Send | <1KB | 8 | 7 | 7 | 7 |
| Send | 64KB-1MB | 8 | 14 | 14 | 14 |
| Send | 1-16MB | 2 | 16 | 0 | 32 |
| Send | 16-256MB | 2 | 16 | 32 | 0 |

## Cross-config comparison at Tier A (production-standardized)

Which collectives fire under each 3D config, at production tensor sizes. A blank cell means that collective never appeared in that config's Tier A trace.

| op | size | nranks | `a2_fsdp2_pp4` | `a3_fsdp2_pp2_tp2` | `all4d_fsdp2_pp2_tp2_ep2` | `b0_fsdp8` | `noppc_fsdp4_tp2_ep2` |
|---|---|---|---|---|---|---|---|
| AllGather | <1KB | 8 | 24 | 24 | 24 | 40 | 24 |
| AllGather | 1-64KB | 2 | 12 | 16 | 4 |  |  |
| AllGather | 1-64KB | 8 |  |  |  | 4800 |  |
| AllGather | 64KB-1MB | 8 |  |  |  | 4800 |  |
| AllGather | 1-16MB | 2 | 8 | 16 | 8 |  |  |
| AllGather | 1-16MB | 4 |  |  |  |  | 16 |
| AllGather | 1-16MB | 8 |  |  |  | 153600 |  |
| AllGather | 16-256MB | 2 | 168 | 220 | 68 |  |  |
| AllGather | 16-256MB | 4 |  |  |  |  | 24 |
| AllGather | 16-256MB | 8 |  |  |  | 4800 |  |
| AllReduce | <1KB | 2 | 8 | 8 | 8 |  |  |
| AllReduce | <1KB | 4 |  |  |  |  | 8 |
| AllReduce | <1KB | 8 |  |  |  | 3208 |  |
| AllReduce | 1-64KB | 8 |  |  |  | 800 |  |
| AllReduce | 64KB-1MB | 2 |  | 36 | 12 |  |  |
| AllReduce | 1-16MB | 2 |  |  |  |  | 8 |
| Broadcast | <1KB | 8 | 8 | 8 | 8 | 24 | 8 |
| Broadcast | 1-16MB | 8 |  |  |  | 8 |  |
| Recv | 0 | 2 |  | 76 | 118 |  |  |
| Recv | <1KB | 2 |  | 112 | 262 |  | 16 |
| Recv | <1KB | 4 | 56 |  |  |  |  |
| Recv | <1KB | 8 | 7 | 7 | 7 | 14 | 7 |
| Recv | 1-64KB | 8 | 14 | 14 | 14 |  |  |
| Recv | 64KB-1MB | 2 |  | 284 | 22 |  |  |
| Recv | 64KB-1MB | 4 | 258 |  |  |  |  |
| Recv | 64KB-1MB | 8 |  |  |  | 28 | 14 |
| Recv | 1-16MB | 2 |  |  | 184 |  | 16 |
| Recv | 1-16MB | 4 | 74 |  |  |  |  |
| Recv | 1-16MB | 8 |  |  |  | 7 |  |
| Recv | 16-256MB | 2 |  |  |  |  | 16 |
| ReduceScatter | 1-64KB | 2 | 12 | 16 |  |  |  |
| ReduceScatter | 1-64KB | 8 |  |  |  | 4800 |  |
| ReduceScatter | 64KB-1MB | 8 |  |  |  | 4800 |  |
| ReduceScatter | 1-16MB | 2 | 6 | 12 |  |  |  |
| ReduceScatter | 1-16MB | 8 |  |  |  | 4800 |  |
| ReduceScatter | 16-256MB | 2 | 120 | 136 |  |  |  |
| ReduceScatter | 16-256MB | 8 |  |  |  | 76800 |  |
| ReduceScatter | 256MB+ | 2 | 14 | 20 |  |  |  |
| Send | 0 | 2 |  | 72 | 116 |  |  |
| Send | <1KB | 2 |  | 112 | 262 |  | 16 |
| Send | <1KB | 4 | 56 |  |  |  |  |
| Send | <1KB | 8 | 7 | 7 | 7 | 14 | 7 |
| Send | 1-64KB | 8 | 14 | 14 | 14 |  |  |
| Send | 64KB-1MB | 2 |  | 280 | 16 |  |  |
| Send | 64KB-1MB | 4 | 262 |  |  |  |  |
| Send | 64KB-1MB | 8 |  |  |  | 28 | 14 |
| Send | 1-16MB | 2 |  |  | 184 |  | 16 |
| Send | 1-16MB | 4 | 74 |  |  |  |  |
| Send | 1-16MB | 8 |  |  |  | 7 |  |
| Send | 16-256MB | 2 |  |  |  |  | 16 |

## Caveats

* **PCIe wallclock is unrepresentative.** Counts and tensor sizes are framework-determined and portable; latency / throughput would shift ~10× on NVLink-class interconnects.
* **Tier C is alignment-load (GBS=12).** Counts here are low-batch and don't reflect production overlap behavior. Use Tier A for replay decisions.
* **CP=2 traces missing.** kimi_linear's KDA layers' fla-core kernel doesn't support ring-recurrence over seq-sharded inputs. See `parallelize.py` CP branch.
* **TP plan is conservative.** Dense MLP only; KDA/MLA/AttnRes stay replicated. See `apply_tp_kimi_linear` docstring.
