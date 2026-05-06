# Phase 10 Stage L — Sustained Inference Workload Sweep

Production-volume fabric trace across 4 workload shapes. Same model
(kimi_linear_436m_block_attn_res_n4 from phase4 step-8000), same mesh
(FSDP=4 × TP=2 × EP=2 = 8 GPUs); only batch + seq + steps vary.

## Sweep configs

| Label | Batch | Seq | Steps | Effective tokens | Wall |
|---|---|---|---|---|---|
| `short_high_bs` | 16 | 256 | 200 | 819K | 55.4 s |
| `mid` | 4 | 1024 | 200 | 819K | 55.9 s |
| `long` | 2 | 4096 | 100 | 819K | 54.1 s |
| `prod` | 8 | 2048 | 100 | 1638K | 102.7 s |

## Per-workload fabric volume

| Workload | fsdp bytes | pp+ep bytes | Total | Per-step rate |
|---|---|---|---|---|
| short_high_bs | 892 GB | 7.36 TB | 8.25 TB | 41 GB/step |
| mid | 892 GB | 7.36 TB | 8.25 TB | 41 GB/step |
| long | 714 GB | 7.36 TB | 8.07 TB | 81 GB/step |
| prod | 1.25 TB | 14.7 TB | 15.96 TB | 160 GB/step |

## Per-token fabric rate

Normalizing by effective tokens (batch × seq × steps):

| Workload | Total bytes | Tokens | Bytes/token |
|---|---|---|---|
| short_high_bs | 8.25 TB | 819K | ~10 MB |
| mid | 8.25 TB | 819K | ~10 MB |
| long | 8.07 TB | 819K | ~10 MB |
| prod | 16.0 TB | 1638K | ~10 MB |

**Key finding**: per-token fabric rate is **~10 MB regardless of
batch/seq decomposition** at the kimi_linear_436m architecture scale.
The fabric volume scales linearly with token count, not with the
batch/seq breakdown. This is the cleanest fabric scaling
characterization in the catalog.

## Per-token throughput

| Workload | TPS (tokens/s) |
|---|---|
| short_high_bs | 14,786 |
| mid | 14,656 |
| long | 15,140 |
| prod | 15,953 |

~15K TPS across all configurations — the model is compute-bound at
this scale, not memory-bound. Fabric is not the bottleneck (network
serves ~10 MB × 15K TPS = 150 GB/s aggregate which fits within
single-host NVLink/SHM).

## Comparison to Stage D 50-step baseline

Stage D was 50 steps × seq=512 × bs=4 = 102K tokens, 928 GB pp+ep,
156 GB fsdp = ~1.08 TB total. Per-step rate ~22 GB.

Stage L's per-step rate is **2-7× higher** because of larger batch
or seq. Per-token rate is roughly the same (~10 MB), confirming
linear scaling.

## IXIA load testing implications

For sustained-inference IXIA modeling, use Stage L as the load
profile:
- Per-token: 10 MB across all axes (fsdp + pp/ep heuristic combined)
- Per-step varies from 41 GB to 160 GB depending on (batch × seq)
- Aggregate TPS: ~15K at this 436M / 4D mesh

Specific configurations (short_high_bs vs long) load fabric **identically
in cumulative bytes** but with very different **temporal distributions**:
short_high_bs fires more numerous smaller bursts; long fires fewer but
larger bursts. IXIA models can use this to test stress under both
short-burst and steady-large patterns.

## Files

* `phase10/run_workload_sweep.sh` — runner for all 4 workloads
* `phase5/runs/workload_short_high_bs/tier_b_trace/ixia_config.json`
* `phase5/runs/workload_mid/tier_b_trace/ixia_config.json`
* `phase5/runs/workload_long/tier_b_trace/ixia_config.json`
* `phase5/runs/workload_prod/tier_b_trace/ixia_config.json`

Each ~28 KB, immediately loadable into IxNetwork.
