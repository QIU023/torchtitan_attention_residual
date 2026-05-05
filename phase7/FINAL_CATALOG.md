# Phase 7 Fabric Pattern Catalog — Final

Aggregated NCCL collective patterns across all training runs in this
project, organized for IXIA Fabric profiling. Each entry produces a
canonical `ixia_config.json` (~30 KB) consumable by IxNetwork.

## Coverage matrix

| Run / config | mesh | trace path | ixia_config | notes |
|---|---|---|---|---|
| **v11 4D pretrain** | FSDP=2 × PP=2 × TP=2 × EP=2 (V=2) | `phase5/runs/v11_4d_*/tier_b_trace/` | ✓ | 5000 step run, 13 attempts, 18 GB raw → 210 MB CSV.gz |
| **v12 4D pretrain (no TP)** | FSDP=2 × dp_rep=2 × PP=2 × EP=2 (V=2) | (trace lost in retry-loop cleanup) | ✗ | 5000 step run, 23 attempts; pattern infer-able from v11 minus TP allreduce |
| **SFT (post-train)** | same as v11 (FSDP=2×PP=2×TP=2×EP=2) | `phase5/runs/sft_v11_llava_instruct_150k_4d/tier_b_trace/` | ✓ | 490 step on LLaVA-Instruct-150K |
| 8gpu_a2 (alignment) | FSDP=2 × PP=4, V=2 | `phase5/runs/8gpu_a2_*/tier_*_trace/` | ✓ (×3 tiers) | older, GBS=16 |
| 8gpu_a3 (alignment) | FSDP=2 × PP=2 × TP=2, V=2 | `phase5/runs/8gpu_a3_*/tier_c_trace/` | ✓ | older, GBS=16 |
| 8gpu_b0 (alignment) | FSDP=8 (no PP), V=1 | `phase5/runs/8gpu_b0_*/tier_*_trace/` | ✓ (×3 tiers) | older, GBS=16 (DP-only) |
| **5D MODE=B (DSv3)** | PP=2×FSDP=2×CP=2×TP=1+EP=2 | — | ✗ deferred | torchtitan train.py CLI not validated; post-mortem in phase7/run_5d_fabric_trace.sh comments |
| **PPO trace smoke** | actor 4D + ref + RM + critic | — | ✗ deferred | requires vLLM/monarch/torchstore; see phase9/PPO_TRACE_DEFERRED.md |

## Per-axis breakdown — v11 4D 50-step trace

```
pp+ep    : 75.5 M flows, 2,566 TB cumulative
fsdp     : 15.3 M flows,   329 TB
dp       :    5 K flows,    4 KB
```

**Important caveat**: heuristic axis labels conflate PP and EP because
NCCL Send/Recv at `nranks=2` are indistinguishable without commId
trace dump from torchtitan. To split EP from PP in any run, dump the
PG-axis-to-commId map from the trainer init and join post-hoc.
Documented in `phase7/expand_to_flows.py:_classify`.

## Per-axis breakdown — SFT (post-train) 490-step trace

```
pp+ep    : 2.56 M flows,  78 TB cumulative
fsdp     : 0.52 M flows,  10 TB
dp       :  280  flows,  224 B
```

Lower absolute volumes vs v11 because shorter run (490 vs 5000 steps)
and smaller GBS (320 vs 400). **Per-step rates similar**, confirming
that mesh dictates fabric pattern, not data.

## Pattern shape per axis (qualitative)

| Axis | Op kinds | nranks | Typical message size | Cluster role in IXIA test |
|---|---|---|---|---|
| **PP** | Send / Recv | 2 (per PP boundary) | 12 MB (partial) + 24 MB (blocks) | inter-stage P2P streaming |
| **EP** (if MoE) | Send / Recv (NCCL implementation of all-to-all) | 2 (per EP group) | 48 MB out + 48 MB in per layer | MoE dispatch + combine |
| **FSDP** | AllGather / ReduceScatter | 2 (per FSDP unit) | 7-15 MB per layer | param all-gather, grad reduce-scatter |
| **TP** | AllReduce (intra-node) | 2 (per TP group) | 12 MB | Stays on SHM/NVLink, **not on fabric** |
| **DP** (full-mesh) | AllReduce | world_size | mostly small (<KB) | Cross-replica sync, infrequent |
| **CP** | (not captured — KDA blocks) | — | — | Future work via fla-core ring-attention |

## How to consume the IXIA configs

```python
import json
cfg = json.load(open('phase5/runs/v11_4d_*/tier_b_trace/ixia_config.json'))
# cfg["topology"]: 8 endpoints with port + IPv4 + MAC
# cfg["trafficItems"]: 44-764 traffic items (aggregated by src,dst,axis)
# cfg["axisSummary"]: per-axis flow / byte / unique-pair counts
```

Each `trafficItem`:
```json
{
  "name": "fsdp_r0_to_r1",
  "type": "L2L3",
  "endpointSet": {"src": ["rank_0"], "dst": ["rank_1"]},
  "frameSize": {"type": "fixed", "value": 9000},
  "frameCount": 381683494,
  "burstStart_us": 0,
  "metadata": {"axis_guess": "fsdp", "bytes_total": 3435151437332}
}
```

Frame size defaults to **9000 (Jumbo)**; override with `--frame-size 1500`
in `phase7/flows_to_ixia.py` for standard Ethernet.

## Future work (catalog completion)

1. **commId-aware axis dump**: add a hook in torchtitan trainer that
   logs each PG's commId at init alongside its axis name. Patches
   `phase7/extract_collectives.py` to read commId from log lines and
   join with `phase7/comm_axis_map.csv` for accurate axis labels.
   Resolves the PP/EP heuristic conflation.
2. **CP trace** via fla-core ring-attention KDA (upstream PR).
3. **PPO trace** when vLLM/monarch/torchstore install is feasible.
4. **Multi-node trace** to capture true wire-level UDP/IP/RoCE packets
   (current SHM-only single-host runs cannot).
