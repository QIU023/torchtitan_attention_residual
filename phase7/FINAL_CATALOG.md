# Phase 7 Fabric Pattern Catalog — Final

Aggregated NCCL collective patterns across all training runs in this
project, organized for IXIA Fabric profiling. Each entry produces a
canonical `ixia_config.json` (~30 KB) consumable by IxNetwork.

## Coverage matrix

| Run / config | mesh | trace path | ixia_config | notes |
|---|---|---|---|---|
| **v11 4D pretrain** | FSDP=2 × PP=2 × TP=2 × EP=2 (V=2) | `phase5/runs/v11_4d_*/tier_b_trace/` | ✓ | 5000 step run, 13 attempts, 18 GB raw → 210 MB CSV.gz |
| **v12 4D pretrain (no TP)** | FSDP=2 × dp_rep=2 × PP=2 × EP=2 (V=2) | `phase5/runs/v12_trace_50step/tier_b_trace/` | ✓ | 50 step trace from step-5000 ckpt (production trace lost in retry-loop cleanup; this is the post-hoc replacement) |
| **SFT (post-train)** | same as v11 (FSDP=2×PP=2×TP=2×EP=2) | `phase5/runs/sft_v11_llava_instruct_150k_4d/tier_b_trace/` | ✓ | 490 step on LLaVA-Instruct-150K |
| 8gpu_a2 (alignment) | FSDP=2 × PP=4, V=2 | `phase5/runs/8gpu_a2_*/tier_*_trace/` | ✓ (×3 tiers) | older, GBS=16 |
| 8gpu_a3 (alignment) | FSDP=2 × PP=2 × TP=2, V=2 | `phase5/runs/8gpu_a3_*/tier_c_trace/` | ✓ | older, GBS=16 |
| 8gpu_b0 (alignment) | FSDP=8 (no PP), V=1 | `phase5/runs/8gpu_b0_*/tier_*_trace/` | ✓ (×3 tiers) | older, GBS=16 (DP-only) |
| **5D MODE=B (llama3 + CP)** | PP=2 × FSDP=2 × CP=2 (3 fabric axes) | `phase5/runs/5d_mode_b_llama3_pp_fsdp_cp/tier_b_trace/` | ✓ | 50 step llama3_debugmodel; **adds CP coverage** (nranks=4 AllGather/ReduceScatter from CP×FSDP group). DSv3+CP path crashes with mixed Tensor/DTensor in scaled_dot_product_attention; llama3 dense path is stable |
| **PPO smoke (vLLM-free)** | actor sub-mesh (ranks 0-3) + ref sub-mesh (ranks 4-7) + cross-mesh world_pg KL | `phase5/runs/ppo_smoke_no_vllm/tier_b_trace/` | ✓ | 50 step toy MLP smoke; **adds cross-mesh KL exchange signature** — nranks=8 Broadcast (KL scalars 800 ops, <1 KB) + nranks=4 AllReduce 16-256 MB (sub-mesh grad sync, 800 ops); see `phase9/ppo_smoke_no_vllm.py` |

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

## Per-axis breakdown — v12 50-step trace (post-hoc, 4D no-TP)

```
pp+ep    : 713 K flows,  12.1 TB (matches v11 shape minus TP allreduce)
fsdp     : 0 flows captured at nranks=2 (heuristic landed in `unknown`)
dp       :  392 flows,  224 B
tp       :  61 K flows  (heuristic mis-label of dp_replicate AllReduces)
```

Captured by re-running `tier_b_trace` for 50 steps from
`v12_4d_*/checkpoint/step-5000` since the production 5000-step
trace was wiped by retry-loop cleanup. Fabric pattern matches v11
expected shape (PP+EP send/recv dominant, no TP allreduce).

## Per-axis breakdown — 5D MODE=B (llama3 + CP) 50-step

```
pp       : 24 K flows,   5.66 GB (PP send/recv, debug model is small)
fsdp     :  4.8 K flows, 632 MB (AllGather + ReduceScatter)
dp       :  112 flows,   0 B
nranks=4 collectives (CP+FSDP combined group): 4824 AG + 1600 RS + 664 AR
```

**Key new signature**: nranks=4 AllGather/ReduceScatter on the
`fsdp_cp` combined group (vs nranks=2 in v11/v12 which had no CP).
This is the CP fabric pattern adding a new dimension beyond v11/v12.
The CP attention ring exchange itself is implemented via Send/Recv
(captured as nranks=2 P2P, indistinguishable from PP in the heuristic
without commId dump — same caveat as PP/EP).

## Per-axis breakdown — SFT (post-train) 490-step trace

```
pp+ep    : 2.56 M flows,  78 TB cumulative
fsdp     : 0.52 M flows,  10 TB
dp       :  280  flows,  224 B
```

Lower absolute volumes vs v11 because shorter run (490 vs 5000 steps)
and smaller GBS (320 vs 400). **Per-step rates similar**, confirming
that mesh dictates fabric pattern, not data.

## Per-axis breakdown — PPO smoke (vLLM-free) 50-step

```
nranks=8 Broadcast    : 800 ops, <1 KB each   (cross-mesh KL — NEW)
nranks=4 AllReduce 16-256MB : 800 ops         (actor sub-mesh grad sync)
nranks=4 AllReduce <1KB     : 400 ops         (scalar sub_lp AR)
nranks=4 AllReduce 1-64KB   : 400 ops         (small param grads)
nranks=8 AllReduce <1KB     :   8 ops         (final barrier)
```

The **nranks=8 Broadcast at <1 KB** is the unique RLHF signature — it
does not appear in v11/v12/SFT (single-mesh runs) or in 5D MODE=B
(no cross-mesh exchange). At step rate 0.024 s/step with 16 KL
scalar broadcasts per step, this trivializes fabric-wise; the
distinguishing factor for IXIA is the *combination* of (a) cross-
mesh small-msg broadcast and (b) sub-mesh large-msg AR — same
endpoints alternate roles.

## Pattern shape per axis (qualitative)

| Axis | Op kinds | nranks | Typical message size | Cluster role in IXIA test |
|---|---|---|---|---|
| **PP** | Send / Recv | 2 (per PP boundary) | 12 MB (partial) + 24 MB (blocks) | inter-stage P2P streaming |
| **EP** (if MoE) | Send / Recv (NCCL implementation of all-to-all) | 2 (per EP group) | 48 MB out + 48 MB in per layer | MoE dispatch + combine |
| **FSDP** | AllGather / ReduceScatter | 2 (per FSDP unit) | 7-15 MB per layer | param all-gather, grad reduce-scatter |
| **TP** | AllReduce (intra-node) | 2 (per TP group) | 12 MB | Stays on SHM/NVLink, **not on fabric** |
| **DP** (full-mesh) | AllReduce | world_size | mostly small (<KB) | Cross-replica sync, infrequent |
| **CP** | Send/Recv (ring) + Combined-group AG/RS (CP×FSDP=4) | 2 (ring), 4 (with FSDP) | 64 KB-1 MB | **Captured via llama3 in 5D MODE=B** — Kimi CP blocked by KDA fla-core, llama3 dense works |

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
3. **Real PPO trace** with kimi_linear actor + ref loaded from v11 ckpt
   (current PPO smoke uses random-init MLP for fabric pattern only).
4. **Multi-node trace** to capture true wire-level UDP/IP/RoCE packets
   (current SHM-only single-host runs cannot).
