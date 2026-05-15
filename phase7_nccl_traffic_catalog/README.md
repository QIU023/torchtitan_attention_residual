# Phase 7 — NCCL collective communication pattern catalog under 3D parallelism

## Scope

Record NCCL collective sequences exhibited by torchtitan's parallelism
machinery under every meaningful 3D combination on the 8-GPU box. The
goal is a static **pattern catalog** — which collective ops fire, what
tensor shapes they carry, which rank topologies they involve, and how
they interleave with compute — usable as documentation for upstream
review and as a baseline for future kernel/scheduling work.

This is **not** a throughput benchmark. The 8-GPU box is PCIe Gen5,
and NVLink-class latency / bandwidth numbers are explicitly out of
scope. PCIe throughput would skew all wallclock figures by ~10× on
large all-gathers, but **pattern data (op type, size, participants,
ordering, overlap-with-compute) is invariant under the physical
interconnect** — it is determined by the framework's plan.

## Why multi-tier recording is necessary

A single small-batch alignment trace is **not enough** to characterize
a configuration's NCCL pattern. Per-config we record at multiple
**load tiers** because the following pattern attributes are
batch/seq-size-dependent and shift qualitatively across tiers:

| Attribute | Behavior across load tiers |
|---|---|
| **Tensor size per call** | scales linearly with LOCAL_BS × SEQ; small-batch traces understate the bytes-per-collective by 10-100× |
| **Microbatch count per step (under PP)** | PP send/recv frequency = (GBS / LOCAL_BS / dp) per step; production load may issue 5-10× more send/recv pairs than alignment load |
| **Compute–communication overlap** | small-batch compute is fast and exposes communication on the timeline; production compute hides comm in flight; the overlap diagram is qualitatively different |
| **Bucket/fragmentation behavior** | FSDP all-gather buckets and EP all-to-all dispatch buckets fill differently at production tensor sizes — small-batch may never trigger bucket overflow paths |
| **Standardization** | catalog must reflect the load Kimi-style production training actually issues, not a toy size |

So: for each 3D config we record **multiple trace files at different
load tiers**, and the catalog ranks them by realism so a downstream
replay (kernel autotuner, schedule planner, reviewer) can pick the
trace closest to their target workload.

## Trace load tiers (ranked by realism, most realistic first)

| Tier | GBS | LOCAL_BS | SEQ | Steps recorded | Use case |
|---|---|---|---|---|---|
| **A — production-standardized** | 384 | varies by mesh, ≥ 8/rank | 2048 (LM) or 260 + 1024-token vision (multimodal long-ctx) | 100 | Matches paper Table 2 (Kimi 436M) and is the closest stand-in for "what Kimi-NextGen-AttnRes actually runs at scale". **This is the headline trace** of phase 7. |
| **B — production-realistic** | 120 | 15 (FSDP=8) / scaled | 260 (multimodal v8/v9 recipe) | 50 | Matches what v8/v9 multimodal pretrain ran on the 4-GPU box. Lower wallclock than A, still production-scale tensors. **Recommended replay tier** when A is too expensive. |
| **C — alignment-load** | 12 | 1–3 (mesh-dependent) | 260 | 50 | Matches phase 6 A1 alignment recipe (GBS=12 LOCAL_BS=3 multimodal). **Free** — recorded as a slice on the same `torchrun` that produces phase 6's alignment claim. Cheapest, but tensor sizes and overlap timeline are not representative of production. |
| **D — smoke** | 4–8 | 1 | 128 | 20 | Sanity / functional check that NCCL trace capture itself works. Not a deliverable; useful for debugging the recording pipeline. |

**Rule for replay recommendation**: prefer **A > B > C > D**. When the
question is "how would Kimi-scale training behave on this config?",
A is the answer; B substitutes when A's wallclock is unaffordable;
C only answers "do these collectives fire at all?"; D only answers
"does our recording pipeline work?"

The `pattern_catalog.md` consolidator emits a `replay_priority`
column for each (config, tier) tuple so the user can pick a trace
file by recipe-realism.

## Relationship to Phase 6

Phase 6 is the merge-readiness deliverable for upstream torchtitan.
Phase 7 piggybacks on three of Phase 6's 8-GPU alignment runs to
collect Tier C traces **for free** — the same `torchrun` invocation
captures both an alignment loss curve and an NCCL trace slice. Two
extra control-group runs are Phase 7-only. Tiers A and B require
**dedicated phase-7 production-load runs** that are not part of
phase 6's deliverables.

| Phase 7 config | Phase 6 alignment run produces Tier C? | Tier A/B production runs |
|---|---|---|
| 1. FSDP=2 × TP=2 × PP=2 | yes (A3 alignment) | dedicated A + B run |
| 2. FSDP=2 × PP=2 × EP=2 (MoE) | yes (A6 full alignment) | dedicated A + B run |
| 3. FSDP=2 × PP=2 × CP=2 | yes (CP=2 stretch alignment) | dedicated A + B run |
| 4. TP=2 × PP=2 × EP=2 (FSDP=1) | phase 7-only alignment slice | dedicated A + B run |
| 5. FSDP=2 × TP=2 × EP=2 (PP=1) | phase 7-only alignment slice | dedicated A + B run |
| 6. **FSDP=8 (anchor baseline)** | yes (B0 alignment) | dedicated A + B run |

So per config: **3 traces** (A, B, C). Tier D smoke is one-shot for
the recording pipeline itself, not per-config.

## 8-GPU parallelism arithmetic

8 GPUs = 2 × 2 × 2 admits at most three non-trivial parallelism axes
(every axis ≥ 2). A fourth non-trivial axis (e.g. CP=2 stacked on top
of FSDP=2 × TP=2 × PP=2) requires ≥ 16 GPUs, which is out of scope.

Useful single-collective coverage per axis:

| Axis | Collectives observed |
|---|---|
| FSDP (DP) | all-gather (forward param unshard), reduce-scatter (backward grad shard) |
| TP | all-reduce (RowwiseParallel post-multiply), all-gather + reduce-scatter (SequenceParallel boundary) |
| PP | send + recv (per stage hop, possibly batched into a single isend/irecv pair) |
| EP | all-to-all (token dispatch into experts + result combine) |
| CP | all-gather of K/V tensors (Ring Attention forward), reduce-scatter on dQ/dK/dV (backward) |

The 6 configs above (5 3D + 1 baseline) hit every collective at
least twice across runs, giving a redundancy check on the trace.

## Configuration matrix

| # | Mesh | World | What it isolates | AttnRes-relevant? |
|---|---|---|---|---|
| 0 | FSDP=8 (PP=TP=EP=CP=1) | 8 | Pure DP anchor; baseline for all alignment claims | yes — Tier A here is closest to v8/v9 production multimodal pretrain |
| 1 | FSDP=2 × TP=2 × PP=2 | 8 | Standard 3D, no expert/seq parallelism | yes — exercises `AttnResProjection` TP plan + cache adapter under FSDP+PP |
| 2 | FSDP=2 × PP=2 × EP=2 | 8 | Adds all-to-all from MoE token routing on top of DP+stage | yes — verifies cache adapter delta interleaves with MoE expert dispatch correctly |
| 3 | FSDP=2 × PP=2 × CP=2 | 8 | Adds ring K/V all-gather from context parallel; multimodal long-ctx | yes — cache adapter delta tensor must shard along seq under CP |
| 4 | TP=2 × PP=2 × EP=2 (FSDP=1) | 8 | EP composing with TP without DP — pure tensor + expert routing | indirectly — control group for EP collective ordering when no FSDP all-gather is interleaved |
| 5 | FSDP=2 × TP=2 × EP=2 (PP=1) | 8 | EP + TP + DP without stage parallelism — pure compute parallelism | indirectly — control group for EP collective ordering when no PP send/recv is interleaved |

## Recording method

For each (configuration, tier), run the prescribed step budget with
the following env + artifacts. Same backbone (kimi_linear AttnRes 436M
flavor for non-EP configs; MoE flavor for EP configs), same seed=42,
same fresh init or step-8000 init (same across tiers within a config
so the catalog can compare tier-by-tier on the same model state).
Record into `phase7_nccl_traffic_catalog/traces/<config_id>/tier_<X>/`:

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=COLL,INIT \
NCCL_DEBUG_FILE=phase7_nccl_traffic_catalog/traces/<config_id>/tier_<X>/nccl-rank-%h-%p.log \
TORCH_NCCL_TRACE_BUFFER_SIZE=20000 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_USE_COMM_NONBLOCKING=1 \
nsys profile \
  --trace=cuda,nvtx,nccl,osrt \
  --output=phase7_nccl_traffic_catalog/traces/<config_id>/tier_<X>/nsys-%h-%p \
  --capture-range=cudaProfilerApi \
  torchrun ...
```

Plus inside the trainer, wrap each grad step in
`torch.profiler.profile(record_shapes=True, with_stack=False,
schedule=torch.profiler.schedule(wait=10, warmup=5, active=N))`
where N is tier-dependent (Tier A: 50; Tier B: 30; Tier C: 20).
This captures a Python-level overlap view at JSON form, dumped under
the same tier subdirectory.

## Artifacts per (configuration, tier)

```
phase7_nccl_traffic_catalog/traces/<config_id>/tier_<X>/
├── nccl-rank-*.log          # NCCL_DEBUG=INFO trace per rank
├── nsys-*.nsys-rep          # one nsys report per host
├── profiler-rank0.json      # torch.profiler trace (Chrome/Perfetto)
├── collective_summary.csv   # post-processed: (op, size, src_rank, dst_rank|group, count)
├── recipe.json              # GBS, LOCAL_BS, SEQ, steps_recorded, init source, model flavor
└── README.md                # tier rationale, what this trace represents, replay caveats
```

A per-config `phase7_nccl_traffic_catalog/traces/<config_id>/INDEX.md` lists all tiers
present and their relative realism rank, so a user opening that
directory immediately sees which trace to replay first.

Post-processing (`phase7_nccl_traffic_catalog/extract_collectives.py`, to be written) parses
the NCCL log into a structured CSV and emits a histogram of (op,
tensor_size_bucket).

## Final deliverable: pattern_catalog.md

After all configs × tiers are recorded, consolidate into
`phase7_nccl_traffic_catalog/pattern_catalog.md`:

* **Per-collective table per tier**: each unique (op, size_bucket,
  mesh_dim) with which (config, tier) hit it, and per-step count.
  Tier-A, Tier-B, Tier-C columns side-by-side so the **shift in
  tensor size between tiers is visible**.
* **Per-tier timeline diagrams**: for each (config, tier) an ASCII
  timeline of which collective fires when in a forward + backward
  step (drawn from nsys timeline). Tier A is the headline diagram
  per config.
* **Overlap regions**: which collectives overlap with which compute
  ops, **broken out per tier** (small-batch exposes communication
  that production hides, and vice versa).
* **Replay priority table**: a single ranked list of all available
  trace files across all configs/tiers, ordered by realism. The
  user opens this table to pick a trace for replay.
* **PCIe caveat**: explicit disclaimer that wallclock is uninterpretable
  on this hardware; pattern data is independent of interconnect.

## Out of scope

* Throughput / MFU / wallclock numbers — PCIe skews everything; the
  pattern is the deliverable, not the speed.
* NVLink-class collectives that don't appear on PCIe (e.g.
  intra-node NVLink P2P fast paths). All collectives recorded here
  go through the same NCCL software path regardless of physical link.
* 4D parallelism (FSDP × TP × PP × EP, etc.) — needs ≥ 16 GPUs.
* Kernel-level NCCL implementation — that's NCCL upstream, not our
  scope. We record what NCCL gets asked to do, not how it does it.

## Status

Format: `<tier> = pending | recording | done`.

| Config | Tier C (alignment-slice, free) | Tier B (production-realistic) | Tier A (production-standardized) |
|---|---|---|---|
| 0. FSDP=8 baseline | pending | pending | pending |
| 1. FSDP×TP×PP | pending | pending | pending |
| 2. FSDP×PP×EP | pending | pending | pending |
| 3. FSDP×PP×CP | pending | pending | pending |
| 4. TP×PP×EP (no FSDP) | pending | pending | pending |
| 5. FSDP×TP×EP (no PP) | pending | pending | pending |

| Tooling | Status |
|---|---|
| `phase7_nccl_traffic_catalog/nccl_trace_capture.sh` (env wrapper) | pending |
| `phase7_nccl_traffic_catalog/extract_collectives.py` (NCCL log → CSV) | pending |
| `phase7_nccl_traffic_catalog/configs/*.sh` (one per config × tier launcher) | pending |
| `phase7_nccl_traffic_catalog/pattern_catalog.md` (final consolidation, replay-priority table) | pending |
