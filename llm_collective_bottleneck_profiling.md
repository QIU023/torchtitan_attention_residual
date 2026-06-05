# LLM Training Collective Communication: Scale-Out Network Bottleneck Profiling

**Status**: Design draft, archived for later execution
**Trigger to start**: torchtitan attention residual PR shipped + current interview pipeline (Fireworks, Waymo) closed
**Estimated effort**: 6–8 weeks part-time
**Estimated cost**: ~$300–500 (multi-node GPU rental)

---

## 1. Goal

Produce a hands-on profiling study of how LLM training collective communication (across FSDP / TP / PP / SP / EP and their 2D/3D combinations) stresses scale-out networks. Identify, quantify, and visualize the dominant bottlenecks, and provide a reusable methodology for going from PyTorch-level training code down to wire-level packet observation.

Output is a public technical artifact (blog post + GitHub repo) targeted at ML infra hiring at Fireworks / Together / CoreWeave / Lambda / Crusoe / SF Compute / Meta AI Platform / Microsoft Singularity / Apple AIML Platform / NVIDIA DGX Cloud / hyperscaler ML platform teams.

The differentiator is the rare combination of:
1. Hands-on torchtitan parallelism tuning across a wide config matrix
2. Trace-driven replay onto a real packet-forwarding testbed (not pure simulation)
3. End-to-end visibility from collective op down to packet-level behavior

---

## 2. Framing principles (read every time before working on this)

- **Frame the project around bottleneck taxonomy, not around any specific fabric technology.** Do not name SRD, RoCE, IB, or any commercial fabric in any public artifact.
- **Refer to the testbed only as "a custom traffic-generation environment with multipath capability."** No topology details, no vendor names, no protocol identifiers, no parameter values.
- **The contribution is the methodology and the bottleneck observations, not the fabric.** Any modern scale-out network exhibits the same bottleneck classes; the relative magnitude differs but the taxonomy is universal.
- **No internal information may appear in trace files, code, or writeup.** Synthesized data and aggregate measurements only.

---

## 3. Non-goals

- Not building a new simulator. Use existing tools (ASTRA-sim, Chakra) where useful, otherwise lightweight in-house glue.
- Not chasing peak throughput numbers. Goal is bottleneck characterization, not performance bragging.
- Not advocating any specific fabric design. Comparisons are between configurations of the same testbed (multipath on/off, hash entropy variants), not between fabric technologies.
- Not training to convergence. 100–500 steps per config is enough to capture stable collective patterns.
- Not publishing raw packet captures. Only aggregate measurements and synthesized traces.

---

## 4. Resources available

- 8× RTX 5090 PCIe (no NVLink, PCIe Gen5 x16) — for single-node trace collection
- Access to a custom traffic-generation testbed with multipath capability — for trace replay and observation
- Budget for short-term multi-node rental on Vast.ai / Lambda / RunPod (H100 or A100, 2–4 nodes for a few hours)

---

## 5. Bottleneck taxonomy under study

This list is the spine of the entire writeup. Each bottleneck class gets at least one configuration that triggers it, one measurement, and one paragraph in the blog.

| # | Bottleneck | Triggered by | Measurement |
|---|------------|--------------|-------------|
| B1 | Incast (fan-in congestion) | reduce_scatter, all_reduce ring final stage | p99/p99.9 latency at receiver, drops at last-hop switch |
| B2 | Hot-spot uplinks (ECMP hash skew) | Long-lived elephant flows, low entropy 5-tuple | Per-uplink utilization variance, theoretical vs achieved spread |
| B3 | Tail latency under collective bursts | All collective types at burst boundaries | p50 vs p99 vs p99.9 gap during bursts |
| B4 | Buffer occupancy spikes | Synchronized collective starts across ranks | Switch buffer high-watermark, duration above threshold |
| B5 | Recovery from in-flight failures | Failure injection during collective | Time to fully drain in-flight + completion time delta |
| B6 | Multipath effectiveness | Compare multipath enabled vs disabled | Aggregate goodput delta, path count actually utilized |
| B7 | Algorithm-fabric mismatch | NCCL ring vs tree vs PAT under same trace | Completion time delta per algorithm |
| B8 | All-to-all tail behavior | MoE EP configs | Per-pair completion time distribution, worst-pair vs average |

---

## 6. Phases

### Phase 1 — Single-node trace collection (Week 1)

On 8×5090 PCIe, run torchtitan with the configuration matrix below. Use a small Llama-3 or Qwen2 base model that fits with activation checkpointing. For MoE configs, use a Mixtral-style or DeepSeek-V2-lite-style small MoE.

Each config: 100 steps, fixed seed, fixed batch size.

| Config | PP | TP | FSDP | SP | EP | Purpose |
|--------|----|----|------|----|----|---------|
| C1  | 1 | 1 | 8 | 1 | – | Pure FSDP baseline |
| C2  | 8 | 1 | 1 | 1 | – | Pure PP, P2P only |
| C3  | 4 | 1 | 2 | 1 | – | 2D: PP+FSDP, PCIe-friendly |
| C4  | 2 | 1 | 4 | 1 | – | 2D: PP+FSDP, alt split |
| C5  | 1 | 2 | 4 | 1 | – | 2D: TP+FSDP, deliberately TP on PCIe |
| C6  | 2 | 2 | 2 | 1 | – | 3D: PP+TP+FSDP minimal |
| C7  | 2 | 2 | 2 | 2 | – | 3D + SP |
| C8  | 1 | 1 | 4 | 1 | 2 | MoE: EP+FSDP |
| C9  | 2 | 1 | 2 | 1 | 2 | MoE: PP+FSDP+EP |
| C10 | 2 | 2 | 2 | 1 | 2 | Full 3D + EP (stretch) |

For each config, capture:
- NCCL trace via `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,INIT,GRAPH`
- PyTorch profiler distributed trace (`record_shapes=True, with_stack=True`)
- NCCL chosen algorithm (ring/tree/PAT) per collective
- Wall-clock per step, broken down by phase

Output: 10 trace bundles, each containing collective op sequence with type, size, participating ranks, timestamps, and chosen algorithm.

### Phase 2 — Multi-node trace collection (Week 4)

Rent 2× 8-GPU nodes (H100 or A100) for a few hours. Goal is to capture cross-node collective patterns that single-node cannot produce. The rented environment's underlying fabric does not matter for our purposes — we extract only the **logical collective sequence**, which is fabric-independent.

| Config | Cross-node parallelism | Intra-node | Captures |
|--------|------------------------|------------|----------|
| M1 | FSDP=16 | – | Cross-node FSDP all_gather/reduce_scatter |
| M2 | PP=2 | FSDP=8 | Cross-node PP P2P + intra-node FSDP |
| M3 | EP=2 | FSDP=8 | Cross-node MoE all_to_all |
| M4 | PP=2 | TP=4, FSDP=2 | Full 3D realistic config |

Algorithm selection (ring vs tree vs PAT) chosen by NCCL in the rented environment will be re-evaluated by our methodology under the testbed's bandwidth/latency model during replay.

### Phase 3 — Trace translation (Week 2 in parallel with Phase 1, finalized Week 5)

Build a translator: NCCL/PyTorch trace → testbed traffic-generation input.

Translator output per collective op:
- Op type (all_reduce, all_gather, reduce_scatter, all_to_all, send, recv)
- Message size in bytes
- Participating ranks (or process group)
- Logical timestamp (relative to step start)
- Algorithm hint (ring, tree, etc.) — methodology may override based on testbed model

Translator output per rank-to-testbed mapping:
- Rank 0..N-1 → endpoint identifier on testbed
- Down-sampling logic for cases where trace has more ranks than testbed can host (preserve collective topology, scale rank count proportionally)

### Phase 4 — Replay and observation (Week 3 initial, Week 5–6 full)

Inject translated traces into the testbed and observe per the bottleneck taxonomy in Section 5.

Measurements per replay run:
- Per-collective completion time
- Per-flow latency distribution (p50, p99, p99.9)
- Path utilization spread
- Buffer occupancy over time at observation points
- Aggregate goodput per collective vs theoretical ceiling

### Phase 5 — Ablation matrix (Week 6)

For each interesting trace, run under multiple testbed configurations to isolate the contribution of each network feature:

- Multipath enabled vs single-path baseline
- Hash entropy: low vs high (configured by varying flow tuple diversity)
- Buffer threshold: default vs aggressive (where tunable)
- Failure injection: drop a forwarding element mid-replay, measure recovery delta

For each ablation: produce a chart showing the bottleneck metric (B1–B8) under each variant.

### Phase 6 — Writeup and ship (Week 7–8)

Artifacts:
1. Blog post (~3000 words, with charts) on personal site or Medium
2. GitHub repo containing: trace translator code, ablation runner scripts, summary CSVs (no raw traces)
3. LinkedIn post linking the blog
4. Resume bullet under torchtitan project, linking the blog

Title candidates:
- "Where Does Your Training Time Go? A Bottleneck Profile of LLM Collectives on Scale-Out Networks"
- "Tracing LLM Training Down to the Wire: Eight Bottlenecks I Found"
- "From torchtitan to Packets: A Hands-On Profiling Study of Modern LLM Communication"

Blog structure (skeleton):
1. Why scale-out network behavior matters for LLM training (1 paragraph)
2. The methodology: trace collection + trace replay (1 section)
3. The configuration matrix and what each config stresses (1 section)
4. The eight bottleneck classes, one subsection each, each with a chart and a takeaway
5. What surprised me (the "story" section — pick 2–3 counterintuitive findings)
6. Reproducibility notes and code pointer
7. What this means for ML infra engineers in practice

---

## 7. Key technical risks and mitigations

| Risk | Mitigation |
|------|------------|
| NCCL trace too coarse to drive packet-level replay | Cross-check with PyTorch profiler; fall back to NCCL Profiler API (NCCL 2.20+) for finer granularity |
| Algorithm selection mismatch between rented env and testbed model | Methodology re-selects algorithm under testbed assumptions; record both choices in writeup |
| Testbed cannot host the rank counts in multi-node traces | Down-sample ranks while preserving collective topology; explicitly disclose in methodology section |
| MoE all_to_all setup hard in rented env | Use small MoE (DeepSeek-V2-lite or Mixtral-tiny variant); 100 steps is enough |
| torchtitan does not support all configs out of the box (e.g., EP) | Use Megatron-LM or DeepSpeed for the configs torchtitan misses; document the switch |
| Project drifts toward looking like a fabric-comparison study | Re-read Section 2 framing principles before every work session |

---

## 8. Compliance and desensitization rules

**This is the most important section of this doc. Re-read it before publishing anything.**

The testbed I have access to is a work resource. To keep this project entirely independent of my employer:

**MUST NOT appear in any public artifact (code, blog, repo, resume, LinkedIn, interview, conference talk):**
- Name or category of the underlying fabric technology (no SRD, no RoCE, no IB, no Ethernet variant identifiers)
- Vendor names of switches, NICs, or protocols
- Topology specifics: switch counts, port counts, oversubscription ratios, bandwidth values
- Configuration parameters, algorithm names, or feature flags from the testbed
- Any code or script from internal repositories
- Raw packet captures from the testbed

**MAY appear in public artifacts:**
- Generic descriptions: "scale-out network with multipath capability"
- Methodology: how trace collection and replay are structured
- Bottleneck taxonomy and measurements
- Open-source training and trace translation code (developed independently from scratch)
- Synthesized / normalized aggregate data
- Comparisons between configurations of the testbed (e.g., multipath on vs off), as long as configurations are referred to abstractly

**Pre-publication checklist (run before any public release):**
- [ ] No fabric, vendor, protocol, or product names mentioned
- [ ] No internal topology, scale, or parameter values
- [ ] All code in repo was written by me from scratch on personal hardware
- [ ] Charts show only normalized or synthesized data
- [ ] Blog reviewed by a trusted second pair of eyes for inadvertent disclosure

When in doubt, leave it out. The project's value is the methodology and bottleneck taxonomy — those alone are sufficient differentiation. Any fabric specifics add risk without adding value.

---

## 9. Resume / interview narrative after ship

Resume bullet (under torchtitan project or as standalone):
> Built an end-to-end profiling pipeline for LLM training collective communication: traced FSDP/TP/PP/SP/EP and their 2D/3D combinations on torchtitan, replayed the patterns on a custom traffic-generation testbed, and characterized eight bottleneck classes (incast, hot-spot uplinks, tail latency under bursts, multipath effectiveness, all-to-all tail behavior, etc.). [link to blog]

Interview talking points:
- "I built a full pipeline from torchtitan trace to wire-level replay. The fabric-agnostic part is the methodology — any modern scale-out network experiences the same bottleneck taxonomy, and I can show you measurements across eight of them."
- "PP is communication-light because it's pure P2P. FSDP and TP are not — and I have measurements showing how their bottleneck signatures differ on a real packet-forwarding testbed."
- "MoE all_to_all has very different tail behavior than ring all_reduce. The worst-pair completion time can be several times the average, and that has implications for how you'd schedule MoE training on any multipath fabric."

If asked "what testbed did you use?":
> "A custom traffic-generation environment I have access to with multipath forwarding capability. The methodology is fabric-agnostic — the bottleneck classes I'm characterizing apply to any modern scale-out network."

If pressed further:
> "I'd rather not get into specifics of the testbed itself; the project is about the methodology and the LLM training communication characterization, not about evaluating any particular fabric design."

This is a clean, honest, professional answer. No reasonable interviewer will push past it.

---

## 10. Open questions to resolve before kickoff

1. Does the testbed traffic generator support replaying from arbitrary trace files, or does it only generate parameterized synthetic patterns? (If only synthetic, the trace translator becomes more complex — it must produce patterns rather than literal trace replay.)
2. What is the rank capacity of the testbed? (Determines down-sampling strategy in Phase 3.)
3. What are the natural observation points for buffer occupancy and per-flow latency? (Determines what metrics are practically collectable for B1–B8.)
4. What configurations of the testbed are tunable (multipath on/off, hash entropy, buffer thresholds)? (Determines ablation scope in Phase 5.)

These should all be answerable from work-side documentation without engaging anyone formally.

---

## 11. Decision log

- **Why include TP on PCIe even though it's known to be slow?** Producing the quantitative evidence is the contribution. Most candidates only know "TP needs NVLink" as folklore; having measurements is differentiation.
- **Why include MoE configs?** MoE is the frontier-model standard. all_to_all is the most communication-intensive and most poorly studied collective. High signal-to-effort ratio.
- **Why blog instead of paper?** Faster ship, broader reach, hiring-aligned audience. Paper publication is not the goal; resume artifact is.
- **Why fabric-agnostic framing?** (a) Compliance: zero risk of disclosing internal information. (b) Reach: every ML infra hiring team values this work, regardless of which fabric they run. (c) Longevity: the methodology and bottleneck taxonomy outlive any specific fabric generation. (d) Positioning: "ML infra engineer who understands collective communication" is a stronger long-term identity than "engineer associated with fabric X."

---

## 12. Trigger to start

Resume work on this design doc when ALL of:
- torchtitan attention residual PR is merged or in late review
- Fireworks FDE pipeline closed (offer / reject / withdrawn)
- Waymo Perception pipeline closed or in stable waiting state
- At least one full week with no scheduled interviews

Until those conditions hold, this document stays archived. Do not partially execute.
