# GRPO Trace Run — Kimi-Linear AttnRes 447M VLM (Option C bf16 stack)

**Trace dir**: `phase11/rlhf/trace_grpo_kimi_attnres_20260515T074648`
**Captured**: 2026-05-15 ~07:46–08:07 UTC
**Hardware**: 8× RTX 5090 (Blackwell consumer, SM 12.0), single node
**Software stack**: torchtitan + monarch + sglang (fork `qiu023/sglang@dc154e785`)

This run is the **first end-to-end GRPO trace** of the Kimi-Linear AttnRes VLM family that produces non-zero reward signal (vs prior v16 hard-collapse to -1.0). Captured for two purposes:

1. **Post-training RL demo data** — verify GRPO pipeline functional on the SFT-step-3100 ckpt
2. **Infra / fabric traffic trace** — feed into Ixia scale-out simulation, characterize NCCL collective patterns under our current parallelism setup

---

## 1. Parallelism configuration

Eight GPUs split into two meshes:

| mesh | PP | DP | TP | EP | CP | ranks | GPUs | scale-out relevance |
|---|---|---|---|---|---|---|---|---|
| **trainer** (FSDP) | 1 | **4** | 1 | 1 | 1 | 0–3 (1/GPU) | 0–3 | **yes** — FSDP all-gather/reduce-scatter |
| **generator** (SGLang TP) | 1 | 1 | **4** | 1 | 1 | 4–7 (shared) | 4–7 | **no** — TP all-reduce is intra-node NVLink |
| **grader** (CPU) | 1 | 1 | 1 | 1 | 1 | spawn_procs default | — | — |
| **cross-mesh** (torchstore RPC) | — | — | — | — | — | trainer→store + store→generator | network | **yes** — full-model push/pull every step |

**`nranks=4` universally** in collective_summary.csv (930k rows) — both meshes are 4-rank process groups; no `nranks=8` cross-mesh collective. Cross-mesh communication is exclusively torchstore Send/Recv (= P2P RPC), 703k each direction.

**No PP, no EP** in this setup. The Ixia post-processor's `axis_guess: "pp"` label on 6 traffic items is a heuristic mislabel (it tags Send/Recv as PP because that's what PP looks like) — these are actually torchstore RPC flows, not pipeline traffic.

---

## 2. Reward trajectory (60 steps, 4 episodes/step)

GRPO **does NOT collapse** (vs v16 reward = -1.0 hard collapse before the inference fixes), but **does NOT show monotonic learning** either — reward bounces in a [-0.96, -0.72] noise band:

| range | reward_mean |
|---|---|
| step 0–9   | ≈ −0.83 to −0.94 (mostly initial) |
| step 10–29 | ≈ −0.78 to −0.97 (exploration band) |
| step 30–59 | ≈ −0.77 to −0.93 |

**Positive-reward episodes** (2/240 total):
* step 35: `reward=+0.000` (BLEU-1 = 0.5)
* step 45: `reward=+0.250` (BLEU-1 ≈ 0.625)

Consistent with prior qualitative-eval verdict: the step-3100 SFT ckpt produces coherent English captions but **not image-grounded** (e.g., "Hello, my name is chris…"). BLEU-1 vs gold caption is therefore near 0 → reward ≈ -1.0 with occasional luck. **GRPO is functional but signal-starved at this base-model quality**. Real learning requires stage-0 backbone completion + proper LLaVA-Pretrain stage-1 alignment first.

Loss is volatile (-19 to +8) — typical of GRPO with small batch (4 ep/step) and sparse positive reward (one outlier episode dominates the policy gradient).

---

## 3. NCCL collective summary

930,938 collective records aggregated from 8 per-rank NCCL logs.

### Op-type distribution (by total bytes)

| op | total bytes | share | dominant source |
|---|---|---|---|
| **AllGather** | **2.46 TB** | **86%** | FSDP forward+backward param gathers (trainer mesh) + generator TP weight materialization |
| **ReduceScatter** | 338 GB | 11.9% | FSDP backward gradient reduce-scatter (trainer) |
| **AllReduce** | 47 GB | 1.7% | TP all-reduce after attention/MLP (generator, intra-node) + small init |
| **Send/Recv** | ~700k each / ~700 kB | ~0% | torchstore push/pull RPC (cross-mesh) |
| Broadcast | 64 B | — | init/topology |

Total ≈ **2.85 TB across 60 steps ≈ 47.4 GB/step** averaged.

The AllGather dominance is the FSDP signature; the small Send/Recv volume tells us the cross-mesh weight transfer is bandwidth-cheap *per call* (torchstore deduplicates / chunks) even though it fires every step.

### Size-bucket distribution

| bucket | op count | character |
|---|---|---|
| 1–64KB    | 717,216 | high-frequency small ops (init, sync, small shards) |
| 64KB–1MB  | 123,692 | medium shards |
| 1–16MB    |  28,320 | sharded FSDP units |
| **16–256MB** | **61,680** | **large unsharded all-gathers — the FSDP "burst" pattern** |
| <1KB      |      30 | init only |

Bimodal: huge tail of small ops + a long tail of large multi-MB bursts. The 16–256MB bucket carries most bytes.

---

## 4. Fabric traffic implications (scale-out only)

For scale-out fabric (excluding TP intra-node NVLink), the relevant flows are:

* **FSDP DP=4** (trainer): AllGather + ReduceScatter between 4 ranks. With 447M params (894MB bf16), per-step FSDP forward+backward traffic ≈ 5–10 GB on the inter-node side.
* **torchstore RPC** (trainer → store → generator): once per step, ~894 MB up + ~894 MB down (×4 generator pullers if TP-sharded), total per-step ~5–10 GB.

**Sum**: roughly 10–20 GB/step of scale-out traffic. The 47 GB/step total in §3 is inflated by intra-node TP all-gathers (NVLink, not scale-out).

The trace artifacts (`collective_summary.csv.gz`, `flows.csv.gz`, `ixia_config.json`) are post-processed by `phase7/extract_collectives.py` → `expand_to_flows.py` → `flows_to_ixia.py`, matching the format of the prior PPO/GRPO sum-digits / qwen3 catalog. They can be fed directly into IxNetwork for fabric simulation.

---

## 5. Workarounds active during this run (Option C)

This trace was captured under the **full bf16 workaround stack** (no fp8) because that's the only currently-known-working configuration:

* `ATTNRES_MLA_FP32_FALLBACK=1` — fp32 eager MLA on prefill (workaround for UPSTREAM_PR_LIST #3 flashinfer_mla bf16 NaN on large AttnRes activations)
* `decode_attention_backend=torch_native` — eager SDPA decode (same root cause)
* `disable_cuda_graph=True` — torch_native has no cuda_graph (perf cost)
* `SGLANG_DISABLE_SHM_MM=1` — inline pickle for multimodal payloads (UPSTREAM_PR_LIST #1; avoid monarch-lifecycle SHM race)
* `SGLANG_FP8_IGNORED_LAYERS=attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts` — fp8 not used here but env is set defensively
* **torchstore Controller monkeypatch** — promote 5 sync `@endpoint` to async (monarch sync/async mix rejection in this torchstore 0.1.2 + monarch combo)
* **grader_mesh sys.path bootstrap** — propagate `phase11/rlhf` for `LlavaCaptionTask` pickle
* **AttnRes block-aggregation einsums → manual broadcast+sum** (cuBLAS bypass; bf16 unaffected)

This stack is what's enabling GRPO to run at all on this hardware. Performance ceiling: ~44.6 tok/s per generator (vs presumably 2–3× higher if flashinfer_mla worked and cuda_graph were on).

---

## 6. Comparison with prior catalog runs

| trace dir | model | task | meshes | per-rank NCCL log size | ops in collective_summary |
|---|---|---|---|---|---|
| `trace_ppo_sum_digits` | Qwen3-0.6B | PPO sum-digits | FSDP=4 + TP=4 | small | (smaller) |
| `trace_grpo_sum_digits` | Qwen3-0.6B | GRPO sum-digits 50 | FSDP=4 + TP=4 | ~400 B | (smaller) |
| `trace_grpo_qwen3_e2e_v0` | Qwen3-0.6B | GRPO sum-digits e2e | FSDP=4 + TP=4 | 16 KB | (~6k rows) |
| **`trace_grpo_kimi_attnres_…` (this run)** | **Kimi-Linear 447M AttnRes VLM** | **GRPO LLaVA-Pretrain caption** | **FSDP=4 + TP=4** | **40 KB / 504 KB** | **930,938 rows** |

Two notable differences vs prior catalog:

1. **Per-rank NCCL log size jumped 10–30×** for generator ranks (504 KB vs 16 KB) — the VLM/AttnRes path has far more collectives per forward (KDA per-layer + AttnRes pseudo-query carrier across blocks + MLA both prefill and decode paths + projector + vision tower).
2. **collective_summary row count jumped 150×** (930k vs ~6k) — same reason: more distinct collective patterns per step.

Same parallelism shape, vastly heavier per-step compute & traffic. This is the realistic post-training-RL fabric load for a small VLM with AttnRes.

---

## 7. Files in this trace dir

| file | size | description |
|---|---|---|
| `recipe.json` | 2.3 KB | run config snapshot (parallelism, workarounds, ckpt paths) |
| `run.log` | 56 KB | stdout/stderr of the full 60-step GRPO run |
| `nccl-rank-*.log.gz × 8` | 40–504 KB | per-rank NCCL `NCCL_DEBUG=INFO SUBSYS=COLL` logs |
| `collective_summary.csv.gz` | 288 KB | parsed collective op records (930k rows) |
| `flows.csv.gz` | 162 B | per-pair flow aggregation (only Send/Recv P2P — small here) |
| `ixia_config.json` | 4.8 KB | IxNetwork-ingestible traffic config (6 P2P flows) |
| `REPORT.md` | (this) | this report |
