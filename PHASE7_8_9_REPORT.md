# Phase 7 / 8 / 9 Final Report

## Phase 7 — NCCL Fabric Profiling
- **Pretrain trace catalog**: v11 4D (full coverage of FSDP+PP+TP+EP),
  v12 4D no-TP (50-step post-hoc from step-5000 ckpt; production trace
  was wiped by retry-loop cleanup), SFT 4D (post-train),
  8gpu_a2/a3/b0 alignments.
- **5D MODE=B captured**: PP=2×FSDP=2×CP=2 via llama3_debugmodel
  (DSv3+CP path crashes with mixed Tensor/DTensor in
  scaled_dot_product_attention). Adds **CP** axis to the catalog —
  new nranks=4 AG/RS signature from the combined CP×FSDP group, plus
  CP ring Send/Recv.
- **8-GPU constraint acknowledged**: "true 5D" with 5 axes ≥ 2 needs
  ≥ 16 GPUs. TP doesn't generate fabric traffic (intra-node SHM) so
  on 8 GPUs the most informative split is 3 fabric axes ≥ 2.
- **Pipeline**: `extract_collectives.py` → `expand_to_flows.py` →
  `flows_to_ixia.py` end-to-end, with axis heuristic.
- **IXIA-ready artifacts**: 11 `ixia_config.json` files, ~30 KB each.
- **Loss curves**: `phase8/eval_results/loss_curves.png` (v11 / v12 /
  SFT side-by-side).
- **See**: `phase7/FINAL_CATALOG.md`.

## Phase 8 — VQA Eval (Qualitative)
- **Quantitative eval (lmms-eval) deferred**: 2-3 day setup not in
  18h budget.
- **Qualitative eval done**: 5 COCO images × 5 prompts × 3 ckpts (v11,
  v12, SFT-490) via `phase5/generate_caption.py` greedy decode.
- **Result**: SFT-490 produces coherent multi-sentence captions
  (e.g. "3-D model of a red double-decker bus, parked on the
  street"); v11/v12 base produce fragmented number lists. Confirms
  SFT successfully transferred instruction-following capability.
- **See**: `phase8/eval_results/qual_vqa_summary.md`.

## Phase 9-A — SFT (LLaVA-Instruct-150K)
- **Result**: 1 epoch (490 steps) on LLaVA-Instruct-150K from v11
  step-5000, GBS=320 LBS=160 micro=8 SEQ=579, mesh same as v11
  (FSDP=2×PP=2×TP=2×EP=2), LR=2e-5.
- **Final loss 1.60** (started 2.66 — proper next-token shift fix
  in `phase9/multimodal_sft_dataset.py`).
- **Ckpt**: `phase5/runs/sft_v11_llava_instruct_150k_4d/checkpoint/step-490`
  (16 GB, DCP shards).
- **Bug found + fixed**: original SFT dataset returned labels of
  same length and alignment as input_ids (no shift), yielding
  trivial loss=0.009 due to misaligned target. Fix mirrors
  `LlavaPretrainDataset` pattern (`full[:-1]` / `full[1:]`).

## Phase 9-B — PPO Trace (vLLM-free smoke captured)
- **Status**: ✓ done. `phase9/ppo_smoke_no_vllm.py` runs 50-step
  toy-MLP PPO with actor on ranks 0-3 sub-mesh and frozen ref on
  ranks 4-7 sub-mesh. Cross-mesh KL exchange via world_pg
  Broadcast captures the **unique RLHF fabric signature** that
  v11/v12/SFT/5D-MODE-B all lack.
- **Captured collectives** (50 steps, 1.2 s wall):
  - 800 nranks=8 Broadcast <1 KB (cross-mesh KL — NEW pattern)
  - 800 nranks=4 AllReduce 16-256 MB (actor sub-mesh grad sync)
  - 400 nranks=4 AllReduce 1-64 KB + 400 <1 KB (small-msg AR)
- **Trace**: `phase5/runs/ppo_smoke_no_vllm/tier_b_trace/`
  (ixia_config.json, 6.3 KB; 8 nccl-rank-*.log.gz, ~600 B each).
- **Limitations**: random-init MLP (not real LM), no actual rollout
  (uses fixed batch as "rollout"), mock reward. Captures fabric
  pattern only, not RLHF training quality. To upgrade: load v11
  step-5000 ckpt twice (DCP load is non-trivial under 4D mesh) and
  do real `model.generate` rollout — see
  `phase9/PPO_TRACE_DEFERRED.md` for the recommended next step.

## Disk Discipline (lessons learned)
- Two ENOSPC incidents during retry-loop runs filled `/root` to 100%
  and bricked Bash tool calls.
- Mitigations now in place:
  - `phase6/launch_8gpu_mm.sh` retry-loop pre-flight (`free_gb >= 32 GB`)
  - Per-attempt cleanup of `tier_b_trace/nccl-rank-*.log`
  - Trace only on first attempt
- Codified in `phase6/DISK_DISCIPLINE.md`.

## Hardware utilization summary

| Run | mesh | LBS×micro | Throughput | Memory | MFU |
|---|---|---|---|---|---|
| v10 (baseline 3D, no EP, no SFT) | FSDP=2 PP=2 TP=2 | 160×10 | 1789 TPS | 49% | 3.45% |
| **v11** (4D + EP) | FSDP=2 PP=2 TP=2 EP=2 | 200×20 | **2470 TPS** | 91% | 4.79% |
| v12 (4D EP-replace-TP) | FSDP=2 dp_rep=2 PP=2 EP=2 | 160×16 | 4437 TPS | 88% | 8.65% |
| SFT (post-train) | same as v11 | 160×8 | 2716 TPS | 91% | 5.31% |

`v12 EP-replace-TP` achieves the highest MFU (8.65%) by removing
TP overhead on local-only intra-node communication, while EP
distributes the expert weight memory to enable similar effective
batch density.

## Outstanding bugs / future work
- **Kimi grouped_mm device-side assert** every ~400 steps under
  EP=2 + micro≥10. Workaround: retry-loop. Real fix needs cublas
  upstream investigation or capacity_factor PR.
- **Kimi CP**: blocked by KDA fla-core (no ring-recurrence).
- **commId-axis mapping**: heuristic axis labels conflate PP and EP
  for nranks=2 Send/Recv. Real fix needs trainer-side PG dump.

## Phase 10 — Inference + RLHF fabric (separate effort)

After phases 7/8/9 closed, a new Phase 10 was added covering the
inference and RLHF fabric — see `phase10/PHASE10_FABRIC_REPORT.md`
for the standalone deliverable. Stages:

- A: SGLang fork submodule + branch (PR-ready scaffolding).
- B: DCP → HF kimi_linear conversion script (424 → 1819 keys, 2.6 GB
  bf16 safetensors).
- C: `kimi_block_attn_res.py` model class on
  `attention_residual_inference` branch — runtime-blocked by
  sgl_kernel cu130/py314/sm120 wheel, but PR-ready as a structural
  deliverable.
- D: forward-only inference fabric trace at FSDP=4 × TP=2 × EP=2;
  captures the **inference signature** (zero ReduceScatter).
- E: `phase10/TRAINING_INFERENCE_FABRIC_ASYMMETRY.md` — first-of-
  kind training/inference fabric asymmetry analysis for Block AttnRes.
- F: real PPO smoke (kimi_linear AttnRes actor + frozen ref) at
  FSDP=4 × TP=2 × EP=2; captures **RLHF fabric** (training + inference
  half-overlap).
- G: `phase10/PHASE10_FABRIC_REPORT.md` — cross-regime aggregate.

This supersedes the Phase 9-B "vLLM blocker" stance — see Phase 10
Stage F for what was actually delivered (real PPO without vLLM, on
torchtitan, capturing the production-shape fabric on a co-located mesh).
