# Phase 7 / 8 / 9 Final Report

## Phase 7 — NCCL Fabric Profiling
- **Pretrain trace catalog**: v11 4D (full coverage of FSDP+PP+TP+EP),
  SFT 4D (post-train), 8gpu_a2/a3/b0 alignments.
- **Pipeline**: `extract_collectives.py` → `expand_to_flows.py` →
  `flows_to_ixia.py` end-to-end, with axis heuristic.
- **IXIA-ready artifacts**: 9+ `ixia_config.json` files, ~30 KB each.
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

## Phase 9-B — PPO Trace (Deferred)
- **Status**: vLLM/monarch/torchstore not installed; 1-2 day setup
  blocker exceeds budget.
- **See**: `phase9/PPO_TRACE_DEFERRED.md` for setup checklist and
  recommended path forward.

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
