# Session handoff -- all changes + context (2026-07-23)

Authoritative pull-target for the 8x5090 session (2026-07-21 .. 07-23). One
`git pull` on each repo below gets the full change set; this doc is the context
map. Detailed docs are linked inline.

## 0. Repos to pull (all clean + pushed)

| repo | remote | branch | HEAD |
|---|---|---|---|
| logbook | QIU023/torchtitan_attention_residual | `main` | `39fe2c4` |
| torchtitan | QIU023/torchtitan | `attention_residual_dev` | `f76b3ae9` |
| verl | QIU023/verl | `kimi_k3_integration` | `c2c2c5a8` |
| sglang | QIU023/sglang | `attention_residual_inference` | `e45f675` (yours; unchanged this session) |

The logbook pins the verl + torchtitan submodules to these SHAs.

## 0b. Environment (exact versions -- reproduce before diagnosing drift)

Box: 8x NVIDIA RTX 5090 (32 GB, PCIe, no P2P/NVLink), driver **580.159.03**.
Three venvs (the titan trainer and the sglang rollout are deliberately split):

| venv | purpose | python | torch | key libs |
|---|---|---|---|---|
| `/venv/main` | torchtitan trainer / actor | 3.12.13 | **2.12.0+cu130** (cuda 13.0) | fla-core **0.5.1**, torchao **0.17.0** |
| `/venv/verl` | verl SFT/RL engine | 3.12 | **2.12.0+cu130** | verl @ `c2c2c5a8` (editable, submodule) |
| `/workspace/sgl_venv` | sglang rollout server | 3.12 | **2.11.0+cu130** | transformers **5.6.0**, sglang @ `e45f675` |

The 2.11-vs-2.12 split is why the GRPO rollout is an EXTERNAL HTTP server
(section 4): sglang pins torch 2.11, the titan/verl trainers run 2.12. torch
wheels bundle their own CUDA runtime (cu130), independent of the box driver.
On H200: rebuild the same trio (or unify verl on a K3-capable vLLM venv at
7.27). torchao 0.17 has no weight-only MXFP4 linear yet -> the MXFP4-base LoRA
path dequant-matmuls (lora.py).

## 1. Change inventory -- FINAL (durable K3 training support) vs TEMPORARY (scaffolding)

**FINAL -- directly supports K3 pretrain / SFT / RL, LoRA + full-param, distributed (the RFC/PR product):**
- **torchtitan `experiments/kimi_k3/`** (HEAD f76b3ae9): the K3 training stack.
  New this session: **CP** for hybrid KDA/MLA (f4b6f46f land + ec417b21/48285050
  head-shard; parity cp2-vs-cp1 6e-4); **LoRA x TP** so LoRA composes with ALL
  axes FSDP/TP/EP/PP/CP+4D (f76b3ae9); **MXFP4-base LoRA** split-storage
  (1ca1bd39). Standing: model.py (KDA/MLA/MoE), attn_res, parallelize.py,
  pipeline_adapter.py, state_dict_adapter.py, lora.py, muon.py,
  quantile_balance.py, mxfp4_qat.py + unit tests.
- **torchtitan `models/common/moe.py`** routing-map scatter fix (129e29de) --
  CORE, fixes any MoE under TP+EP. Flag as a standalone upstream PR.
- **verl `workers/engine/torchtitan/`** (HEAD c2c2c5a8): the torchtitan actor
  engine. New this session: **`interval=1` checkpoint fix** (c2c2c5a8) -- a real
  verl bug: the engine never set torchtitan's checkpoint interval, so
  `trainer.save_freq` silently wrote NO model weights for any run < 500 steps.
  Worth a verl PR.

**TEMPORARY -- scaffolding / demos / pre-7.27 bridges (will be superseded, do NOT productize):**
- `phase11/grpo_titan_sglang_rollout.py`, `grpo_titan_standalone.py` -- GRPO
  MECHANISM demos. Superseded by verl-native GRPO + official K3 vLLM (see 4).
- sglang serve/install/run scripts (`sgl_persist_launch.sh`,
  `run_grpo_sglang.sh`, `sglang_serve_smoke.sh`, `install_sglang_isolated.sh`) --
  rollout scaffolding; obsoleted when official K3 inference lands.
- PP pressure-test launchers + probe harnesses (`run_overnight_pp_pressure_*`,
  `run_bands_*`, `quant_lora_ep_harness`, `kda_ulysses_cp_probe`,
  `mxfp4_lora_fsdp_real`, ...) -- verification EVIDENCE, not product code.
- The `seq_len=512` 48B-carrier PP8xVP4 setting is a 5090 memory workaround.

## 2. What ran / was verified on the 8x5090 this session

- **Full parallelism matrix**: full-param FSDP/TP/EP/PP/TP+EP/TP+PP/**4D** +
  **CP** (composes with FSDP/PP/EP + 4D) + **LoRA x every axis** +
  **MXFP4-base LoRA** under FSDP2. EP bit-exact to FSDP. Details:
  [EIGHTCARD_VERIFICATION_2026-07-21.md](EIGHTCARD_VERIFICATION_2026-07-21.md),
  [MXFP4_LORA_VERIFICATION_2026-07-21.md](MXFP4_LORA_VERIFICATION_2026-07-21.md).
- **Real 48B bf16 LoRA SFT (AttnRes graft flavor, GSM8K)**: ran to step 120,
  loss 0.62 -> 0.35, step-120 LoRA checkpoint at
  `/workspace/ckpt_48b_lora_gsm8k/step-120` (~92GB) -- proves graft+LoRA trains
  on real 48B weights (this is a box-local artifact; not committed).
- **PP cross-stage-adapter numerics REPRODUCED on the current fork** (the RFC's
  evidence, now commit-pinned): PP8xVP4 fixed (the earlier crash was CUDA OOM,
  not a bug -- 48B-carrier seq1024 OOMs on 5090 now, seq512 fits); every adapter
  |dLoss| within its config's naive-vs-naive band; the 07-19 pp4vp2=0.033 was a
  non-reproduced fluke (07-22 = 0.005).
  [PRESSURE_TEST_REPORT_2026-07-22.md](../phase3_attnres_pp_integration/PRESSURE_TEST_REPORT_2026-07-22.md).
- **GRPO mechanism with the sglang overlay as rollout**: full loop on titan K3
  194m -- rollout via the user's sglang server over HTTP (external, torch 2.11,
  no in-process import -> the venv split is bypassed), group advantages + titan
  actor policy update (ours), weight sync via actor->HF-disk +
  `/update_weights_from_disk`. 6 steps, PASS. [GRPO_STATUS_2026-07-20.md](GRPO_STATUS_2026-07-20.md)
  (2026-07-23 update at top).

## 3. RFC status -- READY to upload (issue #3029)

[RFC_K3_SUPPORT_DRAFT.md](RFC_K3_SUPPORT_DRAFT.md). Finalized this session:
numerics bullet re-pinned to the 2026-07-22 report (current fork) with the
honest band criterion; bullet-1 CP claim now cites
EIGHTCARD_VERIFICATION. All fork-evidence links verified to resolve (SHA
f76b3ae9a pushed; all code + report paths exist on origin/main). Honest
scoping intact: inference/serving is OUT of scope (official Moonshot vLLM);
GRPO/post-training framed as training-side exploration, not a landing claim.

## 4. GRPO -- honest positioning (why not to build the verl rollout adapter)

When official K3 lands in vLLM/sglang upstream (~7.27), the rollout side of
this GRPO work is superseded: verl adopts K3 by **installing/bumping to a
K3-capable vLLM in the verl venv** + stock `rollout.name=vllm` -- a dependency
bump (+ maybe a small verl vLLM-compat patch), NOT custom K3 rollout code. So
the custom external-sglang verl rollout adapter was deliberately NOT built (it
would be a soon-deprecated wheel). What survives: the titan actor engine + verl
NATIVE GRPO (compute_grpo_outcome_advantage) driving it. The ONE thing to freeze
at 7.27: the weight-sync tensor-name map (titan `get_per_tensor_param` -> `to_hf`
-> vLLM K3 class names). Full external-rollout contract (if ever needed) is
mapped in [GRPO_STATUS](GRPO_STATUS_2026-07-20.md) + H200_HANDOFF section 5.

## 5. What is left -> H200 (the box did everything feasible)

Blocked on 5090 by memory/speed/infra, open on H200
([H200_HANDOFF_2026-07-21.md](H200_HANDOFF_2026-07-21.md) has the runbook, Exp A-E):

- **Full-param 48B SFT + MXFP4-QAT**: technically runnable on 5090 (bf16 master
  12GB/GPU shards under FSDP8; fp32 optim 384GB offloads to CPU -- the box has
  503GB RAM; QAT's fake-quant is a forward op with NO meta-first storage
  blocker, unlike MXFP4-base-LoRA), but **too slow to be useful**: no
  P2P/NVLink means both the per-layer all-gather AND the CPU-offload optim step
  cross PCIe. -> H200 (NVLink + on-GPU optim).
- **MXFP4/NF4-BASE LoRA through the trainer**: torchtitan.train is meta-first
  (build-meta -> shard -> init) but quantized base needs quantize-THEN-shard
  (build-on-device, which needs 96GB on one GPU for 48B). The streaming-quantize
  half runs on 5090; the sharded-quantized-load trainer path is H200 infra.
- **GRPO end-to-end at scale**: 7.27 official vLLM + weight-name freeze.

**H200 sizing (important -- avoids OOM):**

| task | H200 count | note |
|---|---|---|
| full-param 48B SFT / MXFP4-QAT | **8xH200** (80G) or 4-8x (141G) | master 96G + grad 96G + fp32 optim 384G ~= 576G; on-GPU optim, no offload, NVLink |
| LoRA / QLoRA 48B SFT | 2xH200 | frozen base + small adapters |
| GRPO (post-7.27 vLLM) | co-location dependent | actor + vLLM rollout |

(H200_HANDOFF section 1 said "2xH200" for the LoRA/QLoRA line; full-param/QAT
needs 8xH200 -- do not open full-param on 2 cards.)

## 6. Non-negotiable honesty carries (from CLAUDE.md, keep intact)

- Never claim 2.8T was personally validated -- "validated on 48B real weights +
  K3-faithful topology; scale-out is config-level".
- Never present under-trained (random-init / few-step) results as competitive.
- K3's "<2% overhead" (algorithm FLOPs) != our PP-adapter "+2.7% step-time"
  (PCIe comms). The 5090 ~347s/step is an interconnect artifact, not algorithm
  cost.
- Provisional pieces (Gated-MLA gate form, Quantile Balancing, Per-Head Muon
  variant, MXFP4-QAT recipe, AttnRes N/placement, 2.8T dims) reconcile against
  the official weights + report + vLLM K3 class at 7.27.
