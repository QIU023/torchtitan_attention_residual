# Phase 11 — Multimodal RLHF on Block AttnRes

## Goal

End-to-end VLM RLHF pipeline driven by torchtitan + Monarch
(trainer / generator / grader as separate actors), with **SGLang** as
the rollout engine — proving the engine-agnostic Generator interface
proposed in the upstream RFC
(`torchtitan/torchtitan/experiments/rl/RFC_SGLANG_GENERATOR.md`)
works for production multimodal RLHF.

The RLHF method is GRPO (group-relative PPO without critic). The
framework is method-agnostic — the same wiring runs PPO if you swap
in a critic-bearing trainer.

## Layout

| File | Purpose |
| --- | --- |
| `llava_caption_task.py` | Dataset + reward function (BLEU-1 + length + format) over LLaVA-Pretrain-558K |
| `run_grpo_llava_caption.py` | Top-level entry point: spawns trainer / SGLang generator / grader on disjoint GPU meshes |
| `README.md` | This file |

## GPU layout (8× RTX 5090)

```
ranks 0..3 = PolicyTrainer mesh   (FSDP=4)
ranks 4..7 = SGLangGenerator mesh (TP=4)
Grader     = CPU (rule-based reward, no GPU)
```

Cross-mesh transport:
* **Episodes**: Monarch RPC across actor meshes
* **Weights**: torchstore-direct-RDMA (default) OR DCP→HF disk
  (set via `SGLangGenerator.Config.weight_sync_method`)

## Reward

Verifiable, no separate reward model:

```
length_ok  = 5 ≤ #tokens ≤ 30                  (else r = -1.0)
content    = BLEU-1(completion, gold_caption)  ∈ [0, 1]
format     = +0.2 if starts capital, ends "."
r          = length_ok ? content + format : -1.0
```

We use the gold LLaVA caption as the reference. This frames the RLHF
objective as "stay close to the supervised distribution + format
constraints" — the simplest production-style verifiable reward for
VLM RLHF.

## Run

```bash
# Required env (recorded for reproducibility):
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=phase11/rlhf/trace/nccl-rank-%h-%p.log
export NCCL_DEBUG_SUBSYS=COLL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Run
python phase11/rlhf/run_grpo_llava_caption.py \
    --model-path /root/torchtitan_attention_residual/phase11/hf_aligned_447m_step12500 \
    --num-steps 50

# After: extract collectives + flows + ixia config
python phase7/extract_collectives.py phase11/rlhf/trace/
python phase7/expand_to_flows.py phase11/rlhf/trace/ --world-size 8
python phase7/flows_to_ixia.py phase11/rlhf/trace/ --world-size 8
```

## Status

| Component | Status | Notes |
| --- | --- | --- |
| `LlavaCaptionTask` (multimodal task) | ✅ | Validated: ~558K records load cleanly, reward function checks out |
| `SGLangGenerator` (engine-agnostic actor) | ✅ | Lazy-import + drop-in for VLLMGenerator |
| `run_grpo_llava_caption.py` (entry point) | ✅ | Wires trainer / generator / grader on 8 GPUs, GRPO logic in line |
| Multimodal SGLang model class (Kimi AttnRes + SigLIP + projector) | ❌ | The remaining gap. Half-day of work; see "Multimodal model wiring" below |
| End-to-end run + NCCL trace | ⏸ | Blocked on the model class above; framework-only smoke (text-only path) ready |

## Multimodal model wiring (the remaining gap)

The current SGLang AttnRes overlay
(`sglang/python/sglang/srt/models/attn_res_overlay.py`) is text-only.
For multimodal rollouts to work end-to-end, we need a `VLM`
extension that mirrors what `phase5/multimodal_model.py` does on the
torchtitan side:

1. **Vision tower**: `google/siglip-base-patch16-224` frozen,
   produces 196 vision tokens per image.
2. **Projector**: 2-layer MLP `vision_dim → hidden_size`.
3. **Sentinel token replacement**: vision tokens are scattered into
   the prompt at the position of `IMAGE_TOKEN_ID = 32000` markers.
4. **SGLang wiring**: register the VLM class with
   `register_model_to_sglang_model_registry`, and forward the
   `image_data` request field through the engine.

Estimated: ~half-day. Concrete work items:
* Add `KimiAttnResVLMForCausalLM` class in
  `sglang/python/sglang/srt/models/attn_res_overlay.py` (or a sibling
  `attn_res_vlm_overlay.py`).
* Reuse SGLang's existing `multimodal/processors/llava.py` pattern
  for image-to-token-id pre-processing.
* Bump SGLang submodule + add a smoke test
  (`run_grpo_llava_caption.py --text-only` validates everything
  except image tokens already; once VLM class lands, drop the flag
  to enable multimodal).

## Why GRPO not PPO

GRPO drops the critic — group mean reward replaces the value
function baseline. Same fabric pattern as PPO minus one gradient-
exchange path. We pick GRPO because:

1. **Half the actors** (no value-net trainer) → cleaner NCCL trace
   to inspect.
2. **No critic-warmup phase** — the loop is a single block diagram.
3. **Verifiable reward** (BLEU-1 not value-net) — simpler to reason
   about reward shaping.

The framework is method-agnostic: changing to PPO is a Trainer-side
change (add a `ValueModel` actor), not a Generator/Grader change.

## Trace pattern expectations (NCCL fabric output)

Once running, we expect the trace to show:

* **Trainer mesh** (ranks 0-3): FSDP-style ReduceScatter + AllGather
  of weights and gradients each step.
* **Generator mesh** (ranks 4-7): TP-style AllReduce around each
  attention/MLP forward; under our seq-shard path also
  ReduceScatter+AllGather (from the Block AttnRes work).
* **Cross-mesh**: Send/Recv (Monarch) for Episode and weight
  transfer. With torchstore RDMA this is one-sided RDMA reads
  (NCCL-equivalent ops in the trace, plus low-level UCX/IBV).
* **Per step**: rollout phase (generator-mesh-only) then training
  phase (trainer-mesh-only); the meshes alternate so the trace
  shows clear phase-banding instead of overlap.

The PPO/GRPO contrast is mostly an extra critic AllReduce per
training step in PPO (under the trainer mesh). We can't easily
toggle PPO↔GRPO without a critic actor implementation, but with
GRPO we still get the multi-mesh + cross-mesh signature that
distinguishes RLHF from pure pretrain/SFT traces (those are
single-mesh).
