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
export NCCL_DEBUG_FILE=phase11_rlhf_grpo_infra/rlhf/trace/nccl-rank-%h-%p.log
export NCCL_DEBUG_SUBSYS=COLL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Run
python phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_caption.py \
    --model-path /root/torchtitan_attention_residual/phase11_rlhf_grpo_infra/hf_aligned_447m_step12500 \
    --num-steps 50

# After: extract collectives + flows + ixia config
python phase7_nccl_traffic_catalog/extract_collectives.py phase11_rlhf_grpo_infra/rlhf/trace/
python phase7_nccl_traffic_catalog/expand_to_flows.py phase11_rlhf_grpo_infra/rlhf/trace/ --world-size 8
python phase7_nccl_traffic_catalog/flows_to_ixia.py phase11_rlhf_grpo_infra/rlhf/trace/ --world-size 8
```

## Status

| Component | Status | Notes |
| --- | --- | --- |
| `LlavaCaptionTask` (multimodal task) | ✅ | Validated: ~558K records load cleanly, reward function checks out |
| `SGLangGenerator` (engine-agnostic actor) | ✅ | Lazy-import + drop-in for VLLMGenerator |
| `run_grpo_llava_caption.py` (entry point) | ✅ | Wires trainer / generator / grader on 8 GPUs, GRPO logic in line |
| Multimodal SGLang model class (Kimi AttnRes + SigLIP + projector) | ❌ | The remaining gap. Half-day of work; see "Multimodal model wiring" below |
| End-to-end run + NCCL trace | ⏸ | Two blockers: (a) multimodal SGLang model class (below) and (b) a Monarch-worker / torch-install conflict (below) |

## Multimodal model wiring (the remaining gap)

The current SGLang AttnRes overlay
(`sglang/python/sglang/srt/models/attn_res_overlay.py`) is text-only.
For multimodal rollouts to work end-to-end, we need a `VLM`
extension that mirrors what `phase5_vlm_multimodal_sft/multimodal_model.py` does on the
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

## Env compat fixes (worker spawn)

Monarch's worker subprocess spawn doesn't inherit the parent's full
sys.path; we hit two cascading bugs and fixed both in
``Provisioner._bootstrap``:

1. **`torch._C is not a package`** — Monarch's pickle-deserialize
   handler triggers `import torch` at an unfortunate moment. torch's
   ``_jit_internal → torch.distributed.rpc → torch._C
   ._distributed_c10d`` chain doesn't survive that nested context
   (it's a known torch C-extension init quirk). **Fix**: pre-import
   torch + the full distributed stack in the bootstrap callback,
   *before* Monarch's mailbox handler runs.

2. **`_DeadlockError` on `torch.distributed._shard._utils`** — same
   nested-import context, but a different module lock. **Fix**: same
   pre-import pattern; bootstrap now imports the entire torch
   distributed surface plus the torchtitan modules the message
   handler will need.

Both fixes live in `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_caption.py` in the
`Provisioner.allocate` closure, with comments pointing at the
underlying torch issue. Once these are applied, the trainer mesh
spawns cleanly and PolicyTrainer instantiates.

## α implementation: lead/follower pattern (LANDED)

The friction below is now resolved by the lead/follower pattern in
`torchtitan/experiments/rl/actors/sglang_generator.py`:

* **Provisioner** has two modes — `allocate(n)` (one-actor-per-GPU,
  used for trainer) and `allocate_shared(n)` (all actors in mesh
  see the same N GPUs, used for generator).
* **SGLangGenerator** detects rank: rank-0 is "lead" and constructs
  the Engine; ranks > 0 are no-ops returning `[]` from `generate`
  and skipping `pull_model_state_dict`.
* **Engine async API**: switched from sync `Engine.generate` /
  `update_weights_from_disk` (which call `loop.run_until_complete`
  on their own loop and conflict with Monarch's running endpoint
  loop) to `async_generate` / `tokenizer_manager.update_weights_from_disk`.

Verified the pipeline boots end-to-end: 4 generator actors with
3 followers idling and 1 lead spawning a TP=4 SGLang Engine, all
seeing CVD=4,5,6,7 with 33 GB free per GPU. Generator produces
rollouts, grader scores them, GRPO advantage is computed.

## Last torch-version blocker: `varlen_attn` is nightly-only

After lead/follower is in place, `trainer.step()` runs and hits:

```
NotImplementedError: torch.nn.attention.varlen.varlen_attn is
unavailable on this PyTorch build. Upgrade torch (≥2.10 nightly)
or switch to a non-varlen attention impl.
```

torchtitan's RL trainer uses `varlen_attn` from `torch.nn.attention.varlen`
to compute per-token log-probs over packed prompt+completion
sequences. The function exists only on torch nightly (~2.10+).
On our pinned torch 2.9.1+cu129 stable (required by sgl_kernel ABI),
this is unavailable.

This is the same env-compat boundary documented in
`phase11_rlhf_grpo_infra/TORCHTITAN_VAST_AI_PATCHES.md`. The trainer can't be
patched around — it depends on the actual `varlen_attn` kernel for
correct logprob math, not just a feature flag.

Three paths to unblock:
1. **Switch to torch nightly** on this box — accept the cost of
   re-installing sgl_kernel against nightly's ABI (likely needs a
   source build, ~2-4h).
2. **Re-implement `varlen_attn` for torch 2.9** in the env-compat
   patch — replace with FlashAttention-2 packed-sequence kernels.
3. **Use a non-varlen RL trainer**. Upstream's PolicyTrainer
   asserts varlen, but a fork could substitute a standard
   attention path. Real engineering, ~1d.

The architectural work — engine-agnostic Generator + lead/follower
+ multimodal task wiring — is upstream-PR-ready independent of this
torch version pin.

## Original architectural friction (resolved)

After threading model_spec through the upstream pattern, we hit a
real architecture mismatch: Monarch's `per_host={"gpus": N}` spawn
creates N actor processes (one per GPU). vLLM's RL plugin handles
this via `distributed_executor_backend="external_launcher"`, where
each Monarch actor IS one vLLM worker.

SGLang's Engine doesn't have an external-launcher mode — it spawns
its own TP worker subprocesses internally. So our 4-GPU generator
mesh ends up with 4 actor processes, each trying to start an
Engine with `tp_size=4`, producing 16 total SGLang TP workers
fighting for the same 4 physical GPUs.

Diagnosis is conclusive: each generator actor sees CVD=4,5,6,7,
torch.cuda.device_count()=4, all 4 GPUs at 33 GB free. The OOM is
*after* SGLang spawns its inner TP workers and they collide.

Two paths to resolve:

* **(α) Single-actor with all GPUs**: `per_host={"gpus": 4}` but
  ONE actor process that has all 4 GPUs visible. Then SGLang
  Engine spawns its 4 TP workers without fighting other actors.
  Needs Monarch-specific option (or wrapper actor that ignores
  per-GPU process granularity).

* **(β) HTTP-server SGLang**: deploy SGLang as a service on the
  generator host, have a single Monarch actor call its HTTP API
  for rollouts. Cleaner architecturally; +1 process to manage.

Both are upstream-RFC-worthy: any framework wanting to support
SGLang as a peer of vLLM needs to pick one of these. The
`SGLangGenerator` actor and `SGLangEngine` wrapper we landed are
unchanged either way — only the spawn topology differs.

For this session we stop here; the framework + the multimodal task
+ the engine-agnostic generator are all in place. End-to-end RLHF
trace requires the topology fix above.

## Remaining gap: model_spec wiring

The framework works end-to-end *up to* `PolicyTrainer.__init__`,
which then fails with:

```
AttributeError: 'NoneType' object has no attribute 'state_dict_adapter'
```

This is a real config issue — we pass `model_spec=None`. The upstream
`config_registry.py` shows the expected pattern:

```python
from torchtitan.models.qwen3 import model_registry
config.model_spec = model_registry("0.6B_varlen")
config.hf_assets_path = ".../Qwen3-0.6B"
```

To run end-to-end we need ONE of:

* **(a) Qwen3-0.6B path (fast smoke)** — download the HF ckpt, set
  `model_spec = qwen3.model_registry("0.6B_varlen")`. Validates the
  framework + emits a clean NCCL trace, but uses Qwen3 not our 447m
  AttnRes. ~30 min including download.

* **(b) 447m AttnRes path (production)** — register a `model_spec`
  for our Kimi Linear AttnRes config that PolicyTrainer can build.
  This requires:
    1. A `model_registry("447m_aligned_block_attn_res")` entry in
       `torchtitan/torchtitan/experiments/kimi_linear/config_registry.py`
       returning a `ModelSpec`.
    2. The trainer's `parallelize_fn` plumbed for the AttnRes overlay
       (already exists at `experiments/kimi_linear/parallelize.py`).
    3. The HF-converted ckpt at `phase11_rlhf_grpo_infra/hf_aligned_447m_step12500/`
       (already done, validated).
  ~1-2h.

Once either lands, end-to-end NCCL trace becomes available — the
framework is otherwise complete.

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
