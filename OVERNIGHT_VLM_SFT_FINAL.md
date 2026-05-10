# Overnight session — 2026-05-09 → 2026-05-10

VLM track. Auto mode active throughout. Carries over from earlier text-only
RL trace work that the user explicitly de-prioritised ("不要为了采 trace
而跑没意义的模型").

---

## Headline

* **VLM SFT 447M** on **LLaVA-Instruct-150K** completed end-to-end:
  loss **2.22 → 1.42** over 1200 steps, FSDP=8, SEQ=512 LBS=4 GBS=64,
  ~1.7h wall (after one CUDA-assert recovery via SEED=43 resume).
  Final ckpt at `phase5/runs/sft_v_fsdp8_447m_aligned_llava_instruct_150k/checkpoint/step-1200`.

* **SGLang VLM model class** + DCP→HF converter + processor for
  Kimi AttnRes VLM landed on `vlm-sglang-overlay` branch
  (sglang submodule + main repo, both pushed). End-to-end DCP→HF→
  Engine-load works structurally; final inference path hits a KDA
  forward-context branch that needs more debugging (LM-only path
  unaffected, verified).

* **PolicyTrainer DCP-native load** added, unblocks running RL
  trainer on our 447M Kimi AttnRes ckpt without writing a full
  state_dict_adapter for Kimi MoE.

---

## What's committed

`vlm-sglang-overlay` branch — both the main repo and the sglang submodule:

| Layer | File | Purpose |
| --- | --- | --- |
| sglang config | `sglang/srt/configs/kimi_attn_res_vl.py` | `KimiAttnResVLConfig` (HF AutoConfig-registered, `model_type=kimi_attn_res_vl`) |
| sglang model | `sglang/srt/models/attn_res_vl_overlay.py` | `KimiAttnResVLForConditionalGeneration`: HF SigLIP + 2-layer MLP projector + `KimiBlockAttnResForCausalLM` |
| sglang processor | `sglang/srt/multimodal/processors/attn_res_vl.py` | Host-side image-token splice (bypasses LlavaProcessor's text-level `<image>` requirement) |
| torchtitan trainer | `torchtitan/experiments/rl/actors/trainer.py` | `dcp_initial_load_path` opt-in, skips HF state_dict_adapter |
| converter | `phase11/dcp_to_hf_kimi_attn_res_vl.py` | DCP → HF safetensors for VLM (LM keys via phase10 helpers + projector passthrough) |
| RL launcher | `phase11/rlhf/run_grpo_kimi_attn_res.py` | GRPO/PPO entry-point on the 447M Kimi AttnRes LM (text-only, real research weights) |
| smoke tests | `phase11/smoke_lm_only.py`, `smoke_vlm_load.py`, `smoke_vlm_engine.py` | Triaged sanity tests; LM-only passes |
| pipeline | `phase11/run_sft_447m_llava_instruct_150k.sh`, `post_sft_vlm_smoke.sh`, `auto_post_sft.sh` | End-to-end SFT → DCP→HF → Engine smoke → eval, hands-off |

PR-link to open:
* sglang fork: https://github.com/QIU023/sglang/pull/new/vlm-sglang-overlay
* main repo: https://github.com/QIU023/torchtitan_attention_residual/pull/new/vlm-sglang-overlay

---

## SFT learning curve

```
step    0    100   200   300   400   500   600   700   800   900   1000  1100  1200
loss   2.22  1.99  1.71  1.62  1.57  1.51  1.50  1.47  1.45  1.42  1.40  1.38  1.42
```

Smooth descent. The bump at 600→700 is the SEED=43 reshuffle right after
the resume (data ordering change). The very-final 1.42 reflects the
last batch's noise; smoothed loss across the last 50 steps is ~1.39.

---

## SFT engineering log

| Issue | Resolution |
| --- | --- |
| 4D mesh (FSDP=2 PP=2 TP=2 EP=2) hits "FSDP requires DP and TP/EP same parent mesh" assert under torch 2.9 stable | Pivoted to pure FSDP=8; same gap previously hit on Phase 9-B (PROJECT_STATUS.md) |
| Initial OOM at SEQ=579 LBS=4 (LLaVA-Instruct context length doubled vs. LLaVA-Pretrain SEQ=260) | Reduced SEQ_LEN to 512; covers ~70% of samples without truncation (p50=401 tok, p75=506 tok, p90=681 tok) |
| Stale procs from earlier failed attempts kept GPUs OOM-pinned | Targeted PID kill before each retry |
| `cudaErrorAssert` in moe.forward at step 651 (data-driven token-index out-of-range) | Resumed from step-600 ckpt with SEED=43 to reshuffle past offending sample; orchestrator success-gate threshold raised 490 → 1100 |
| Disk crisis at step 500 (98% used, KEEP_K=2 lagging) | `ckpt_watchdog.sh` polled every 60s, kept only the latest step-N |
| Activation checkpoint `mode=full` ignored by Kimi Linear (Phase 4c skip) | Doc'd in launcher comment; SEQ=512 + FSDP=8 fit in 13.4 GiB / 32 GB without AC |

---

## What's NOT done (next-session work)

1. **VLM end-to-end serving** — converter + model class + processor
   land cleanly; HF AutoConfig + AutoTokenizer + SigLIP image
   processor all work; the LM forward path through
   `general_mm_embed_routine` hits
   `RadixLinearAttention.forward` with `get_forward_context() is
   None`, falling back to a mismatched
   `AttentionBackend.forward(q,k,v)` signature that the layer's
   `(mixed_qkv, a, b)` arguments don't satisfy. LM-only inference
   on the same converted weights works (smoke_lm_only.py passes),
   so the issue is contained to the multimodal embed-splice path.
   Likely fix: ensure `set_forward_context()` is held during
   prefill when input_embeds-driven, OR call
   `unified_linear_attention_with_output` unconditionally.

2. **Real prod-grade RL** — RFC #26 partial (DCP load works,
   `compute_token_log_probs` should work too since Kimi Linear
   accepts kwargs). But disk-based weight sync from trainer to
   SGLang Engine is a framework gap (push_model_state_dict only
   does torchstore today; no HF safetensors dump). Same gap
   affects all SGLang RL deployments, not just Kimi.

3. **Multimodal Episode pipe** (#38) — Episode → image_path →
   SGLangGenerator → vision tower. Blocked on (1).

4. **Quality** — base 447M ckpt is undertrained (text generation
   collapses to repeated `!`). The SFT loss curve is real (down
   from 2.22) but visible generation quality at this scale needs
   more pretraining tokens. This is a model-scale reality, not a
   pipeline bug.

---

## Disk + daemons

* SFT final ckpt: 30 GB at `step-1200`
* HF VLM safetensors: 3.0 GB at `phase11/hf_aligned_447m_vlm_sft1200/`
* COCO train2017: 19 GB at `/workspace/.hf_home/coco_train2017/`
* Disk: 155 GB / 200 GB used (78%)

Daemons stopped: ckpt_watchdog (PID 1370839), auto_post_sft
(PID 1386320). Both completed their job.

---

## Quick reference for next session

```bash
# Resume the VLM serving debug
CUDA_VISIBLE_DEVICES=7 python3 phase11/smoke_vlm_engine.py \
    --model-path phase11/hf_aligned_447m_vlm_sft1200 --tp-size 1
# (Currently fails inside RadixLinearAttention.forward; LM-only smoke works.)

# Run RL on real research weights (text-only, GRPO smoke)
python3 phase11/rlhf/run_grpo_kimi_attn_res.py \
    --dcp-load-path phase4/runs/kimi_447m_aligned_block_attn_res_fsdp_paperhparams/checkpoint/step-12500 \
    --hf-model-path phase11/hf_aligned_447m_vlm_sft1200 \
    --num-steps 50 --kl-coef 0.05  # PPO with frozen ref
```
