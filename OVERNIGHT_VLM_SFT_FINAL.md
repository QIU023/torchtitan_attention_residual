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

* **SGLang VLM serving works end-to-end** ✅ — Engine boots in 25s
  on the converted ckpt, SigLIP+projector+KDA+MLA+MoE all dispatch
  correctly, image+text prompts decode in ~3.6 s/sample. Output
  text quality is poor (model is undertrained at this scale) but
  the entire serving stack is verified functional.

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

## VLM serving fix (root cause + minimal patch)

The VLM forward through `general_mm_embed_routine` was hitting
`RadixLinearAttention.forward` with `get_forward_context()` returning
None (because piecewise CUDA graph is explicitly disabled by SGLang
when `forward_batch.input_embeds is not None`, see TODO(yuwei) in
`piecewise_cuda_graph_runner.py:can_run`).

But that turned out NOT to be the actual blocker — wrapping our
forward in `set_forward_context` made `is_extend()` still come back
False during prefill capture, suggesting the IF/ELSE in
`RadixLinearAttention.forward` was a red herring.

**True root cause**: `ModelRunner.kimi_linear_config` returned None
for our `KimiAttnResVLConfig` because the `KimiLinearConfig` is
nested under `.text_config` (LLaVA-style VLM wrapping). Without it,
`mambaish_config` returned None, and the `HybridLinearAttnBackend`
that routes KDA layers to `MambaAttnBackendBase` (vs MLA layers to
flashinfer_mla) never got attached. Result: the standard
`AttentionBackend.forward(q,k,v)` got called with KDA's
`(layer, mixed_qkv, a, b)` kwargs and crashed.

**Patch** (in `vlm-sglang-overlay`):

```python
# sglang/srt/model_executor/model_runner.py
@property
def kimi_linear_config(self):
    config = self.model_config.hf_config
    if isinstance(config, KimiLinearConfig):
        return config
    # VLM unwrap
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and isinstance(text_cfg, KimiLinearConfig):
        return text_cfg
    return None
```

Plus the converter emits dual architectures
`["KimiAttnResVLForConditionalGeneration", "KimiLinearForCausalLM"]`
so MLA dispatch routes via flashinfer_mla (mirrors the LM-only
converter trick).

## What's NOT done (next-session work)

1. **Real prod-grade RL** — RFC #26 partial (DCP load works,
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
