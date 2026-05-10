# Vision injection bug — root cause analysis (in progress)

## Symptoms

| Test | Result |
| --- | --- |
| LM-only greedy on SFT'd 2344 ckpt, prompt "Once upon a time, there was a young man who was very fond of music" (9 words) | ✅ generates real English |
| LM-only greedy, prompt "Hi" (1 word) | ❌ all `!` |
| LM-only greedy, prompt "Once upon a time, there was a young man. He lived in the countryside near a river. He worked hard every day from morning to night, supporting his elderly parents and three young siblings." (34 words) | ❌ all `!` |
| VLM greedy, ANY prompt with image | ❌ all `!` |
| VLM with `dtype=bfloat16` + `disable_cuda_graph=True` | ❌ all `!` |
| VLM with projector output scaled 0.0115× to match text-embed magnitude | ❌ all `!` |

**Output `!` is token id 0 in Llama-3.1 tokenizer**. Greedy argmax over a NaN logit array returns first index = 0 = `!`. So `!!!!!` is **NaN-fallback**, not a learned model preference.

`return_logprob=True` confirms: every position's logprob is `NaN`, top-10 alternatives all `nan`.

## Confirmed NOT root cause

* CUDA graphs (disable still fails)
* Vision feature magnitude (87× larger than text — but training saw same scale)
* Projector NaN/inf (clean output: abs mean=3.91, max=43.25, no NaN/inf)
* SFT undertraining (loss 1.22 at 1 epoch, LM-only sometimes generates real text)

## Suspected root cause

**bf16 numerical instability inside Kimi Linear LM forward** under SGLang inference path. Affects:

* LM-only AND VLM (NOT VLM-specific)
* Intermittent — some prompt lengths/contents succeed, others fail
* Independent of CUDA graph

Likely culprit: KDA's recurrent state accumulation in bf16 over many tokens, OR RMSNorm divisor approaching zero, OR lm_head matmul overflowing bf16 range.

## What this means for VLM post-training

* **SGLang generator** produces NaN logits → all `!` rollouts → no GRPO/PPO signal
* **PolicyTrainer** (training-time logprob recompute via direct
  `model.forward(vision_embeds=, image_mask=)` path) probably DOES work
  because the training-time forward path used same bf16 and converged
  loss 2.22 → 1.22 — but uses different code internals than inference

## Practical overnight options

| Option | Cost | Output |
| --- | --- | --- |
| A. Retry with fp32 dtype | requires SGLang triton kernel fixes (currently fp16/fp32 hit `Mismatched type` errors) | ~6h debug; might fix |
| B. Switch backbone to Qwen3-0.6B + SigLIP + projector, train 1-2h on LLaVA-Pretrain, do VLM GRPO on it | ~8h overnight | working VLM RL on a different (well-pretrained) backbone |
| C. Continue debug Kimi Linear bf16 instability (instrument forward, find NaN-source layer) | open-ended | might unblock our research model |
| D. Use trainer-side compute_token_log_probs only — accept generator producing `!` rollouts as KNOWN broken; demonstrate trainer-step on captured episodes | ~1h | partial demo, no real RL |

Option B is the most pragmatic overnight unblock. Loses Kimi AttnRes
research alignment but produces a working VLM RL artifact.
Option C is the proper fix but open-ended.
