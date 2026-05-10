# Overnight VLM GRPO run — 2026-05-10

## Setup
- Model: 447M Kimi Block AttnRes (DCP `phase5/runs/vlm_447m_sft_instruct/checkpoint/step-2344`)
- Generator: SGLang Engine TP=4 with VLM ckpt `phase11/hf/vlm_sft_1ep`
- Trainer: torchtitan PolicyTrainer FSDP=4
- Task: LlavaCaptionTask (LLaVA-Pretrain 50K records, BLEU-1 + length + format reward)
- Steps: 1200, episodes/step = 4, kl_coef=0
- Env: `ATTNRES_MLA_FP32_FALLBACK=1 SGLANG_DISABLE_SHM_MM=1`

## Headline numbers
- Total steps: 1200/1200 (completed cleanly)
- Wall time: 260.5 min (~4h21m)
- Avg step time: 13.0s (early ~12.5s, late ~20s)
- Steps with positive group-mean reward: 78/1200 (6.5%)

## Reward / loss trajectory (6 evenly-spaced chunks)
```
chunk:    1       2       3       4       5       6
reward:  -0.337  -0.306  -0.276  -0.326  -0.311  -0.316
loss:    -17.5   -45.5   -88.9   -158.3  -241.0  -276.4
```

Reward flat, loss magnitude grows monotonically — vanilla policy gradient is
diverging without clipping or KL regularization (kl_coef=0 here).

## Verdict

* **Infrastructure works.** Trainer + SGLang VLM generator + Grader + Episode
  flow + advantage computation + push_state_dict round-trip all functional on
  real research weights for a full 1200 steps. This was the overnight goal
  ("保证torchtitan GRPO work") and it's met.

* **Policy doesn't improve.** Three reasons:
  1. **Undertrained VLM** — 1 epoch on LLaVA-Instruct-150K (~75M tokens) is far
     short of what a 447M model needs to learn vision-language alignment;
     greedy decode collapses to the `!`-pad EOS trap on most prompts.
  2. **No grad clipping / KL** — loss magnitude blows up (peak ~−684) without
     trust-region constraints; updates wander far from the SFT distribution.
  3. **Sparse reward signal** — BLEU-1 vs gold caption needs the model to
     actually see the image; with no image grounding the reward is mostly
     length-filter / format-bonus noise.

## Path forward (post-overnight)

1. **Continued VLM SFT** before next GRPO attempt — image-text alignment is
   the bottleneck, not the RL trainer. Target ≥3 epochs on LLaVA-Pretrain
   captions or longer LLaVA-Instruct schedule.
2. **Add grad clipping** (`grad_norm_clip=1.0`) and **KL regularization**
   (`--kl-coef 0.05`) to PolicyTrainer to prevent loss blow-up.
3. **Reward shaping** — penalise the `!!!!` pad pattern explicitly so the
   policy can move away from it without needing image-grounded BLEU first.
