# Kimi Linear 436M — pretraining closure criteria + distillation transition

Companion to `multi_modal_idea.md`. That doc answered "can the
12,500-step ckpt be the backbone for from-scratch multimodal
pretraining?" (no — too undertrained). This doc answers the earlier
question: **what level of pretraining is enough to close the core
project narrative** (architecture port → pretraining behavior
validation → PP system validation), and how to bridge from that
closure point to a multimodal-ready backbone **without** another
100 GPU-days of LM pretraining — via knowledge distillation from
Kimi-Linear-48B-A3B.

---

## Part 1 — Pretraining closure criteria

### The three narrative planks and what each one needs

| Plank | Minimum bar to claim the plank | What "enough pretraining" means here |
|---|---|---|
| 1. Architecture port | KDA + MLA + MoE + AttnRes forwards cleanly, matches paper component shapes, FSDP + PP stable over 10K+ steps | **Already met** at step 4,500 — no NaNs, grad_norm healthy (~0.05), loss trajectory smooth |
| 2. Pretraining behavior validation | AttnRes Δloss vs baseline is stable, sign-consistent, and outside bf16+NCCL noise on a credible token budget | Needs loss curves where AttnRes's advantage over baseline (currently Δ≈0.06 at step 4,500) is **clearly outside the run-to-run seed noise band** |
| 3. PP system validation | Naive-PP vs adapter-PP loss equivalence on the full Kimi stack (KDA+MLA+MoE, not just Llama3) | Can be validated with a shorter run (10-20K steps) — the *adapter itself* was already validated on Llama3 L=16 N=8 in Phase 3 |

Plank 2 is the rate-limiting one. The question becomes: **how many
steps before AttnRes Δ is publishable?**

### Token budget ↔ step count

At current config (GLOBAL_BS=12, SEQ_LEN=2048, so ~24,576 tokens per
step):

| Steps | Training tokens | Multiple of Chinchilla (8.6B) | Multiple of paper 119B |
|---|---|---|---|
| 4,500 (today)  | 0.11B | 0.013× | 0.001× |
| 12,500 (one overnight) | 0.31B | 0.036× | 0.003× |
| **30,000** (target — minimum closure) | **0.74B** | **0.086×** | **0.006×** |
| 60,000 (target — ideal closure) | 1.47B | 0.17× | 0.012× |
| 350,000 (Chinchilla-optimal) | 8.6B | 1.0× | 0.07× |
| 4,850,000 (paper) | 119B | 14× | 1.0× |

### Closure targets

**Minimum closure — 30,000 steps (~0.74B tokens)**
- Rationale: the AttnRes Δloss trajectory needs enough time to stabilize past the warmup/LR-schedule transient. Between step 3K and 5K we already see AttnRes pulling ahead by ~0.06; running to 30K gives roughly 6× more post-transient data for the Δ trend to either persist or collapse.
- Runtime: ~2 overnight runs (each 12,500 steps on 4× RTX 5090 takes ~12h; chained runs hit 30K in ~30h wall-clock).
- Deliverable: baseline vs AttnRes loss curves over 30K steps with a credible Δ measurement + confidence interval from seed variance.

**Ideal closure — 60,000 steps (~1.47B tokens)**
- Rationale: brings us to ~17% of Chinchilla-optimal — still under-pretrained as an LM, but **sufficient to make the AttnRes scaling claim publishable** under the "small-scale validation" framing the AttnRes paper itself uses in its Table 2 sweeps.
- Runtime: ~4-5 overnight runs.
- Deliverable: same as minimum + one extra seed per arm to bound Δ noise.

**What 30K/60K does NOT unlock:**
- Absolute loss is still 3.5-4.0 territory, perplexity ~30-50 — not usable as a real LM for downstream benchmarks.
- Not enough to validate long-horizon training dynamics (MoE expert balancing under 200×+ over-Chinchilla, for example).
- Those are accepted non-goals for this project; they're what the paper's H100-cluster budget buys that ours doesn't.

### Problem B (PP adapter) does NOT need 30K steps

Phase 3 already proved naive-PP ≡ adapter-PP on 175M Llama3 L=16 N=8
over 200K steps. On Kimi Linear the question being validated is
strictly "does the adapter's FQN remapping + delta-cache plumbing
carry over to KDA+MLA+MoE without breaking anything?" — that is a
10-20K step question, not a 60K step question. Problem B's loss
alignment signal saturates quickly; running it out to 30K just to
match Problem A adds no information.

---

## Part 2 — Current state and immediate plan

- **Baseline 436M FSDP run:** completed at step 12,500 (`kimi_436m_baseline_fsdp_overnight` — "Training completed" at 05:29).
- **AttnRes 436M FSDP run:** at step 4,880, loss 4.22, grad_norm 0.054. On track to finish step 12,500 in another ~4h.
- **AttnRes Δ vs baseline at matched steps:** roughly +0.06 in AttnRes's favor (baseline was around 4.28 at step 4,900).
- **Problem B (PP adapter) run:** queued via `run_after_baseline.sh`, but auto-chained off AttnRes FSDP completion — will launch after AttnRes run ends.

### Next overnight after today's pair finishes

Two options; recommendation is (a):

**(a) Continue Problem A to 30K steps (minimum closure).**
Chain a 17,500-step continuation on both baseline and AttnRes arms
resuming from step 12,500 ckpts. Problem B runs in parallel if GPU
free; else sequential. This closes plank 2.

Launcher: `phase4_kimi_attnres_lm_pretrain/experiments/kimi_436m_attnres/launch_continue_30k.sh`.
Important subtlety: the launcher pins `--lr_scheduler.total_steps
12500` on the continuation so the original cosine schedule is
preserved. Without that pin torchtitan rebuilds the LR lambda over
the new 30K step target, which at the resume point (step 12,500)
lands ~8× higher than the ~2e-4 the model ended at — a hot-restart
that spikes loss for ~1K steps. Pinning leaves the continuation
running at the min-LR floor (0.1 × peak = ~2e-4), which is the
standard continued-pretraining recipe.

**(b) Go straight to 60K (ideal closure).**
Higher-confidence Δ, but another ~2 overnights beyond (a). Defer
unless (a) shows the Δ shrinking into noise — in which case we'd
need more data anyway to rule the effect out cleanly.

---

## Part 3 — Distillation transition (bridge to multimodal)

Once plank 2 is closed at 30K/60K steps, the LM backbone is
**project-narrative-complete but multimodal-unfit** (loss 3.5-4.0,
perplexity 30-50). To reach the loss ~2.5 region needed for a
credible multimodal backbone without paying the H100-cluster cost,
use knowledge distillation from a pretrained Kimi Linear teacher.

### Teacher selection

**Kimi-Linear-48B-A3B-Base** (Moonshot's open weights) is the
correct teacher:
- Same KDA + MLA + MoE architecture, so hidden-state distillation
  *could* be added later if useful (not needed for the first pass).
- Strong base-LM quality (trained on 5.7T tokens in Moonshot's run).
- Tokenizer compatibility — if we use the same tokenizer as the
  teacher, logit-level KD is a dense learnable signal at every
  position.
- Architectural fidelity preserves the AttnRes claim: we're
  distilling into a backbone with the AttnRes modification still in
  place, so the ported architecture remains the thing being
  validated.

### KD loss

Standard token-level KD, no fancy intermediate-layer matching for
the first pass:

```
L = α · CE(student_logits, gold_tokens)
  + (1 − α) · T² · KL(softmax(student_logits/T) ‖ softmax(teacher_logits/T))
```

- `α = 0.3` — weight most of the signal on matching the teacher's
  full distribution (KD dense gradient), keep a small CE anchor so
  the student doesn't drift from the data distribution.
- `T = 2` to `T = 4` — temperature smooths teacher logits; 2-4 is
  the standard range, T=2 usually works; only try T=4 if student
  overfits to teacher mode.
- Teacher runs in eval mode, bf16, no grad. No teacher forcing
  beyond shared input sequences.

### Memory plan — 48B teacher on 4× RTX 5090

- **Teacher memory (bf16 weights, MoE-aware activation):** ~96 GB
  total params in bf16. Does NOT fit in one 5090's 32 GB, but fits
  across 4× 5090 = 128 GB aggregate (tight).
- **FSDP2 for the teacher in eval mode:** shards weights across the
  4 ranks → ~24 GB/rank just for weights. Plus student weights
  (~4 GB/rank sharded at 436M), plus activations, plus KV cache in
  the teacher forward.
- **CPU offload of inactive MoE experts:** Moonshot's 48B-A3B
  activates only A3B params per token (top-8 of 256 experts). The
  other ~15/16 of experts are idle per microbatch and can live on
  CPU. torchtitan supports `--training.enable_cpu_offload`;
  extending that selectively to MoE experts that aren't routed to
  buys back the memory headroom.
- **Fallback if memory still tight:** run teacher in int8 (vLLM-style
  post-training quantization) — Kimi Linear-48B-A3B degradation under
  int8 for logit distillation is negligible vs full-bf16 teacher, per
  standard KD-quantization results.

### Token budget

KD is **signal-dense** — each position gives a 128K-way probability
target (full vocab), vs CE's 1-hot target. Empirically, 1 KD-token
≈ 5-10 CE-tokens for distillation convergence on small students.

| KD tokens | Equivalent CE tokens | Wall-clock on 4× 5090 |
|---|---|---|
| 1B   | 5-10B  | ~1-2 days |
| 3B   | 15-30B | ~3-5 days |

3B KD tokens is the target — roughly Chinchilla-optimal-equivalent
(8.6B CE tokens) via the dense signal.

### Expected loss after KD

- CE component: 2.5-3.0 (multimodal-ready backbone zone).
- KD component: teacher-student KL < 1.0 (student closely tracks
  teacher distribution on routine text).
- Validation: running a held-out LM-eval-harness subset (Hellaswag,
  PIQA, ARC-E) should show the student reaching 30-40% of teacher's
  absolute score — weak but categorically different from the
  pre-KD "near-random" regime.

### Does KD kill the project narrative?

Concern: "if we distill, doesn't that make the pretraining run
meaningless?" No, because the project planks are:

1. Architecture port — **validated by the port existing and
   forwarding cleanly**, independent of how much it's trained.
2. Pretraining behavior validation — **validated by the 30K/60K
   baseline vs AttnRes Δ curve**, independent of whether the final
   model is used post-distillation.
3. PP system validation — **validated by Problem B's loss
   equivalence**, independent of absolute loss level.

KD is the mechanism that takes the validated-but-undertrained 30K
ckpt and makes it a useful *artifact* for downstream multimodal
work. The closure of planks 1-3 happens BEFORE KD and stands on its
own. KD is explicitly labeled "post-closure transition" in the
project writeup — not claimed as part of the pretraining validation.

---

## Part 4 — Full timeline (4× RTX 5090 budget)

```
Today (2026-04-23)
├── AttnRes FSDP overnight finishing (ETA step 12500: ~12:30 today)
├── Problem B PP-adapter run auto-chains after AttnRes FSDP ends
│
Overnight N+1 / N+2  (≈2 more nights)
├── Continue Problem A (baseline + AttnRes) to step 30,000
│   ├── Deliverable: minimum-closure Δ curves
│   └── This closes planks 2 & 3
│
Overnight N+3 / N+4  (OPTIONAL — if Δ noisy)
├── Push Problem A to step 60,000
│   └── Ideal-closure Δ curves
│
Post-closure — Distillation phase (1-3 days)
├── Download Kimi-Linear-48B-A3B-Base weights
├── Wire up KD data pipeline (student fwd, teacher fwd, KD loss)
├── 1-3B KD tokens → loss 2.5-3.0
│
Multimodal phase (Phase 5 from multi_modal_idea.md)
├── 5a: projector-only caption pretraining (2-4h)
├── 5b: LLaVA-style instruction SFT overnight
└── 5c (optional): PP=4 stress test with multimodal data
```

Total before the multimodal handoff: **~1 week of overnights +
~2 days of KD + ~1 overnight of multimodal**. That is the full
end-to-end pipeline the fork demonstrates.

---

## TL;DR

- **Closure = 30K steps minimum, 60K ideal.** Not paper-level
  training, but enough for the AttnRes Δ and PP adapter claims on
  Kimi Linear. 2-5 more overnights.
- **Distillation bridges the gap between "architecture validated"
  and "multimodal-ready"** without paying the 100+ GPU-day LM
  pretraining cost. Teacher = Kimi-Linear-48B-A3B-Base; KD loss is
  standard `α·CE + (1-α)·T²·KL`; 1-3B KD tokens → loss 2.5-3.0.
- **KD does not dilute the narrative** — planks 1-3 close before KD
  starts and stand alone; KD is explicitly a post-closure transition
  to make the ckpt useful for multimodal, not part of the
  pretraining validation itself.
