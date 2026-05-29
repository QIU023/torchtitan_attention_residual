# D-7 OPD Eval Cascade (Mantis SigLIP teacher, caption task, 300 steps)

Baseline (SFT step-5200): **12.3%** (37/300)
Teacher LLaVA-NeXT-8B upper:  63.7%
D-4 (CLIP teacher 50 step): 9.3%
D-6 (Mantis 50 step):       10.67%

| Ckpt | GQA acc | Notes |
|---|---|---|
| step-50 | 0.1133 | eval_step50.log |
| step-100 | 0.1200 | eval_step100.log |
| step-150 | 0.0767 | eval_step150.log |
| step-200 | 0.0700 | eval_step200.log |
| step-250 | 0.0400 | eval_step250.log |
| step-300 | 0.0367 | eval_step300.log |
