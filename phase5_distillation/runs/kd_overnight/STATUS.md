# KD Overnight Run — Status

## Run config

- **Started:** 2026-04-25 10:38 UTC
- **Steps target:** 10,000 (~17h at 6.3 s/step; 10h = ~5,700 steps)
- **Student:** kimi_linear_436m_block_attn_res_n4 from
  phase4/runs/kimi_436m_block_attn_res_fsdp_overnight/checkpoint/step-12500
- **Teacher:** NousResearch/Meta-Llama-3.1-8B (local /root/hf_cache/Llama-3.1-8B)
- **Hardware:** 4× RTX 5090, FSDP2 student + FSDP2 teacher
- **Batch:** LOCAL_BS=1 GLOBAL_BS=8 SEQ_LEN=2048 grad_accum=2
- **LR:** 2e-4 constant (warmup 100 steps, no decay)
- **KD:** α=0.3 (CE weight), T=2.0
- **Memory:** 24.54 GiB / 31 GiB per rank (78%)
- **Throughput:** 647 tps/rank ≈ 6.3 s/step
- **Ckpt:** every 500 steps, keep latest 3

## Progress trajectory

```
step    1   loss 3.00062
step  100   loss ~2.5  (rough; warmup-decay band)
step  200   loss ~2.30
step  300   loss ~2.20
step  400   loss ~2.30
step  410   loss 2.18  (44 min in)
```

Loss dropped 0.82 in first 410 steps. KD signal (dense softmax target)
is much faster than CE-only training would be.

## Files to check

- `rank0_stdout.log` — symlink to torchelastic per-rank stdout. Has
  full step:N loss:X grad_norm:Y memory:Z tps:T mfu:M lines.
- `train.log` — torchrun launcher's tee output (limited; init lines
  only, real metrics go through Python logger to torchelastic stdout).
- `tb/` — TensorBoard event files.
- `checkpoint/step-N` — student DCP ckpts every 500 steps.

## Quick check commands

```bash
# Latest progress
tail -5 phase5_distillation/runs/kd_overnight/rank0_stdout.log

# Loss curve (every 100 steps)
grep "step:.*loss" phase5_distillation/runs/kd_overnight/rank0_stdout.log \
    | awk 'NR%10==1'

# Current GPU
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

## Goal

Eval the final ckpt against c4_validation using
`phase4/experiments/kimi_pp_adapter/eval_val.sh` (already wired). Target
delta vs teacher's val_loss ≤ 0.05. Pre-KD student val_loss was 3.7326;
teacher (Llama-3.1-8B) val_loss on c4-en typically ~2.4-2.6.

## Known caveats

- Loss reported here is KD-interpolated `α·CE + (1-α)·T²·KL`, not pure
  CE. Compare directly to teacher's loss only after running
  `phase4/.../eval_val.sh` on the resulting student ckpt.
- Student vocab is 163,840 (Kimi-Linear config) but training data only
  uses Llama tokens [0, 128256). Upper 35K embedding rows are still
  random untrained (KD doesn't touch them).
- FSDPAttnResProjection warning at startup is benign — that module
  participates in AttnRes-N=4 paths but only fires backward after the
  forward conditional, FSDP probe doesn't hit it on first forward.
