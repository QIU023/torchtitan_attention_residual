# Phase 2 Runbook — Block AttnRes Loss-Curve Alignment

Goal: on a single GPU, train a baseline Llama3-175M and a Block AttnRes
Llama3-175M under identical hyperparameters, then compare their loss
curves. The expected outcome is that the AttnRes curve sits visibly below
the baseline curve, with the gap widening over training. This is the
proof-of-correctness needed before opening the torchtitan RFC PR.

Everything below is runnable on a fresh Linux box with conda installed.
Start to finish: ~12 hours (mostly the training runs). Budget: ~$5-10 on
Vast.ai for an RTX 5090.

> **This folder is NOT part of the torchtitan PR.** It's the local workflow
> for reproducing the results that go INTO the PR description. The actual
> committable code lives in `../torchtitan/`.

---

## 0. Workspace layout

The scripts in this folder assume `phase2/` and `torchtitan/` are siblings:

```
<workspace>/                        # e.g. ~/work/ on the rental box
├── phase2/                         # this folder: scripts + runbook + run outputs
│   ├── setup_env.sh
│   ├── launch.sh
│   ├── compare_losses.py
│   ├── smoke_test_attn_res.py
│   ├── README.md                   # this file
│   └── runs/                       # created at launch time
│       ├── baseline/
│       └── attn_res/
└── torchtitan/                     # cloned fork on feat/block-attn-res
    └── ...
```

If you already have `torchtitan/` somewhere else, export
`TORCHTITAN_DIR=/path/to/torchtitan` before running the scripts.

---

## 1. Prerequisites

Hardware:
- 1x CUDA GPU with >= 24 GB VRAM (tested target: RTX 5090 / 4090, A100).
- 50 GB free disk: ~15 GB env + ~1 GB tokenizer + ~10 GB checkpoints.

Software:
- conda (miniconda or anaconda) on the `PATH`.
- `git` with access to the torchtitan fork.
- Hugging Face access (two options):
  - **Ungated mirror (default, no login needed)**: the setup script pulls the
    tokenizer from `NousResearch/Meta-Llama-3.1-8B`, a public mirror with a
    byte-identical tokenizer. Zero setup.
  - **Official gated repo**: set `HF_REPO=meta-llama/Llama-3.1-8B` and run
    `huggingface-cli login` first. Requires the Llama-3.1 license accepted
    on the HF web UI.

---

## 2. Clone both sides

```bash
# From the workspace root you want to use:
mkdir -p ~/work && cd ~/work

# 1. The torchtitan fork, on the feature branch
git clone -b feat/block-attn-res https://github.com/QIU023/torchtitan.git

# 2. This phase2/ folder (copy from your laptop, or keep in a gist and clone)
# If you sync it via scp:
#   scp -r ~/AttnResidualTorchTitan/phase2 rental-box:~/work/
```

Verify:

```bash
cd ~/work/torchtitan && git log --oneline -1
cd ~/work/phase2 && ls
```

---

## 3. Environment setup

From the workspace root (`~/work/` in the example):

```bash
bash phase2/setup_env.sh
```

This:

1. Creates conda env `attnres` (python 3.11).
2. Installs torch nightly + torchtitan editable + dev deps.
3. Downloads the Llama-3.1 tokenizer to `torchtitan/assets/hf/Llama-3.1-8B/`.
4. Runs the standalone smoke test (`phase2/smoke_test_attn_res.py`).
5. Runs the torchtitan unit tests (`torchtitan/tests/unit_tests/test_attn_res.py`).

Expected final output:

```
[PASS] core primitive: identity / uniform / gradient flow
[PASS] multi-layer flow: 3 committed blocks, final partial (2, 4, 16)
[PASS] backward reaches token embedding through the AttnRes chain
[PASS] zero-init: finite output / loss / gradients on all layer params
All smoke tests passed.
...
PASSED tests/unit_tests/test_attn_res.py::TestBlockAttnResFunction::... (several)
PASSED tests/unit_tests/test_attn_res.py::TestDecoderWithAttnRes::...
...
[setup_env] DONE. Activate with: conda activate attnres
```

If any test fails, **STOP and fix it before continuing**. Training with a
broken primitive wastes GPU hours.

Activate the env for subsequent commands:

```bash
conda activate attnres
```

---

## 4. Dry-run config validation (no GPU compute needed)

torchtitan ships a "fake backend" mode that builds the model, runs one
forward/backward pass on fake comms, and exits. Use it to verify that our
configs load cleanly before burning GPU hours.

```bash
cd ~/work/torchtitan

COMM_MODE=fake_backend bash run_train.sh \
    --module attn_res --config llama3_175m_baseline

COMM_MODE=fake_backend bash run_train.sh \
    --module attn_res --config llama3_175m_attn_res
```

Both should print model-parameter summaries and exit with status 0. Look
for lines like `Model params: 1.82e+08` confirming the ~175M target, and
for the AttnRes run, log lines mentioning `attn_res_proj` parameters.

---

## 5. Optional: quick smoke run on the debug model (30s)

Before the 10-hour main runs, do a 100-step sanity pass on the tiny
debug model. Catches env issues (CUDA OOM on a weird config, dataloader
crashes, etc.) in under a minute.

```bash
cd ~/work/torchtitan

# Baseline debug
bash run_train.sh --module llama3 --config llama3_debugmodel \
    --training.steps 100 --dump_folder ~/work/phase2/runs/smoke_baseline

# AttnRes debug
bash run_train.sh --module attn_res --config debugmodel_attn_res \
    --training.steps 100 --dump_folder ~/work/phase2/runs/smoke_attn_res
```

Both runs should:
1. Reach step 100 without NaN / OOM.
2. Show a loss that decreases from ~8 to ~7 (small drop; only 100 steps).
3. Print `Training completed` at the end.

---

## 6. Launch the main Phase 2 runs

From the workspace root:

```bash
bash phase2/launch.sh
```

Starts a tmux session (`attnres`) with 4 windows:

- `baseline`: 20k steps of `llama3_175m_baseline`
- `attn_res`: 20k steps of `llama3_175m_attn_res` (waits for baseline to finish)
- `monitor`:  `nvidia-smi` in watch mode
- `guardian`: tails `baseline/train.log` for "Training completed", then
  touches the `DONE` flag that unblocks attn_res

The script prints attach/tensorboard/compare commands at the end.

**Detach from tmux**: `Ctrl-b` then `d`.
**Reattach later**: `tmux attach -t attnres`.

### Override training length

The default is 20k steps (~650M tokens at bs=16, seq=2048). For a quick
1-hour smoke instead of an overnight run, use `STEPS`:

```bash
STEPS=2000 bash phase2/launch.sh
```

### Override output directory

```bash
OUT_ROOT=~/experiments/run1 bash phase2/launch.sh
```

---

## 7. Monitor progress

### TensorBoard (recommended)

In a separate terminal on the training box:

```bash
conda activate attnres
tensorboard --logdir ~/work/phase2/runs --port 6006 --bind_all
```

From your laptop, if you're SSH-ed in:

```bash
ssh -L 6006:localhost:6006 user@training-box
# Now open http://localhost:6006
```

Look at the `loss_metrics/loss` scalar. The two runs will appear as
separate TB runs once both have started writing events.

### Tail the log directly

```bash
tail -f ~/work/phase2/runs/baseline/train.log
```

### Check if the guardian flipped yet

```bash
ls -la ~/work/phase2/runs/baseline/DONE 2>/dev/null && echo "baseline done" || echo "baseline still running"
```

---

## 8. Compare loss curves

After both runs finish:

```bash
python ~/work/phase2/compare_losses.py \
    --baseline ~/work/phase2/runs/baseline/tb \
    --attn_res ~/work/phase2/runs/attn_res/tb \
    --out ~/work/phase2/runs/comparison.png
```

Expected stdout:

```
[compare_losses] wrote ~/work/phase2/runs/comparison.png

=== Summary ===
  baseline :
    train final: 4.XX
    val   final: 4.XX
  attn_res :
    train final: 4.YY (lower than baseline)
    val   final: 4.YY

  train loss delta (attn_res - baseline): -0.0XXX (AttnRes better)
```

**Success criterion**: the train loss delta is negative (AttnRes < baseline)
and visible in the plot. At 650M tokens, expect a delta in the ballpark of
-0.01 to -0.03 train loss. If the delta is zero, positive, or NaN, there's
a bug to chase.

---

## 9. What to do with the results

Copy the comparison plot + final numbers back to your laptop:

```bash
scp training-box:~/work/phase2/runs/comparison.png .
scp training-box:~/work/phase2/runs/baseline/train.log ./baseline.log
scp training-box:~/work/phase2/runs/attn_res/train.log ./attn_res.log
```

These three artifacts go into the RFC PR description:
1. `comparison.png` — the money shot
2. `baseline.log` tail — the final val loss
3. `attn_res.log` tail — the final val loss + param count confirming AttnRes
   module adds negligible overhead

---

## 10. Expected resource usage

| Item | Baseline run | AttnRes run | Total |
|---|---|---|---|
| Steps | 20,000 | 20,000 | 40,000 |
| Tokens processed | ~650M | ~650M | ~1.3B |
| Wall clock (RTX 5090) | ~4 h | ~4.2 h | ~8.2 h |
| Peak VRAM (bs=16, seq=2048, bf16) | ~12 GB | ~13 GB | — |
| Cost @ $0.362/hr | ~$1.45 | ~$1.52 | ~$3 |

If you want a stronger signal, extend to 50k steps (~1.6B tokens per run)
by passing `STEPS=50000`. That roughly triples the run time and cost.

---

## 11. Troubleshooting

### "CUDA out of memory" on startup

Lower the batch size:

```bash
cd ~/work/torchtitan
bash run_train.sh --module attn_res --config llama3_175m_baseline \
    --training.local_batch_size 8
```

### Loss is NaN at step 1

Something's broken in the init. Re-run `phase2/setup_env.sh` — it will
re-run the unit tests, specifically `test_pseudo_queries_are_zero_after_init`.
A non-zero pseudo-query at step 0 will cause the first-step softmax to be
non-uniform and can produce large initial losses or NaN under bf16.

### `Training completed` never appears

Either the run crashed (check the log) or torchtitan renamed its final
log line. Adjust the grep in `phase2/launch.sh` (guardian window) to match.

### AttnRes run never starts

The guardian window is waiting on the baseline `DONE` flag. Check:

```bash
grep 'Training completed' ~/work/phase2/runs/baseline/train.log
```

If the line is there but `DONE` isn't, touch it manually:

```bash
touch ~/work/phase2/runs/baseline/DONE
```

### Tokenizer download failed (403)

You're hitting the gated `meta-llama/Llama-3.1-8B` without a license.
Either run `huggingface-cli login` with a token that has access, or use
the ungated mirror (the default):

```bash
HF_REPO=NousResearch/Meta-Llama-3.1-8B bash phase2/setup_env.sh
```

---

## Appendix: file inventory

**In this folder** (not committed to torchtitan):

| File | Role |
| --- | --- |
| `phase2/setup_env.sh` | Create conda env, install torch + torchtitan, download tokenizer, run tests |
| `phase2/launch.sh` | Start tmux session with sequential baseline + attn_res runs |
| `phase2/compare_losses.py` | Read TensorBoard events, plot loss curves, print final delta |
| `phase2/smoke_test_attn_res.py` | Standalone torch-only smoke test (runs without torchtitan install) |
| `phase2/README.md` | This runbook |

**In `../torchtitan/`** (committed to the PR):

| File | Role |
| --- | --- |
| `torchtitan/experiments/attn_res/attn_res.py` | `block_attn_res()` primitive, `AttnResProjection`, stack/unstack helpers |
| `torchtitan/experiments/attn_res/model.py` | `AttnResLlama3TransformerBlock`, `AttnResLlama3Model` subclasses (core `decoder.py`/`model.py` untouched) |
| `torchtitan/experiments/attn_res/__init__.py` | Model flavors (`debugmodel_attn_res`, `175M_attn_res`) + `model_registry` |
| `torchtitan/experiments/attn_res/config_registry.py` | Trainer configs: `llama3_175m_baseline`, `llama3_175m_attn_res` |
| `torchtitan/experiments/attn_res/tests/test_attn_res.py` | Unit tests for all AttnRes components |
| `torchtitan/experiments/__init__.py` | Registers `attn_res` in `_supported_experiments` |
