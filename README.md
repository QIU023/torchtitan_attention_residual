# Block Attention Residuals (Kimi, 2026) — torchtitan integration workspace

Project-level workspace for integrating **Block Attention Residuals** into
[pytorch/torchtitan](https://github.com/pytorch/torchtitan), covering the
full arc from single-GPU correctness proof to 8-GPU PP benchmark.

> **This repository is the logbook / playbook / RFC draft.** The actual
> code that lands in torchtitan lives in the fork branch
> [`QIU023/torchtitan:attention_residual_dev`](https://github.com/QIU023/torchtitan/tree/attention_residual_dev)
> under `torchtitan/experiments/attn_res/`.

## Algorithm

Kimi Team's Block AttnRes ([arXiv 2603.15031](https://arxiv.org/abs/2603.15031))
replaces fixed residual accumulation with softmax attention over block
outputs, using a zero-initialized per-layer pseudo-query. Block AttnRes
partitions layers into `N` blocks, uses standard residuals inside a block,
and attention only at block boundaries — so cross-stage traffic is `O(N d)`
instead of `O(L d)`.

Paper: AttnRes ≈ baseline × 1.25 compute at matched size, PP-compatible at
`N ≈ 8`.

## Repository layout

| Path | What it is |
| --- | --- |
| [`ROOT_PLAN.md`](./ROOT_PLAN.md) | Full phased project plan (hardware, budget, risk register, references to Kimi infra notes) |
| [`RFC_DRAFT_v2.md`](./RFC_DRAFT_v2.md) | RFC to post as a GitHub issue on pytorch/torchtitan. Covers motivation, two-PR scope, Phase 2 evidence (loss-delta table), Phase 3/4 plan, open design questions. |
| [`phase2/`](./phase2/) | Single-GPU FSDP reproduction playbook. `setup_env.sh` + `launch.sh` + `compare_losses.py` + results (`runs/comparison.png`). |
| [`phase3/`](./phase3/) | Pipeline-parallel playbook for the 8-GPU run. Adapter design notes, fake-PG smoke, 8-GPU launch scripts, PP-vs-single numeric compare. |
| [`reports/`](./reports/) | Interim written reports (en + zh) for portfolio / interview walkthroughs. |
| [`Attention-Residuals/`](./Attention-Residuals/) | Kimi's reference implementation + paper PDF (unchanged vendor copy). |

## Phase 2 one-line result

Single-GPU Llama3 150M dense on FSDP, 20 k steps on C4-en, identical config
except `model_spec`:

| step | baseline | attn_res | Δ |
|---:|---:|---:|---:|
| 500  | 6.1412 | 6.0146 | −0.1265 |
| 5000 | 4.3575 | 4.2696 | −0.0879 |
| 10000 | 4.3235 | 4.2192 | −0.1043 |
| 15000 | 3.7368 | 3.6861 | −0.0507 |

AttnRes loss is consistently below baseline across every logged milestone.
See [`phase2/runs/comparison.png`](./phase2/runs/comparison.png).

## Phase 3/4 status

Draft code for the cross-stage caching adapter is ready
([`phase3/adapter_design.md`](./phase3/adapter_design.md) + the adapter
itself lives in the fork at
`torchtitan/experiments/attn_res/pipeline_adapter.py`). 8-GPU RTX 5090 PCIe
validation is the next session's work.

## Not in this repo

- The torchtitan fork itself (separate repo:
  [`QIU023/torchtitan@attention_residual_dev`](https://github.com/QIU023/torchtitan/tree/attention_residual_dev)).
  Cloned alongside this workspace as a peer directory:

  ```
  <workspace-root>/
  ├── <this repo>/
  └── torchtitan/        # git clone -b attention_residual_dev git@github.com:QIU023/torchtitan.git
  ```

- TensorBoard event files and checkpoints — gitignored (too large / not
  essential). Only the comparison plot and the training log tails are
  committed as evidence.
