# Rows for the PR bodies, measured 2026-09-02

Every cell here is on the branch MERGED WITH upstream/main `1dcb14a0c`, seed 42,
`--debug.deterministic`, one seed checkpoint per flavor, warm + measure passes
(the measure pass is the row). titan's SM100 whitelist in `kda.py` was lifted
locally for every run -- attn-gym `b37773f` routes SM120 through its portable
kernels, the whitelist is the only thing in the way, and it is the maintainers'
to remove. Batch: 8192 tokens per step, 256 per microbatch per rank, unless
stated.

## Control (plain `kimi_k3_debugmodel`, core's sign-step hook)

| cell | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 12.52977 | 7.27107 | 2.98077 |
| dp2 | 12.53137 | 7.31248 | 3.15823 |

## QB (`kimi_k3_debugmodel_qb`) -- pasted into PR_BODY_QB.md

| cell | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 12.52977 | 7.30620 | 3.11376 |
| dp2 | 12.53137 | 7.19897 | 3.24552 |

Step 1 equals the control to the digit: QB rewrites the bias at the optimizer
step, so the first forward cannot differ. Re-measure with ep cells after the EP
PR merges.

## LoRA (`kimi_k3_debugmodel_lora`)

| cell | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 12.45603 | 11.93088 | 10.42394 |
| dp2 | 12.48369 | 11.89999 | 10.51706 |

QLoRA-MXFP4 (`kimi_k3_debugmodel_qlora_mxfp4`): <pending>

## AC (ac_review1)

| cell | change | step 1 | step 3 | step 10 | peak memory |
|---|---|---|---|---|---|
| dp1 | main | 12.52977 | 7.27107 | 2.98077 | -- |
| dp1 | residual checkpoint wrap | 12.52977 | 7.27107 | 2.98077 | 12.68 GiB |
| dp1 | wrap + ac_reuse_attention | <pending> | | | |

The wrap is bitwise against main: the residual math is recomputed, not changed.

## PP cache offload (pp_review1), `kimi_k3_debugmodel_32l`, 4096 tokens/step

| cell | offload | step 1 | step 3 | step 10 | peak memory |
|---|---|---|---|---|---|
| pp2 x vp2 | off | 12.49999 | 6.89362 | 3.28050 | 10.43 GiB |
| pp2 x vp2 | on | 12.49999 | 6.89362 | 3.28050 | 10.42 GiB |

Bitwise; the parked blocks are ~1 MB each at this scale, so memory does not move.

## TP rewrite (tp_review1)

tp2 / dp1: <pending>; tp2 + SP: <pending>

## veRL

MoE GRPO (bf16, vLLM 6dc76a9ad): <pending>; ep2 / cp2 / pp2 / QAT ladder: <pending>

## Environment notes that cost time today

- `venv_verl`'s old vLLM (`1.0.0.dev20260801+cu130`, a source build) lists
  `torch/distributed/checkpoint/*` and `ray/data/checkpoint/*` in its RECORD;
  uninstalling it deletes both. That, not a matrix cleanup, is what emptied
  `torch.distributed.checkpoint` -- it happened again, reproducibly, on the
  swap to `0.1.dev1+g6dc76a9ad`. Repaired with exact reinstalls
  (`torch==2.14.0.dev20260801+cu130` from the nightly index, `ray==2.58.0`).
- verl gates vLLM on `>= 0.7.0` by package version; the source build reports
  `0.1.dev1+g<sha>`. `VERL_VLLM_VERSION=0.11.0` is verl's own override.
- mx3's seed cache is keyed on flavor + batch, not tree; a cross-tree hit fails
  loudly on DCP shape (`dt_bias [16,64] vs [16,128]`); use `SEED_ROOT` per tree.
