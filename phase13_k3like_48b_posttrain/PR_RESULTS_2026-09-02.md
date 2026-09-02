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

QLoRA-MXFP4 (`kimi_k3_debugmodel_qlora_mxfp4`): dp1 12.48328 / 12.00891 / 10.42474; dp2 12.50176 / 12.00203 / 10.45663

## AC (ac_review1)

| cell | change | step 1 | step 3 | step 10 | peak memory |
|---|---|---|---|---|---|
| dp1 | main | 12.52977 | 7.27107 | 2.98077 | -- |
| dp1 | residual checkpoint wrap | 12.52977 | 7.27107 | 2.98077 | 12.68 GiB |
| dp1 | wrap + ac_reuse_attention | 12.52977 | 7.27107 | 2.98077 | 12.81 GiB |

The wrap is bitwise against main: the residual math is recomputed, not changed.

## PP cache offload (pp_review1), `kimi_k3_debugmodel_32l`, 4096 tokens/step

| cell | offload | step 1 | step 3 | step 10 | peak memory |
|---|---|---|---|---|---|
| pp2 x vp2 | off | 12.49999 | 6.89362 | 3.28050 | 10.43 GiB |
| pp2 x vp2 | on | 12.49999 | 6.89362 | 3.28050 | 10.42 GiB |

Bitwise; the parked blocks are ~1 MB each at this scale, so memory does not move.

## TP rewrite (tp_review1)

tp2 12.55057 / 7.52677 / 3.00361 against dp1 12.52977 / 7.27107 / 2.98077 (same tree); tp2 + SP 12.54164 / 7.44412 / 3.08160

## veRL

MoE GRPO (bf16 rollout on vLLM 6dc76a9ad, fsdp2, 3 steps, `kimi-k3-debug` with its 32 routed experts): ran to completion. The rollout-vs-actor log-prob gap is the evidence that the two MoE implementations agree; entropy is ln(vocab) because the weights are a seeded init, and pg_loss is 0 because a random model earns no reward.

| step | rollout_probs_diff_max | rollout_probs_diff_mean | entropy |
|---|---|---|---|
| 1 | 6.41e-4 | 1.05e-5 | 11.520 |
| 2 | 5.79e-4 | -- | 11.520 |
| 3 | 6.71e-4 | -- | 11.520 |

The parallelism ladder on the same model and vLLM, 2 GPUs unless stated, `rollout_probs_diff_max` per step:

| cell | step 1 | step 2 | step 3 | note |
|---|---|---|---|---|
| fsdp2 (above) | 6.41e-4 | 5.79e-4 | 6.71e-4 | |
| fsdp2 x ep2 | 7.09e-4 | 5.60e-4 | 5.30e-4 | EP carved from the dp axis |
| fsdp2 x cp2 (4 GPUs) | 4.28e-4 | 5.25e-4 | 5.32e-4 | first attempt was misconfigured (dp2 x cp2 on 2 ranks); a dataloader worker was OOM-killed at teardown after step 3, as in the fsdp2 cell -- host memory, not the run |
| pp2 (`rl_vit1`, engine PP, token budget 2048, micro-batch 2, offload off) | 2.57e-4 | 2.41e-4 | 1.83e-4 | five attempts to get here: DEP's two vision stages need a 3-stage pipeline (`rl_vit1`); torch's 1F1B needs one microbatch per stage (chunked driver); no `fsdp` mesh at dp1 (tolerant lookup); the HF adapter's layer-0 placeholders assumed a whole-model state dict (stage-aware); stages size their P2P buffers once, so micro-batches are padded to a fixed token budget; `offload_fsdp_model_to_cpu` still trips on a stage parameter that is not an FSDP DTensor, so offload stays off under PP for now |
| QAT (`kimi_k3_debugmodel_rl_mx_qat`, fsdp2, micro-batch 2) | 7.12e-4 | 5.75e-4 | 5.21e-4 | MXFP4/MXFP8 fake-quant on the routed experts, bf16 rollout; at micro-batch 4 the step-3 log-prob pass ran out of the 16 GB (the STE keeps dequantized expert copies alive), so the row is the micro-batch-2 rerun |

QAT under each parallelism (micro-batch 2; PP cells with the 2048-token budget, offload off):

| cell | step 1 | step 2 | step 3 |
|---|---|---|---|
| QAT x ep2 | 7.17e-4 | 5.87e-4 | 6.06e-4 |
| QAT x cp2 (4 GPUs) | 5.30e-4 | 4.88e-4 | 5.27e-4 |
| QAT x pp2 (`rl_mx_qat_vit1`) | 3.31e-4 | 1.72e-4 | 1.56e-4 |
| QAT x ep2 x cp2 x pp2 (8 GPUs, fsdp2) | 1.81e-4 | 3.31e-4 | 2.06e-4 | three attempts: the token-budget padding had to run on the whole packed stream before the CP split, and the pipeline bridge had to unfold the stage's [T, V] output before the CP gather (gathering [T, V] on dim 1 concatenates vocabularies) |

QAT GRPO runs: the fake-quantized actor trains against a bf16 rollout of the same weights, and the log-prob gap stays at the bf16 level -- the STE forward consumes dequant(quant(w)) while the rollout serves the bf16 masters, so this gap is also the quantization error the deployment path will see.

ep2 / cp2 / pp2 / QAT ladder: see table; PP in the engine was implemented tonight (verl `6ad61b56`: one verl micro-batch per schedule step, a loss bridge on the last stage) -- the ladder's pp2 cell is its first run.

## Environment notes that cost time today

- `venv_verl`'s old vLLM (`1.0.0.dev20260801+cu130`, a source build) lists
  `torch/distributed/checkpoint/*` and `ray/data/checkpoint/*` in its RECORD;
  uninstalling it deletes both. That, not a matrix cleanup, is what emptied
  `torch.distributed.checkpoint` -- it happened again, reproducibly, on the
  swap to `0.1.dev1+g6dc76a9ad`. Repaired with exact reinstalls
  (`torch==2.14.0.dev20260801+cu130` from the nightly index, `ray==2.58.0`).
- verl gates vLLM on `>= 0.7.0` by package version; the source build reports
  `0.1.dev1+g<sha>`. `VERL_VLLM_VERSION=0.11.0` is verl's own override.
- The disk watchdog's pruner sweeps every directory named `checkpoint` and the seed caches were not on its protected list: at 09:25 it emptied all seven caches at once (three SEED-COPY-FAIL aborts followed). The cache now stores `seed_ckpt`, the pruner excludes `.mx3_seeds*`, and an empty cache rebuilds instead of being trusted.
- mx3's seed cache is keyed on flavor + batch, not tree; a cross-tree hit fails
  loudly on DCP shape (`dt_bias [16,64] vs [16,128]`); use `SEED_ROOT` per tree.

## pp_balance under memory pressure (pp_review1 `1c0c1416c`, pp8 x vp4, 32-layer flavor, 8 GPUs)

Per-GPU peak from nvidia-smi sampled every second, six steps, tokens per microbatch raised; sources are the two heaviest ranks, dest the lightest.

| arm | tok/mb | rank 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 512 | 8.2 | 4.1 | 4.1 | 3.4 | 4.1 | 4.0 | 4.0 | 7.7 |
| baseline | 1024 | 8.9 | 4.8 | 4.8 | 4.1 | 4.8 | 4.7 | 4.7 | 8.1 |
| balanced (0,7 -> 3) | 1024 | 9.2 | 4.6 | 4.6 | 4.1 | 4.6 | 4.5 | 4.7 | 8.1 |

Not the case that shows the feature: at one layer per stage the activations a stage saves are small (doubling tokens per microbatch adds 0.7 GiB per rank), and the imbalance is the 163840-vocab embedding and head on ranks 0 and 7, which balancing activations cannot touch. The Transfer Engine came up (TCP on this box) and the run stayed at the baseline's loss to the digit at step 1. Next: pp2 with 16 layers per rank, where the saved activations are 16 layers deep, tokens per microbatch up to 8192.

### pp_balance, second and third shapes (pp2 with 16 layers per rank; pp4 with 8), reserved and allocated peaks

pp2, 32-layer flavor, 1F1B, mb 8 (nvidia-smi reserved peak, GiB):

| arm | tok/mb | rank 0 | rank 1 | note |
|---|---|---|---|---|
| baseline | 2048 | 13.66 | 14.13 | |
| baseline | 4096 | 14.95 | 15.32 | |
| baseline | 8192 | OOM | 15.43 | |
| balanced, 1 -> 0 | 4096 | 15.38 | 15.22 | rank 1 parked 27.6 GB over 6 steps, all fetched |
| balanced, 0 -> 1 | 4096 | 15.12 | 15.12 | rank 0 parked 11.9 GB, all fetched; loss to the digit at s1 and s6 |

pp4, 8 layers per rank, sources 0 and 1 parking on 3, tok/mb 4096, with the allocator's own peaks per rank:

| rank | baseline max_allocated | balanced max_allocated | baseline reserved | balanced reserved | parked per step |
|---|---|---|---|---|---|
| 0 | 3.84 | 3.84 | 11.18 | 11.60 | 0.96 GB |
| 1 | 2.62 | 2.62 | 7.40 | 7.25 | 1.0 GB |
| 2 | 2.72 | 2.72 | 6.95 | 6.95 | -- |
| 3 (dest) | 4.11 | 4.11 | 11.42 | 11.42 | -- (TCP: the pool is host memory here) |

The mechanism runs -- about a gigabyte a step leaves each source rank and comes back, the loss is the baseline's to the digit at step 1 -- and the peak does not move, because at this shape the tensors autograd saves per microbatch are ~120 MiB (720 parks over 6 steps of 8 microbatches, 8 MiB each) against a 3.8 GiB allocated peak that is parameters, optimizer state, grads, the embedding/logits and the AttnRes block stack. The headroom is per-microbatch saved bytes times in-flight depth; at K3's width and depth that is gigabytes per microbatch, at the debug width it is 3% of the peak. The reserved numbers cannot show it at all: freed blocks stay reserved. What this box can show is the transport and the numerics, not a peak.
