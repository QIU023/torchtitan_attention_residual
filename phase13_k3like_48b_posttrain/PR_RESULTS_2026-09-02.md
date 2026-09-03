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
| pp2 (`rl_vit1`, engine PP, token budget 2048, micro-batch 2, offload off) | 2.57e-4 | 2.41e-4 | 1.83e-4 | five attempts to get here: DEP's two vision stages need a 3-stage pipeline (`rl_vit1`); torch's 1F1B needs one microbatch per stage (chunked driver); no `fsdp` mesh at dp1 (tolerant lookup); the HF adapter's layer-0 placeholders assumed a whole-model state dict (stage-aware); stages size their P2P buffers once, so micro-batches are padded to a fixed token budget; the offload failure that kept offload off here was a masked OOM, see the offload row |
| pp2 + param/optimizer offload (`rl_vit1`, token budget 1024, micro-batch 2) | 2.57e-4 | 2.40e-4 | 1.77e-4 | the `reset_sharded_param` AttributeError was never the bug: the last stage's lm_head ran out of memory at the 2048 budget (640 MiB for the padded logits, 14.3 GiB already resident), and the train-context exit then offloaded a half-finished iteration, whose FSDP2 root group sits in FORWARD state where reshard() is a no-op, so the offload's error replaced the OOM. verl `261594b7`: the context exits reshard every FSDP module on a normal exit and call FSDP2's `reset_iter_state` on an exception, so the real error surfaces. Offload-on cells carry a one-time peak growth after the first offload/reload cycle (ep2 11.2 -> 14.0 GiB, cp2 10.0 -> 11.3 GiB, then flat); pp2's last stage sits at 12.6 GiB at the 2048 budget without offload and 13.6-13.7 GiB at the 1024 budget with it, so on 16 GB cards offload under PP needs the smaller budget |
| tp2 (`tp_review1` tree, micro-batch 2) | 3.66e-4 | 4.44e-4 | 5.34e-4 | at micro-batch 4 the entropy pass over the TP-gathered `[T, V]` logits ran out of the 16 GB (2 GiB per micro-batch); micro-batch 2 completes all three steps |
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

- 2026-09-03 00:14: the disk watchdog's pruner (`prune_run_artifacts.sh`, `find /workspace -name checkpoint -type d`) deleted `torch/distributed/checkpoint` in `venv_verl` and `venv_ptnightly`, `ray/data/checkpoint`, and the checkpoint a running matrix cell (dp2 x cp2) had just copied in, which then trained from a fresh init and was caught only by the seed assertion. The same sweep ran on 2026-09-02 09:25, which is when torch DCP first went missing from `venv_verl`; the note below that blamed the old vLLM package's RECORD was wrong. Pruner narrowed: skips site-packages and any venv, requires `step-*` inside the folder, and leaves folders touched in the last 20 minutes alone. Both venvs reinstalled (`--reinstall --no-deps`, same versions).

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

## CP ported onto Attention Gym's delta-rule recipe (`cp_review2` = `785cf2072`, on `tp_review1`)

attention-gym PR 453 (`7c83f6c`, unmerged, checked out in the editable submodule) adds `context_parallel_kda` and `context_parallel_conv_history`: per-fragment affine state summaries exchanged by all-gather, the ordinary local kernel run from the true entry state, backward as the mirror image. `cp_review1` (fla `build_cp_context` / `causal_conv1d_cp`, written before main's KDA moved to attn-gym) was ported onto main's KDA: `InnerKDA` builds a `ContextParallelPlan` for equal contiguous shards, takes the conv history from the plan, and `KDAKernel` calls `context_parallel_kda` in place of `chunk_kda`; MLA keeps Ulysses; the contracts moved to `context_parallel.py` on `SpmdType` (main removed `SpmdLayout`). One document per batch under CP, packed boundaries raise. The SM100 guard is lifted locally as before, not committed.

`kimi_k3_debugmodel`, 8192 tokens per step, 256 per micro-batch, seed 42, deterministic, same seed checkpoint as the TP matrix:

| cell | world | step 1 | step 3 | step 10 | tps |
|---|---|---|---|---|---|
| dp1 (same tree) | 1 | 12.52977 | 7.27107 | 2.98077 | |
| cp2, warm pass | 2 | 12.53972 | 7.18344 | 2.93330 | 34 |
| cp2, measure pass | 2 | 12.53972 | 7.18344 | 2.93330 | 34 |

Warm and measure passes are bitwise equal, and dp1 on this tree is bitwise equal to tp_review1's dp1. The step-1 gap to dp1 (0.010) is the size the old branch showed at seq 1024 (12.53996 vs 12.60544). Throughput, profiled (step-4 trace, cp2 rank 0, 139.7 s per step): NCCL kernels 99.8 s of wall, of which ReduceScatter_Sum_f32 47.3 s (928 launches, all FSDP2 gradient reduce-scatters) and AllGather 52.1 s (5440 launches, mostly FSDP2 parameter all-gathers); the recipe's own ranges (`_ContextParallelChunk` fwd 4.05 s + bwd 2.19 s, conv halo 0.68 s, Ulysses k_rope gather 0.34 s) total ~7 s, and KDA, conv and flex kernels under 1 s. titan folds cp into the FSDP mesh (fsdp = dp_shard x cp = 2), so at 256-token micro-batches cp2 pays a parameter all-gather and an fp32 reduce-scatter per layer per micro-batch on PCIe for 32 micro-batches per rank (plain dp2 pays it for 16 and lands at 80-87 tps), and the LL kernels spin while the two ranks wait on each other. The recipe's cost is visible only at equal per-rank work:

| cell | tokens per micro-batch (global) | tokens per rank | FSDP rounds per rank per step | tps per device | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|---|---|
| dp2 | 2048 | 2048 | 2 | 601 | 12.46307 | 8.54511 | 3.72989 |
| cp2 | 2048 | 1024 | 4 | 264 | 12.62527 | 7.63562 | 3.40683 |
| cp2 | 4096 | 2048 | 2 | 509 | 12.60312 | 9.05076 | 3.90039 |
| dp2, `cp_review1` before the attn-gym port (fla KDA, same upstream base) | 2048 | 2048 | 2 | 657 | 12.60264 | 8.69954 | 3.99567 |
| cp2, `cp_review1` before the attn-gym port (fla `build_cp_context` KCP) | 4096 | 2048 | 2 | 550 | 12.55015 | 9.29933 | 4.08062 |

At 2048 tokens per rank cp2 is 15% below dp2 on the attn-gym tree and 16% below on `cp_review1` before the port (the same upstream base with fla's KDA kernels, so both of its cells run ~9% faster in absolute terms), which is the KCP state exchange, conv halo and Ulysses all-to-alls and is the same for both recipes; the rest of the gap at short micro-batches is FSDP rounds and rank skew, not CP. (dp2 rows change the data order, so their losses are not comparable to cp2's.) CPU: the two ported CP tests pass (6), the K3 CPU sweep passes (19; `test_torch_checkpointing.py` fails to collect on this box regardless of branch). Not run: cp2 x tp2, dp2 x cp2, the multimodal splice under CP, the CI cell (`kimi_k3_cp2`, ported).

Combinations, same tree and seed (`cp_review2` = `22b89b46a`, 256-token micro-batches, 8192 per step):

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp2 | 2 | 12.53137 | 7.31248 | 3.15823 |
| dp2 x cp2 | 4 | 12.52908 | 7.21769 | 3.16754 |
| tp2 (no SP, from the TP matrix) | 2 | 12.55057 | 7.52677 | 3.00361 |
| tp2 x cp2 (no SP) | 4 | 12.55340 | 7.15046 | 3.05836 |

tp2 x cp2 needed three port-time fixes (`22b89b46a`): the CP vision splice runs `masked_scatter` on the local shard (DTensor has no sharding strategy for it) and re-wraps with the stream's placements, whose cp axis is already Shard(0); the Ulysses exchange strips the DTensor shell on entry and again after the attention module, whose TP sharding config re-wraps its output. CP with sequence parallel is rejected (the splice needs the whole sequence). The first dp2 x cp2 attempt was silently trained from a fresh init because the disk watchdog deleted its seed mid-copy (environment notes); the rows above are the rerun with the seed asserted.

Where this stands against upstream's CP design, checked on main 2026-09-03: upstream CP is ShardingConfig-driven and gated on `spmd_backend='spmd_types'` (`validate_cp_backend`); full attention gets CP from `set_gqa_inner_attention_local_map`, whose local_map keeps q token-sharded on the cp axis and all-gathers k/v to Replicate (grads Partial), i.e. all-gather-KV, not Ulysses; tp_review1 already installs that declaration on K3's MLA inner attention, but all-gather-KV keeps the full-length K/V resident on every rank (no K/V memory saving from cp, and ~cp/2 more bytes per layer than Ulysses), so for MLA at long context Ulysses stays the algorithm; under spmd_types it is a declared redistribution on the inner attention (cp axis S(0) -> S(1) in, the reverse out), which is exactly the ULYSSES pair in CP_DECLARATIVE.md, replacing the all-gather declaration rather than being replaced by it. Upstream's hybrid precedent qwen3_5 rejects CP outright ("GatedDeltaNet requires full-sequence allgather"), so there is no upstream shape for linear-attention CP; KCP would be K3's own local_map body with the cp axis Shard(0) on both sides and the attn-gym recipe inside. K3 has no spmd_types path at all: the config registry pins `partial_dtensor`, and with that pin lifted the tree fails in `parallelize_kimi_k3` at `get_mesh(["fsdp"])` (spmd_types names the axis `dp_shard` and upstream models branch through `resolve_fsdp_mesh`). So a CP PR that passes `validate_cp_backend` without the `_validate_cp_backend` override needs the K3 spmd_types migration first (parallelize, FSDP mesh resolution, and the sharding declarations becoming load-bearing); the old `k3_cp_declarative` branch got EP running under spmd_types and TP running but not matching numerically, on a much older base.

## K3 under spmd_types, probed (`spmd_probe` = `b151fcf06` on the fork, on top of `cp_review2`)

Why it matters: upstream removed the partial_dtensor CP path on 2026-08-19 (PR-4218, "too much to maintain different CP paths; only config-based CP is preserved"), the default backend is now `spmd_types`, every model except K3 has both branches, and three TODOs say partial_dtensor itself goes next. K3 pins `partial_dtensor` and refuses anything else. EP and TP were never imperative: they are ShardingConfig declarations consumed through the partial_dtensor driver, which is why the EP PR needed nothing.

What the probe changed (2 steps, 512 tokens per step, no seed checkpoint, losses only comparable within this table):

| step | change | lines |
|---|---|---|
| parallelize | `spmd_types` branch: `resolve_fsdp_mesh` / `resolve_sparse_fsdp_mesh`, `dp_mesh_dims` / `edp_mesh_dims` through to the decoder and the vision encoder, `model.parallelize` under spmd_types, pin check removed (the deepseek_v3 shape) | +26 / -18 |
| model config | the TP declarations issued whenever the backend is spmd_types, not only at tp > 1 | +4 |
| `preprocess_inputs` | override giving `pixel_values` / `grid_thw` SPMD layouts (`multimodal_input_sharding`, the qwen3_5 shape) | +35 |
| sharding | MoonViT tower declared (the kimi_k2_7 plan, K3's projector norms after its second linear) | +30 |
| kda | head views use -1: under spmd_types the projections hand back the TP-local head slice | +4 |
| vision helpers (shared) | cp axis added to every SpmdType, a mechanical sed for the probe: upstream's VLMs never carry cp because none supports CP | probe only |

| cell | spmd_types | partial_dtensor, same flags |
|---|---|---|
| dp1 | 12.53529 / 11.44489 | 12.53529 / 11.44489 (bitwise) |
| dp2 (3 steps) | 12.50788 / 11.20827 / 9.87694 | 12.50788 / 11.20827 / 9.87694 (bitwise) |
| ep2 | 12.50788 / 11.18843 | 12.50788 / 11.18843 (bitwise) |
| tp2 (no SP) | 12.51636 / 11.50026 | not run: the probe's unconditional tower declaration breaks partial_dtensor TP (`pos_embed` becomes a DTensor and MoonViT's position lookup indexes it with a plain tensor); the declaration has to be gated on the backend |
| cp2 | reaches the MLA kernel: `block_mask was created for a smaller length` | n/a |

An earlier spmd_types dp2 run (before the cp entries were added to the vision helpers) read 11.19137 at step 2; the rerun above matched partial_dtensor bitwise and the difference was not reproduced, cause not established.

The cp2 failure is the collision predicted in the CP note: the imperative Ulysses has already all-to-all'ed q/k/v to the full sequence and rebuilt the full mask when the inner attention's declaration (`set_gqa_inner_attention_local_map`, all-gather-KV on the cp axis) gathers k/v a second time. The KDA layers before it (layers 0-2) passed: `inner_kda`'s local_map declares the cp axis S(0) on both sides and the attn-gym recipe's collectives inside the body are invisible to the type system, so KCP's "identity pair with the recipe in the body" already holds. MLA is the one piece that has to change shape: Ulysses becomes the inner attention's declaration (cp S(0) -> S(1) in, the reverse out, replacing the all-gather-KV entry), and the full-sequence mask has to be supplied to those layers instead of the Q-sharded one.

Size of the migration from here: the five rows above (about 100 lines, of which the probe's shared-helper sed needs a real form, likely a `cp=` argument on the vision helpers), the backend gate on the tower declaration, the Ulysses declaration plus mask handling, then 10-step seeded matrices for dp/ep/tp/cp under spmd_types against the partial_dtensor rows. EP PR delta: the parallelize rows only; the EP declarations and dispatcher ran unchanged under spmd_types.

## K3 under spmd_types on `k3_on_4025` (2026-09-03, in progress)

Branch map first, since the earlier notes misnamed trees: `old_tree_full_impl` and the `attention_residual_dev` family are the old tree; everything based on 30eb5e502 or later is the new tree, including `k3_on_4025` (the integration branch with every parallelism ported in), `cp_review1` / `k3_cp` (fla KDA), `tp_review1` / `cp_review2` / `spmd_review1` (attn-gym KDA, on `k3_ep`).

Done on `/workspace/tt_4025/torchtitan` (k3_on_4025, fla KDA), all under `--parallelism.spmd_backend spmd_types`:

| change | why |
|---|---|
| parallelize: spmd_types branch (`resolve_fsdp_mesh` / `resolve_sparse_fsdp_mesh`, `dp_mesh_dims` to decoder and tower, `model.parallelize` under spmd_types, pin check gone) | the deepseek_v3 shape |
| `_set_sharding_config(declare_all=...)`: dense declarations issued under spmd_types at any TP degree; the ep+tp special cases stay on the real TP flag | spmd_types needs a layout for every parameter |
| MoonViT tower declared (kimi_k2_7 plan) but invariant at TP and rank-local over cp, with a cp axis on every layout; `pos_embed` / `inv_freq` R | the shared helpers shard the tower over TP and all-gather its k/v over cp; here it runs whole on every rank, as under partial_dtensor |
| KDA projections, convs, forget/beta/gate, attn-res and output-res projections, `routed_up`: `tp=I`; `routed_down`, `wq_a`, `wkv_a`: `tp=R` | R sums the gradient across TP, right only when the consumer is TP-sharded and each rank's gradient is a partial sum; on identical replicas it doubles (KDA family was 2.15x, `ffn_res_proj` 2.56x) |
| `A_log` / `dt_bias` carry a cp entry | cp2 could not place them |
| empty residual stack converted R -> I at the model entry | a fresh tensor reads R; the stream is I; the stack concatenates both |
| `routed_norm` boundary P -> I keyed `"x"` | the experts' rowwise output is Partial and nothing reduced it before the norm under spmd_types (DTensor did implicitly); `norm_config`'s `"input"` key never binds because `nn.RMSNorm.forward` names its argument `x` |
| `routed_up` output I -> P, MoE module exit R -> I, MLA module entry I -> R, `q_norm` / `kv_norm` R state-only, KDA `output_norm` state-only | each a type mismatch `--debug.spmd_typechecking` reported in turn |
| MLA head unflatten and rope join in `spmd.local()` regions re-typed head-sharded | the checker cannot propagate a feature shard through the split, core's `local_qkv_head_split` pattern |
| KDA head views use -1 | spmd_types hands the projections' TP-local slice back |
| `pixel_values` / `grid_thw` asserted `{DP: V, CP: V, TP: I}` in the model | the trainer types only tokens/labels/positions on this base |

`--debug.spmd_typechecking` (needs `activation-checkpoint:none`; SAC with Flex is refused, and the tower had to be skipped behind a typed boundary because kimi_k2_7's own declarations are not typecheck-clean) now passes end to end for tp2. Seeded 10-step rows so far (same tree, `.mx3_seeds`, 8192 tokens per step, 256 per micro-batch): dp2 spmd_types 12.45679 / 6.92636 / 3.26972 equals dp2 partial_dtensor bitwise; dp1 spmd_types 12.46442 / 7.27823 / 3.07888 equals the clean tree's dp1 partial_dtensor bitwise; ep2 spmd_types 12.45810 / 6.93768 / 3.28616 (run before the TP-type fixes; dp/ep paths do not exercise them). The clean-tree partial_dtensor references: cp2 12.44640 / 7.39556 / 3.09412. tp2 at 2 steps (512 tokens per step, no seed, same flags) after the declarations above: spmd_types 12.55577 / 10.92986 with grad norms 21.625 / 15.5 against partial_dtensor 12.52067 / 10.80010 with 21.5 / 15.375; per-parameter gradient norms agree within 5% for every family except `output_res_proj` (15% high), so the remaining difference is small and localized but not explained. Committed as k3_on_4025 `9f87e0891`. Seeded 10-step rows on the final code (same seed and batch as above):

| cell | spmd_types | partial_dtensor | dp1 (both backends, bitwise) |
|---|---|---|---|
| tp2 (no SP) | 12.47125 / 7.22017 / 3.10949 | 12.47604 / 7.27227 / 3.06511 | 12.46442 / 7.27823 / 3.07888 |
| cp2 | 12.44640 / 7.37571 / 3.10927 (Ulysses declared, KCP in the body) | 12.44640 / 7.39556 / 3.09412 (clean tree, imperative Ulysses) | |

tp2 under either backend sits within 0.06 of dp1 at every step and the two backends within 0.05 of each other, the size of TP's own reduction-order effect (partial_dtensor tp2 is 0.012 / 0.006 / 0.014 from dp1); nothing systematic is left. cp2 matches the imperative path bitwise at step 1 and within 0.02 after, the declared all-to-all against the packed one.

## The same declarations on the attn-gym base (`spmd_review1` = `4b88ada6b`, CP review branch = `a6d26f19f` + merge)

Decided with the user: validation continues on `k3_on_4025` (every parallelism is there to cross-check), landing happens on the attn-gym base (`tp_review1` lineage), no more work on the fla base. The declaration fixes port with one difference: this tree's KDA is head-parallel (colwise projections behind a `local_map` on `inner_kda`), so the KDA-side R/I changes do not apply; `forget_a` and the KDA output norm stay R because their consumers are TP-sharded. Everything else carries over: residual and output-residual projections and `routed_up` invariant, the R -> I stack seed, the MLA entry I -> R, replicated state-only q/kv norms, the `routed_norm` P -> I boundary keyed `x`, `routed_up` back to Partial and the MoE exit to I, the tower invariant at TP and rank-local over cp with its own attention entry, and the head unflatten / rope join in local regions.

tp2 at 2 steps on this base (no seed, 512 tokens per step): spmd_types 12.37781 / 10.92669 with grad norms 24.375 / 19.125 against partial_dtensor 12.36441 / 10.92794 with 25.125 / 20.0. Seeded 10 steps (`.mx3_seeds_upmain`, 8192 tokens per step, 256 per micro-batch):

| cell | spmd_types | partial_dtensor |
|---|---|---|
| dp1 | 12.52977 / 7.27107 / 2.98077 | same (bitwise, the tp_review1 dp1) |
| dp2 | 12.53137 / 7.31248 / 3.15823 | same (bitwise) |
| ep2 | 12.53146 / 7.20212 / 3.10296 | |
| tp2 (no SP) | 12.55332 / 7.38015 / 3.00522 | 12.55057 / 7.52677 / 3.00361 |

The CP review branch's CPU sweep: 718 passed, 0 failed, the three known collection errors of this box.

The CP review branch (worktree `cp_review1_new`, to be pushed over `cp_review1` once verified) now carries: the attn-gym port of CP (`785cf2072`, `22b89b46a`), the spmd_types support, and Ulysses as the MLA inner attention's declaration (`a6d26f19f`): cp axis S(0) -> S(1) on q/k/v in and the reverse out with TP's head split kept inside, the full-sequence masks held out of the CP input sharding in `preprocess_inputs`, the hand-written all-to-all, mask rebuild and contract classes removed with their tests, a CPU test asserting the declaration; KDA unchanged (recipe in the `inner_kda` body). CP therefore requires spmd_types, which is what upstream's `validate_cp_backend` demands; the `_validate_cp_backend` override is gone; the `kimi_k3_cp2` test flavor selects spmd_types (`84fd3ee22`). Seeded 10-step cp2 on the merged branch (same seed and batch as the CP port table): 12.53972 / 7.18619 / 3.00631 against the imperative CP under partial_dtensor 12.53972 / 7.18344 / 2.93330 and dp1 12.52977 / 7.27107 / 2.98077: bitwise at step 1, the declared all-to-all against the packed one after.

## EP PR re-merge (2026-09-03, `k3_ep` = `cd8856107`)

The PR had gone dirty against main (21 commits, two of them on the MoE path: the grouped-mm seam takes the expert weight untransposed, and the attention shape suffixes) and tianyu-l asked for `set_kimi_k3_sharding_config`. One merge commit resolves the only conflict, `kimi_k3/__init__.py`, where upstream's `refactor max_context_length` made `model_registry` take `seq_len` and the registry hold `(builder, max_context_len)`; the resolution keeps `moe_comm_backend` and passes it to the builder, nothing else. The rename is a 3-line commit. Before/after from one seed (`.mx3_seeds_upmain`, 8192 tokens per step, 256 per micro-batch), 69f84292d against the merge, with the SM120 guard lifted locally on both:

| cell | before | after |
|---|---|---|
| dp1 | 12.52977 / 7.27107 / 2.98077 | same |
| dp2 | 12.53137 / 7.31248 / 3.15823 | same |
| ep2 x fsdp2 | 12.53146 / 7.20212 / 3.10296 | same |

Bitwise on all three, so the two upstream MoE commits do not change K3's EP numerics. CPU: 37 passed on the merged tree. Pushed to `k3_ep` as a fast-forward on the user's word; GitHub reports mergeable with CI running.



## PP PR 4312: first review (Tianyu, 2026-09-03) and the one-line fixes

Review is CHANGES_REQUESTED with 18 inline comments. The one-line ones are on fork branch `pp_review2` (`0c0e48908`, two commits on the PR head `087c4d177`; `pp_review1` is the integration branch with offload and Mooncake, so it was left alone): the shape-suffix legend restored (the reflow was churn), `opens_block` renamed `first_layer_in_block` (his suggestion), `_infer_block_layout_tables_from_stages` made public (private call across files), the pp8 x vp4 integration cell dropped in favour of pp2 alone, and `pipeline_adapter.py` reformatted with the current ufmt (the PR head's file already failed the local hook). `test_kimi_k3_pp_fqn_injection` 4/4. Not pushed to the PR branch.

The rest is design or numerics, not one-liners:

| comment | nature |
|---|---|
| debugmodel shape (no defaults, irregular like 93 layers, block size always 12, `full_attention_layers` deducible) | conflicts with the adapter's even-split precondition; changing the flavor reruns the pp x vp matrix |
| why no `// 2` as in the paper | the paper's pseudo-code counts sub-layers (`layer_number % (block_size // 2)`, "each transformer layer has 2"); ours counts layers, 12 layers = the paper's 24, so no `// 2` -- answer in thread |
| cat at the start of the layer, then `_apply_attention_residual` with `prefix_sum` None | equivalent (same V stacking order); readability refactor, needs a dp1 bitwise check |
| `attn_res_cache` on the model config is a PP-infra flag; "inject"; only-incremental transfer and release; block tensor lifecycle; how 93 layers divide | the design notes he asked for ("how you got there") |
| step-10 spread far above reduction order | fallback transport pp2/pp4/pp8 at step 10: 3.42 / 3.50 / 3.63, monotone in the pp degree; looks systematic, needs its own investigation |


## 2026-09-03/04 overnight: CP kernel restructure, PP review fixes, the review answers

Everything is in `REVIEW_ANSWERS_PP_CP_2026-09-04.md` (same folder): the adapter's design history reconstructed from the April phase3 logs, the answer to every comment on PR 4312 and the CP comments on PR 4313, the torch PP cache question, and the numerics evidence. Summary of what changed and what was measured:

- `cp_review2` = `adc012ce4`: Ulysses and KCP as `ContextParallelKernel`s that own their collectives (`f66b5de3a`), the plain-tensor splice rejecting CP with SP (`3970bcd1c`), the all-gather KV kernel copied from PR 4322 as the second MLA choice (`adc012ce4`). With one shared warm inductor cache, all 750 parameter gradients are sha1-identical between the kernel tree and the declarative tree at steps 1 and 2; the step-2 spread seen earlier (9.53739 / 9.53178 / 9.54240) is inductor autotune picking different FlexAttention kernels at cold compile, not a code difference.
- `pp_review2` = `ca5f34ea8`: the one-line fixes, the transport switch off the model config, the split as a pure function, the stale naive-mode warning removed, and uneven pipeline splits supported (layer-to-stage map gathered over the PP group).
- PP step-1 gradients, K3 24-layer debug flavor, seed checkpoint, 32 micro-batches of 256 tokens in every cell: dp1 vs pp2 fallback median 1.3e-4 / max 8.6e-3 relative per-parameter norm difference; transport off vs on at pp2 x vp2 median 1.6e-4 / max 1.2e-2; dp1 vs pp2 x vp2 delta median 1.2e-4 / max 1.3e-2; step-1 loss identical (12.59997) in all of them. The delta transport is as far from the fallback as the fallback is from a single GPU.
- Two probe-design traps recorded: a dp1 run with 1024-token micro-batches packs the data differently and is not comparable with PP cells that use 8 pipeline micro-batches of 256 in 4 accumulation rounds; and a PP cell that compiled while another job shared its GPUs came out with a different step-1 loss (12.60011).
- PP review-branch validation (`ca5f34ea8`): pp2 x vp2 even split on the new code is sha1-identical in all 750 gradients to the PR head on the same warm cache; an uneven split (7 layers per stage, first/last stage one less: 6 / 7 / 6 / 5 layers, block boundary inside stage 1) trains with the delta transport from the gathered layer-to-stage map, step-1 loss 12.59997 = dp1, gradients vs dp1 median 9.7e-5 / max 2.3e-2, transport off vs on median 1.3e-4 / max 2.0e-2.

- `pp_review2` continued (2026-09-03 evening): `eef340d25` the cat-at-the-start refactor (bit-exact at dp1 and pp2 x vp2 on a shared cache), `d72faf339` the rank cache releases a micro-batch's blocks after the rank's last virtual stage forward (bit-exact; no memory change at debug scale, the cache is a few MB there), `395fc6b30` one irregular debug model (30 layers, block 12, MLA at every fourth layer and the last, no defaults; the 32-layer flavor gone), `c3df74847` a stage passes its inputs through `_DenseGradient` (identity, contiguous backward). The last one is a torch.distributed.pipelining bug the 32-stage cell exposed: the backward receive buffer is allocated with the strides of the next stage's input gradients, and a stage whose first op on an input is a cat (the head-only last stage) gets view gradients, so c10d rejects the buffer; reproduced on 2 GPUs as pp2 x vp16 with the transport off, fixed model-side, worth an upstream issue.
- The pp x vp matrix on the 30-layer model (4096 tokens per step, 256 per micro-batch, 8 pipeline micro-batches, first/last stage one layer less, each cell twice, second run read): dp1 12.44394 / 7.32431 / 3.44458; pp2 / pp4 / pp8 (1F1B) 12.44394 at step 1 with 3.40260 / 3.37749 / 3.43359 at step 10; pp2 x vp2 12.44394 / 7.38716 / 3.72131 (transport off 7.47149 / 3.68055); pp2 x vp4 7.40650 / 3.46752; pp4 x vp2 7.49078 / 3.47512; pp4 x vp4 7.42482 / 3.40743 (off 7.45038 / 3.39832); pp8 x vp2 7.38036 / 3.38767; pp8 x vp4 (32 stages, embedding-only and head-only stages, after `c3df74847`) 12.44394 / 7.29935 / 3.29156. Runs: `mx3_pp30_0903_165734`, `mx3_pp30n_0903_174622`, `mx3_pp30fix_0903_182739`, `mx3_pp30fixn_0903_183337`.
- pp8 x vp4 with the transport off: the matrix cell (eight ranks autotuning FlexAttention at once) read 12.45856 at step 1; rerun on the delta cell's warm inductor/triton caches it is 12.44394 / 10.09382 / 7.45038 (steps 1-3), i.e. the cold-compile lottery moved step 1, not the code. Every one of the 13 cells is now at 12.44394 on a warm cache.
- `3af70c9ee` design B: `AttnResPipelineStage`, a `PipelineStage` subclass reached through the new `pipeline_llm(stage_class=...)`, replaces the adapter (1228 lines of wrappers gone; `pipeline_stage.py` 380 lines, `pipeline.py` the entry and the split). Validation on the 30-layer model, seed checkpoint, per-parameter step-1 gradients: subclass vs hook adapter at pp2 x vp2 on one shared cache median 3.5e-5 / p90 5.3e-4 / max 9.2e-3 (the runtime sums the block gradients where autograd accumulated them); subclass delta vs dp1 1.5e-4 / 1.2e-3 / 1.6e-2, the same as the adapter's 1.2e-4 / 1.2e-3 / 1.6e-2; transport off vs dp1 1.3e-4 / 1.1e-3 / 2.1e-2; 32 stages on 2 GPUs (pp2 x vp16, embedding-only and head-only stages, no model-side gradient boundary) 2.0e-4 / 1.7e-3 / 1.5e-2. Cells: pp8 x vp4 12.44394 / 7.36221 / 3.50686, pp4 x vp4 12.44394 / 7.42872 / 3.52641, pp2 x vp4 12.44394 / 7.37798 / 3.58093, pp8 x vp4 transport off 12.44394 / 7.45038 / 3.40014 (`mx3_ppsub2_0903_195842`, `mx3_ppsub2n_0903_201330`).
