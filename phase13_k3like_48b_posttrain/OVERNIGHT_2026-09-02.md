# Overnight 2026-09-02: bodies, TP/SP on the new KDA, veRL parallelism ladder

Rows and the per-cell notes live in `PR_RESULTS_2026-09-02.md`; this is the
map. Every torchtitan run tonight lifted the SM100 whitelist in `kda.py`
locally (attn-gym `b37773f` routes SM120 through its portable kernels; the
whitelist is the maintainers' to remove).

## Branches pushed (fork), with what each carries

| branch | state | evidence |
|---|---|---|
| `k3_ep` = `69f84292d` | approved PR, merge commit onto main `1dcb14a0c`, CI cell moved out | dp2/ep2 bitwise before/after (2026-09-01) |
| `tp_review1` (4 commits on `k3_ep`) | TP rewritten for main's `InnerKDA`: head-parallel MLA and KDA behind a `local_map`, SP overlay, grad-norm mesh grouping, latent-MoE norm under SP | dp1 bitwise with main; tp2 12.55057 / 7.52677 / 3.00361; tp2+SP 12.54164 / 7.44412 / 3.08160 |
| `ac_review1` (2 commits on main) | residual math under `torch.utils.checkpoint`; `ac_reuse_attention` | both bitwise with main; 12.68 / 12.81 GiB |
| `pp_review1` (merge + format + offload) | `attn_res_cache_offload` | pp2 x vp2 bitwise with offload off |
| `k3_qb` | unchanged; body carries the sign-step control rows | dp1/dp2 with controls |
| `k3_lora_extras` | the leaked QAT flavor removed | lora dp1/dp2, qlora dp1/dp2 |
| `k3_qat` | merged main; `QuantizationConverter` base; exported | deferred behind QB/LoRA (`QAT_UPSTREAM_AUDIT_2026-09-02.md`) |
| `k3_mtp` | merged main, one import resolution | 2 CPU tests |
| `k3_on_4025` | rl flavor keeps MoE; `rl_mx_qat`, `rl_vit1` flavors; moonep return fix | -- |
| verl `kimi_k3_integration` = `d778b810` | pipeline parallelism in the torchtitan engine: one pipeline microbatch per stage, verl micro-batches fed in chunks with zero-loss padding, verl's loss reaching the last stage through a bridge on the schedule | pp2 three steps, 2.6e-4 |

Bodies in `Raising_PRs/PR_K3_PARALLELISM/`: QB, LoRA, AC, PP (offload
section), QAT (draft with its relation section), TP (rewritten), MTP (header),
CP (status note). Every row that landed tonight is already pasted in; the
placeholders that remain are named in the ledger.

## veRL on the MoE debug model, vLLM 6dc76a9ad

The rollout-vs-actor log-prob gap (`rollout_probs_diff_max`) is the agreement
metric; every cell holds it at the bf16 level:

| cell | steps | max gap |
|---|---|---|
| fsdp2 | 3 | 6.7e-4 |
| fsdp2 x ep2 | 3 | 7.1e-4 |
| fsdp2 x cp2 (4 GPUs) | 3 | 5.3e-4 |
| QAT (rl_mx_qat, micro-batch 2) | 3 | 7.1e-4 |
| pp2 (rl_vit1, engine PP) | 3 | 2.6e-4 |
| QAT x ep2 | 3 | 7.2e-4 |
| QAT x cp2 (4 GPUs) | 3 | 5.3e-4 |
| QAT x pp2 | 3 | 3.3e-4 |
| QAT x ep2 x cp2 x pp2 (8 GPUs) | 3 | 3.3e-4 |

The goal's veRL line -- QAT GRPO with EP, DP, CP and PP on -- runs on the
debug model. PP in the engine is new tonight (verl `dbc08fd7`); the
attempt trail for pp2 and for the eight-card cell is in the ledger.

The "GRPO with MoE is blocked on SITU kernel coverage" note is withdrawn
(`VERL_MOE_ROLLOUT_2026-09-02.md`).

## What blocked and how it was routed around

- The disk watchdog's pruner swept every `checkpoint` directory including the
  seed caches (three SEED-COPY-FAIL aborts); caches now store `seed_ckpt`, the
  pruner excludes them, an empty cache rebuilds.
- verl's engine raised NotImplementedError for PP; implemented tonight
  (above). torch's 1F1B needs one microbatch per stage, hence the chunking.
- DEP's default of two vision stages needs a three-stage pipeline; `rl_vit1`
  keeps the tower whole on one stage for pp2.
- The QAT cell ran out of the 16 GB at micro-batch 4 (the STE keeps dequantized
  expert copies alive); micro-batch 2 runs.
- uninstalling the old vLLM dev package deletes `torch/distributed/checkpoint`
  and `ray/data/checkpoint` (its RECORD lists them); exact reinstalls repair it.

## Still open

- KCP: attention-gym keeps the CP orchestration in an example behind private
  seams; #445 made it run on Hopper/SM120. The CP PR's KDA half waits on the
  promoted API; the Ulysses half is reviewable now.
- `pp_balance` (Mooncake) held for a dependency decision.
- QAT filing route: fold into PR-3889, or draft with the relation section.
