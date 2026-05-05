# Phase 7 Extension — Post-training Fabric Patterns

Continues the pretraining fabric profile work (v10/v11/v12 +
MODE=B 5D) with two additional traces from phase 9 post-training:

## SFT trace (from phase 9-A)

Same mesh/model as v11 but on LLaVA-Instruct conversational data.
Expected fabric pattern:
* Identical PP / FSDP / TP / EP collective shapes (mesh unchanged)
* Slightly different message-size distribution (SFT data has shorter
  conversations than pretrain captions on average)
* Lower per-step time due to smaller per-mb token count

## PPO trace (from phase 9-B)

Multi-model RLHF infra creates novel fabric patterns NOT seen in
pretraining or SFT:

| Phase | Models in flight | Mesh-multiplexing | NCCL pattern |
|---|---|---|---|
| **Rollout** | actor (4D mesh) | actor only | inference-style: small batch, long generation, KV-cache allgather |
| **Logprob compute** | actor + reference | both 4D | dual-model forward, no backward |
| **Reward eval** | reward_model (1D mesh) | RM only | small-mesh forward |
| **PPO update** | actor backward | actor only | training-style: like SFT but loss = clipped surrogate |

The cross-model communication (actor sends rollouts to RM, ref returns
logprobs to actor) is **the differentiating fabric pattern** vs
pretrain — it's many small inter-mesh messages rather than a few big
intra-mesh collectives.

## Output artifacts

```
phase7/traces/
├── v11_pretrain_4D_with_TP/        # already done
├── v12_pretrain_4D_no_TP/          # in progress
├── 5D_fabric_modeB_pretrain/       # MODE=B trace
├── 9A_sft_4D/                      # phase 9-A SFT trace
└── 9B_ppo_smoke/                   # phase 9-B PPO trace
```

Each dir has `collective_summary.csv.gz`, `flows.csv.gz`,
`ixia_config.json`, `recipe.json`.

## Catalog update

After all 5 traces collected, regenerate `phase7/pattern_catalog.md`
with the full grid: pretraining vs SFT vs PPO patterns side by side.
