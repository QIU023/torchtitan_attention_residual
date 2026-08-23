# PR-EP 正文(PR-text 规则)

--- PASTE BEGIN ---

Adds expert parallel to Kimi K3. ep2 trains on both debug flavors and combines with cp2.

Most of it is wiring core already has: the sparse FSDP mesh for the routed experts, ep_degree, and the model.parallelize call deepseek_v3 makes. Two things were missing.

The model declared no sharding at all, so model.parallelize had nothing to act on and left all 680 parameters plain tensors. The declarations now come from core's helpers -- set_decoder_sharding_config for the root and set_moe_sharding_config for each MoE -- rather than restating their values here. Only the parts core already has a helper for: an undeclared module is inert, a wrongly declared one is a silent numerics change.

Then the token dispatcher. The config hard-codes LocalTokenDispatcher, which only reorders tokens within a rank, so it hands the experts the global per-expert counts. Once the expert weights are sharded on E that meets the grouped GEMM as "matrix batch sizes have to match", which reads as a shape bug in the model and is not one. Under EP the dispatcher is now AllToAllTokenDispatcher; the two configs carry the same fields. Measured after the swap: 16 local experts against 16 offsets, and the two ranks receive 1030 and 1018 tokens, so tokens really are crossing. Step 1 loss is 12.38712, the same as the dp2 baseline to all five digits, which is what expert parallel should do to a forward.

Sequence parallel is deliberately not offered. The sequence is already the axis context parallel shards here, and this model's CP is not ShardingConfig-driven, so the two would be describing one axis from two places.

    torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 \
      --config kimi_k3_debugmodel --parallelism.expert_parallel_degree 2 \
      --parallelism.data_parallel_shard_degree 2

--- PASTE END ---

## 不进正文的支撑

| 声称 | 证据 |
|---|---|
| 之前完全没生效 | `model.parallelize` 后 680 个参数全是 plain |
| dispatcher 真在跨 rank 分发 | 16 本地专家 / 16 offsets;两 rank 收到 1030 与 1018 |
| EP 不改变前向 | step-1 loss 12.38712,与 dp2 基线五位全同 |
| 组合 | ep2 x cp2 四卡 3 步通过 |

## 未包含

MoonEP(`moon_ep_dispatcher.py` 未搬)。
