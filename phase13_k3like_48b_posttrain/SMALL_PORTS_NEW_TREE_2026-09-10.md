# The small ports onto the new tree (2026-09-10)

Goal items 3 and 5 on `k3_int_20260910`, each its own commit (built in the detached worktree `/tmp/wt_moonep`, cherry-picked onto the branch).

## Empty optimizer container (item 5)

Old tree: `lr_scheduler.py` and `optimizer.py` tolerate a model part with nothing to train (a frozen-only pipeline stage under LoRA). New tree asserted on it in four places. Ported: the param-group matcher raises on an empty match only when the model part has trainable parameters, otherwise warns and skips; the container initialises torch's `Optimizer` with an empty param group (torch rejects an empty list, accepts an empty group) so hooks and `param_groups` exist; the scheduler container accepts zero schedulers, returns an empty state dict for them and restores nothing on load. `tests/unit_tests/cpu/test_lr_scheduler.py` and `test_optimizer_param_groups.py` unchanged and green (29).

## Compile carve-out (item 5)

Old tree: `_disable_dynamo_on_fla_ops` (fla kernels, `ShortConvolution`, `FusedRMSNormGated`, `block_attn_res`, `KimiDeltaAttention.forward`) plus `_apply_compile_kimi_k3` (dynamo knobs, recompile limit 64, per-layer compile). New tree: the KDA kernels are Attention Gym's (`attn_gym.linear.kda.chunk_kda`, `bound_gate`), so none of the fla bindings exist; the attention residual math has no DTensor unwrapping to trip on. Ported as: `KDAKernel.forward` marked `torch.compiler.disable(recursive=True)` once (class-level), and the shared `torchtitan/distributed/compile.py:apply_compile` gains a `fullgraph` argument so the K3 blocks compile with `fullgraph=False`; the shared helper already sets `capture_scalar_outputs` and the checkpoint side-effect flag. `Kimi K3 does not support model compilation yet` is gone.

| cell (33-layer debug model, dp1, 4096 tokens per step, seeded) | loss 1 | loss 3 | peak memory step 1 |
| --- | ---: | ---: | ---: |
| eager | 12.40087 | 7.71281 | 12.34 GiB |
| compiled (`--compile.enable --compile.components model`) | 12.40087 | 7.71281 | 12.34 GiB |

Bitwise over three steps. The first attempt at 4096 tokens shared its GPU with the tower probe and ran out of memory at step 2; alone on the card the compiled run sits at the eager peak. Not measured: throughput (the debug model's steps are seconds either way), TP or PP under compile (llama3's helper path, untested here on K3).

## Kimi K3 DistMuon recipe (item 3)

Old tree: a hand-written `Muon` optimizer with `_muon_heads` tags on parameters (per-head orthogonalisation by row blocks), not sharding-aware, MoE experts silently on its AdamW path. New tree: upstream `DistMuon` with FQN-keyed compute layouts (`ComputeLayout`, `BlockShard`, `Owned`), which is where per-head tagging lives now. Ported as a recipe, `kimi_k3_debugmodel_muon` in `torchtitan/models/kimi_k3/config_registry.py`, mirroring the Kimi K2.5 registry: MLA `wq_b` per query head (`qk_nope + qk_rope`), `wkv_b` per key-value head (`qk_nope + v_head`), KDA `q_proj/k_proj/v_proj` per head (`head_dim`), routed experts per expert (EP/EFSDP layouts rebuilt from the final parallelism in `__post_init__`), every other matrix owned whole (MLA `wq_a/wkv_a/wo/gate`, KDA `output_gate/output_proj/beta/forget_a/forget_b`, the latent projections, shared experts, the router gate, the dense feed-forward); AdamW keeps norms, biases, the convolutions, `A_log`/`dt_bias`, the one-row residual projections, embeddings, the LM head and the vision tower. The per-expert layout helper is imported from the K2.5 registry rather than copied; it belongs in `flex_shard` if a third model needs it. Tensor parallelism is refused, as K2.5 does, until PR 4353 (DistMuon TP storage layouts, open, updated 2026-09-10) lands.

On the 33-layer debug model every Muon-pattern parameter has a compute layout and every layout names a parameter (537 on Muon, 465 on AdamW... see the check in the session: 537 matrices with layouts, the rest AdamW), 34 buckets.

GPU cells: (pending: dp1, fsdp2, ep2 x fsdp2, three steps)
