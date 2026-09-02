# PR title: [Draft] [Kimi K3] tensor parallelism: head-parallel MLA and KDA

Branch `tp_review1` (four commits on `k3_ep` `69f84292d`: TP, SP, the grad-norm mesh grouping, the latent-MoE norm under SP, which is upstream/main `1dcb14a0c` plus the expert parallel PR). Stacked on the EP PR: it adds to the `sharding.py` EP creates. Rewritten on 2026-09-02 for main's attn-gym KDA (`InnerKDA` behind a `local_map`) -- the earlier branch unwrapped DTensors at an fla kernel call that no longer exists. Paste between the markers.

--- PASTE BEGIN ---

### Summary

Adds tensor parallelism to Kimi K3. Before this change `parallelize_kimi_k3` raises `NotImplementedError` when `tensor_parallel_degree > 1`; after it the model is declared through the sharding-config system, structured as qwen3_5's -- `sharding.py` declares, `model.parallelize()` applies -- and both attention kinds are head-parallel. Sequence parallel is not offered here: the stream stays whole on the TP axis and only head and feature axes shard.

### Design

- MLA is head-parallel: `wq_b`, `wkv_b` and the output gate split colwise on the head axis, `wo` rowwise; the two compressions (`wq_a`, `wkv_a`) stay whole because they are rank-sized, not head-sized. The FlexAttention body runs under the shared inner-attention `local_map`, the helper qwen3_5's full-attention layers use.
- KDA is head-parallel too, because the delta rule is independent per head: `q_proj`, `k_proj`, `v_proj`, `forget_b`, `beta` and the output gate split colwise, `output_proj` rowwise, and the per-head state -- `A_log`, `dt_bias`, the three depthwise convolutions -- shards with the heads. The one low-rank compression, `forget_a`, stays whole.
  - The kernel runs on the local heads behind a `local_map` on `inner_kda`: every forward argument is declared (activations feature- or head-sharded, the conv weights and the state head-sharded), `local_map` hands the kernel plain tensors and wraps the output back head-sharded. `cu_seqlens` becomes keyword-only on `InnerKDA.forward` so the positional map covers tensors only; it is metadata and passes through untouched.
  - No unwrap helper and no redundant compute: each rank runs its own heads, which is what makes this TP rather than TP-invariant replication.
- Every norm and residual projection on the block stream is declared replicated. Left undeclared they meet DTensor activations as plain tensors: the block-residual aggregation multiplies a norm weight and a projection weight together, and one declared, one not is a mixed mul.
- The latent MoE pair (`routed_down`, `routed_up`) stays whole -- it compresses to a rank, not heads or experts. The MoE internals keep the declaration the EP PR installs.
- The multimodal splice: under TP the embedding output is a DTensor while the replicated, undeclared vision tower returns a plain tensor, and the splice's `copy_` refuses the mix. The tower output is replicate-consistent across the mesh -- replicated weights, the same pixels on every rank -- so it is lifted with `DTensor.from_local` before the scatter: a wrap, not a transfer.

### Results

Training loss on `kimi_k3_debugmodel`, one seed, warmed compile cache. TP splits heads across ranks and reduces the rowwise outputs, so the bar is agreement rather than bitwise; the tp2 row tracks dp1 the way the earlier fla-based branch's tp2/tp4/tp8 rows tracked theirs (0.01 at step 1).

| cell | world | step 1 | step 3 | step 10 |
|---|---|---|---|---|
| dp1 | 1 | 12.52977 | 7.27107 | 2.98077 |
| tp2 | 2 | 12.55057 | 7.52677 | 3.00361 |
| tp2 + SP | 2 | 12.54164 | 7.44412 | 3.08160 |

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.tensor_parallel_degree 2
```

### Changed files

    torchtitan/models/kimi_k3/
      sharding.py                    +125/-0  the TP declarations: MLA, KDA, norms, residual projections, latent MoE pair
      model.py                       +20/-1  the TP hook in update_from_config; the vision splice wrap
      parallelize.py                 +1/-2  tensor parallel leaves the unsupported list
      kda.py                         +2/-1  cu_seqlens is keyword-only on InnerKDA.forward

### CI/CD Coverage

No GPU cell is added on this branch; the K3 integration lane is B200-only after the CI reorg, and a tp2 cell there is a follow-up once the lane owner agrees. The CPU suite exercises the declarations through the existing K3 and MoE tests.

--- PASTE END ---

Notes for us, not for the body:

- The earlier design (fla kernel, TP-invariant KDA, `dtensor_ops.py`) is retired with the kernel it wrapped. The grad-norm mesh grouping in `distributed/utils.py` came back as its own commit: the first tp2 run tripped `get_total_norm` on two meshes exactly as before (the undeclared vision tower's grads live on the fsdp-only mesh).
- SP is the second commit on this branch; its tp2 + SP row is in Results. Under SP without EP the MoE boundary gathers the sequence shard, the experts hand it back sequence-sharded, and the latent-MoE norm after them runs on the shard (the fourth commit).
