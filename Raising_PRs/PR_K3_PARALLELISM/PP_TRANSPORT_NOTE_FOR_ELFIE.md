# Kimi K3 pipeline parallelism on the new tree: the stage design, your transport fix on it, and the two-node test

For Elfie. Everything below is on `QIU023:pp_review4`, which is PR 4312's branch (`k3_pp_text`, main after the K3 EP merge) plus three commits: the edge-communicator warm-up, your transport isolation composed onto the new stage class, and its tests.

## 1. What changed between the old tree's pipeline and PR 4312's

The old tree (PR 4281, the branch you tested) ran the attention residual through a 1228-line adapter around `torch.distributed.pipelining`: it wrapped `forward_one_chunk`, `backward_one_chunk` and `step`, kept a thread-local micro-batch id, bridged the same-rank gradient path with a tensor grad hook, and, because the block stack a hop carries depends on the stage, let the runtime run in DYNAMIC mode: every micro-batch exchanged metadata objects (`_send_meta` / `_recv_meta`, `send_object_list` on the PP NCCL group) between the schedule's tensor P2P batches, and the inference-mode vote was a serial P2P chain on the same group.

PR 4312 replaces that with one `PipelineStage` subclass, `AttnResPipelineStage` (`torchtitan/models/kimi_k3/pipeline_stage.py`), and a routing table built once from the layer layout:

- No metadata on the wire. Sender and receiver derive what a hop carries from the same table before the first send, so every hop has a fixed payload and the stage runs STATIC. The only object P2P left is the runtime's own one-time shape inference at initialization.
- The block a stage commits rides the schedule's own forward P2P to the next stage, hop by hop (adjacent stages only, for exactly $P-1$ hops), and its gradient rides the schedule's own backward P2P back. No collective and no P2P of our own, in forward or backward.
- Blocks a rank already holds come from a per-rank store keyed by the schedule's micro-batch id, as values, not graphs; the gradient for a same-rank consumer is deposited in a slot and picked up by the producer's backward, which every schedule orders after the consumers' on that rank.
- `pipeline_llm(..., stage_class=...)` is the one core hook; the model owns the rest.

So the traffic pattern of a K3 pipeline on the new tree is the traffic pattern of any dense model under `torch.distributed.pipelining`: batched tensor P2P between adjacent stages, nothing else.

## 2. Your diagnosis, read against the runtime

`torch/distributed/pipelining/schedules.py` (2.15 nightlies) carries a TODO for the "STATIC mode group communicator warm-up gap": the vote protocol warms the two-rank sub-communicators the homogeneous `_batch_p2p` path uses, but the group communicator behind the mixed send/receive batch is created lazily at the first 1F1B steady-state batch; DYNAMIC-mode metadata inference happens to warm it, STATIC mode does not. The fix it prescribes is `_get_init_p2p_neighbors_ops` + `_batch_p2p` after the vote.

The old tree's hang was one layer deeper than that: per-micro-batch object P2P and the vote chain shared the group communicator with the tensor batches, and under 1F1B the neighbouring ranks reach those operations in different orders. That is why `d1ec535d1` (the warm-up, already in the branch you tested) did not unblock it: it fixed creation timing, and the old tree's problem was ordering. Your isolation removes the ordering coupling by construction: metadata on a CPU group, one two-rank NCCL group per edge, one collective for the vote.

On the new tree the ordering problem does not exist (no traffic outside the schedule's batches), and what remains is exactly the TODO's gap. `pp_review4` therefore carries both remedies, so the two-node run can tell them apart:

- default: `_warmup_pp_edge_communicators` right after the schedule build (every rank enters it at the same point, dummy payloads, communicators up before step one);
- `TORCHTITAN_PIPELINE_NEIGHBOR_P2P=1`: your transport, as a mixin composed in front of `AttnResPipelineStage` (`torchtitan/distributed/pipeline_parallel.py`: `_NeighborP2PTransportMixin`, `_create_pipeline_transport_groups`, `_configure_neighbor_p2p_schedule`; `ParallelDims._create_pipeline_neighbor_groups`; the `device_id` binding in `init_distributed`). Two changes from your branch: edge groups are keyed by the sorted global-rank pair and the peer index is looked up per edge, so looped schedules (Interleaved1F1B, the stage `k*P-1 -> k*P` wrap from the last rank to the first) get their group too; and the stage subclass is composed rather than replaced, so the residual routing keeps running on top of your transport.

## 3. What is verified on one node (8 x RTX 5060 Ti, SM120, Attention Gym's Triton KDA path)

- Warm-up: dp1 and pp2 five steps bitwise with and without it; the `kimi_k3_debugmodel_pp8_vp4` recipe (Interleaved1F1B, 8 ranks x 4 virtual stages) bitwise with and without it on one inductor cache (a second cache moves step 2 by 0.2 percent on this box, the autotune lottery, not the code).
- Transport on: see the line appended below once the runs finish (pp2 five steps, pp8 plain 1F1B three steps, the pp8 x vp4 recipe).
- Checkpoint resume on the new tree (dp1 and pp2, save at step 2, resume, re-run steps 3-4): the step-3 loss and grad norm are bitwise with the continuous run, the step-4 loss is not (`8.59245` vs `8.59860` at dp1). Every parameter has Adam state in the checkpoint (1002 of 1002; the 32 tensors without state are the `expert_bias_E` buffers), so this is not the missing-state case you hit on the old tree; what is not restored exactly is being located (learning-rate schedule, data position, or the update itself).

## 4. The ask

1. PP=8 across your two GB200 nodes on `QIU023:pp_review4`, twice: default, and with `TORCHTITAN_PIPELINE_NEIGHBOR_P2P=1`. Plain 1F1B first (the configuration that hung), then `torchtitan_recipes.tests.features:kimi_k3_debugmodel_pp8_vp4`. Two or three steps are enough; what matters is whether the late edges hang, and the step-1 loss of each run (they should agree).
2. The optimizer-state fix (`344fccf17`) is independent of the tree: torch's `_init_optim_state` skips every parameter once any state exists. The new tree's layer 0 has no residual projection, so it does not hit it today, but an unused parameter under PP, LoRA or MTP would. Please open it as a standalone PR against main with your test.
3. The old tree (PR 4281) is frozen. The sequence on main: PR 4527 (Shuhua's multimodal spmd declarations, the base), PR 4499 TP/SP (`k3_tp_sp`), PR 4412 Quantile Balancing (`k3_qb`), PR 4312 PP text (`k3_pp_text`), PR 4500 CP (fegin's stack); multimodal PP and CP follow as PR 4381 and PR 4380 once the text sides land.

## 5. Reproduction (one node, for reference)

```
COMMON="-m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --checkpoint.enable --parallelism.data_parallel_shard_degree 1 --metrics.log_freq 1"
torchrun --nproc_per_node=1 $COMMON --training.steps 1 --checkpoint.create_seed_checkpoint --dump-folder seed
# pp8, plain 1F1B, eight 256-token micro-batches
TORCHTITAN_PIPELINE_NEIGHBOR_P2P=1 torchrun --nproc_per_node=8 $COMMON --training.steps 3 --training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.pipeline_parallel_degree 8 --parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule 1F1B --dump-folder pp8
# pp8 x vp4 recipe
torchrun --nproc_per_node=8 -m torchtitan.train --module torchtitan_recipes.tests.features --config kimi_k3_debugmodel_pp8_vp4 --debug.seed 42 --debug.deterministic --training.steps 3 --metrics.log_freq 1 --dump-folder vp4
```
