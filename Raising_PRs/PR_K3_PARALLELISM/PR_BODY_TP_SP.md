Enables tensor parallelism for Kimi K3, with SequenceParallel on the same
mesh. Before this change the model declares no TP sharding and
tensor_parallel_degree > 1 is rejected; after it the config tree carries the
declarations and TP comes off the unsupported list, with
parallelism.enable_sequence_parallel (core default: on) deciding whether the
token stream between modules is Replicate or the TP-axis Shard(0).

The declarations live in kimi_k3/sharding.py following qwen3_5/sharding.py:
declarations only, applied through the Module protocol, nothing touches a
mesh or a device. MLA shards on the head axis (wq_b/wkv_b/gate colwise, wo
rowwise, the two compressions replicated: they are rank-sized, not
head-sized); KDA is TP-invariant and declared so -- its kernels never see a
DTensor, and the declaration keeps every parameter on one mesh so
clip_grad_norm_ can stack them. Norms and the attention-residual
projections sit on the block stream, which TP does not split. Under SP the
norms compute on the sequence shard, the MLA and KDA module boundaries
gather it on the way in (the attention cores consume the whole sequence;
KDA's delta recurrence slices back to the shard on the way out, so SP's
activation saving does not extend into KDA bodies -- that is the
recurrence, not a choice), and the rowwise outputs reduce-scatter back:
the llama3 template. The multimodal splice is a token-indexed seam and
redistributes to the full stream before scattering the tower's features:
vision_positions index the global token axis, and slicing the mask per tp
rank instead would hand the replicated tower disjoint partial gradients
that nothing reduces over the tp axis.

Training loss on kimi_k3_debugmodel, one seed, warmed compile cache
(steps 1/3/10):

| config | step 1 | step 3 | step 10 |
|---|---|---|---|
| TBD: dp1 | | | |
| TBD: tp2 (SP on) | | | |
| TBD: tp2 (SP off) | | | |
| TBD: fsdp2 x tp2 | | | |

Changed files: torchtitan/models/kimi_k3/sharding.py (new),
torchtitan/models/kimi_k3/model.py, torchtitan/distributed/ (grad-norm
grouping by parameter mesh), tests.
