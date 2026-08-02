# GRPO under parallelism, through veRL's torchtitan engine

## Working

  single GPU        grad_norm 120.15 / 109.23
  dp_shard=2        grad_norm 120.15 / 110.23
  cp=2              grad_norm 219.68 / 197.63

The loss moving between steps is what makes this a real result: the GRPO
objective reaches the actor's parameters through veRL's engine and the update
takes effect. A zero-gradient assertion guards the run, because an earlier
version passed while computing nothing.

dp_shard=2 needed one fix in our code. K3's `apply_fsdp` asks for a mesh axis
named `"fsdp"`, which torchtitan builds (= dp_shard x cp) but veRL does not --
its axes are `['pp','batch','loss','dp_replicate','cp','tp','ep','efsdp','dp',
'dp_shard']`. The fallback composes the same product from the axes veRL does
have, so the semantics are preserved rather than narrowed to dp_shard. Native
torchtitan FSDP is unaffected (verified: 7.68238 / 7.16806 at dp_shard=2).

## Failing, five distinct causes

| config | error |
|---|---|
| tp=2 | `Size mismatch between saved torch.Size([224, 256]) and current` |
| pp=2 | `Given normalized_shape=[512], expected input with shape [*512]`, then `'tuple' object has no attribute 'float'` |
| cp=2 | `'Replicate' object has no attribute 'dim'` |
| dp_shard=2 x tp=2 | `spmd_types parameters require fully_shard() to be called with...` |
| dp_shard=2 x ep=2 | same as above |

These are five different problems, not one. Reading them:

- **tp=2** has TWO independent blockers, and separating them took disabling the
  checkpoint load (engine.initialize() calls checkpointer.load() unconditionally,
  with no config switch).

  The first is a checkpoint layout limitation, not a model bug. The expert weight
  is saved as [224, 256] and the TP-sharded model wants [112, 256] -- exactly
  half. `_get_local_experts_weights` documents itself as handling "FSDP + EP"
  (dim-0 sharding) and has no path for TP sharding experts on dim-1. Extending
  it changes checkpoint semantics for every MoE model, so it is left alone
  here.

  The second only becomes visible once the load is skipped: `aten.mm.default got
  mixed torch.Tensor and DTensor`. That one is this smoke's, not the model's --
  it calls `module(ids)` with a plain tensor while the TP-wrapped model expects
  a DTensor input. The engine's `forward_backward_batch` would supply it
  correctly, which is the same entry the PP path needs, so both TP and PP wait
  on the same piece of work rather than two.
- **pp=2** was two problems, one of them mine. The tuple return is fixed (under
  PP the stage returns AttnRes block tensors alongside the hidden states, so the
  logits are element 0). The remainder is structural: this smoke calls the module
  directly, and under PP the stages must be driven by the pipeline schedule --
  a direct call hands stage 1 the raw token ids instead of stage 0's hidden
  states, which is the `normalized_shape=[512] ... got input of size [1, 8, 128]`.

  `engine.forward_backward_batch` is the right entry, since it owns the
  schedule, micro-batching and the SPMD mesh context. Four attempts at its
  TensorDict contract did not land: `max_token_len_per_gpu` is read off the
  batch rather than the engine config (verl/workers/engine/utils.py:68-74), and
  supplying it via `assign_non_tensor_data` then fails with "values expected
  sparse tensor layout but got Strided". Rather than keep guessing, the direct
  path stays as the verified one and the smoke raises NotImplementedError under
  PP instead of silently measuring the wrong thing.
- **cp=2** is FIXED. `_caculate_indices_from_placements` (models/utils.py) reads
  `.dim` off every placement in the tuple, but only `Shard` has one. Under CP the
  expert weights are replicated on the cp axis, so the checkpoint load died
  before training started. A mesh axis the tensor is replicated over contributes
  no sharding of dim-i, so it is now skipped rather than probed. Not
  K3-specific: any MoE model with a replicated axis reaches it. Native
  torchtitan paths re-verified bit-identical afterwards.
- **the two 4-GPU combinations** are partly diagnosed and NOT fixed. veRL sets
  `spmd_backend="spmd_types"`, and under that backend `fully_shard` must be told
  which SPMD mesh axes are data-parallel via `dp_mesh_dims`. Upstream llama3 and
  deepseek_v3 resolve both mesh and dims through `resolve_fsdp_mesh` /
  `resolve_sparse_fsdp_mesh`; K3 passed only the mesh, which is why it worked
  under torchtitan's default backend and failed under veRL.

  Supplying the dims moves the failure one layer on, to "Expected param's
  DTensor mesh to be the same mesh passed to fully_shard" -- TP has already
  placed the parameters on the tp mesh while FSDP receives the resolved full
  SPMD mesh. That is the two mesh systems meeting, and three attempts did not
  land it. The work is stashed rather than committed half-done.

## Not claimed

Only single-GPU and dp_shard=2 are verified. The other five configurations do not
run, and no number from them exists to report. The sampler is also still stubbed
throughout -- sequences are deterministic ids rather than model samples -- so
what is verified is the plumbing, not sample quality.


## TP under veRL is the same problem the k3_refactor branch found

Peeling the layers off tp=2 one at a time -- skip the checkpoint load, set
`activation_checkpoint="none"` (veRL's own config comment recommends this under
spmd_types), disable torch.compile -- leaves this:

    File "torchtitan/protocols/module.py", line 291, in forward_with_redistribution
    File "torch/nn/modules/linear.py", line 134, in forward
    RuntimeError: aten.mm.default got mixed torch.Tensor and DTensor

`forward_with_redistribution` is the DECLARATIVE path. veRL sets
`spmd_backend="spmd_types"`, which makes `Module.parallelize()` the mechanism,
and that is exactly the wall the k3_refactor branch hit: the declarative
vocabulary has no `use_local_output`, so it cannot express K3's plain-tensor
module boundaries, and the boundaries exist because PP P2P sends raw tensors,
block_attn_res stacks plain and DTensor operands, and fla's triton kernels
dispatch on data pointers.

So TP-under-veRL and the imperative-to-declarative migration are ONE problem,
not two. It is not fixable in the smoke or the engine config: either the
declarative vocabulary gains a way to say "hand the next module a plain tensor"
(a core change, belongs upstream), or K3 gives up the boundary convention (which
costs the three things above).

The three layers peeled off on the way are worth keeping as findings in their
own right, since each was masking the next: the unconditional checkpoint load,
activation checkpointing recomputing the forward outside the DTensor context,
and torch.compile turning the error into a fake-tensor failure.
