# GRPO under parallelism, through veRL's torchtitan engine

## Working

  single GPU        loss -0.053495 -> -0.939649, grad_norm 120.15 / 108.42
  dp_shard=2        grad_norm 120.13 on both ranks

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

- **tp=2** is a checkpoint shape mismatch: 224 x 256 saved against something else
  current. The seed checkpoint was written without TP, so this is the loading
  path, not the model.
- **pp=2** is the GRPO smoke's own fault before it is veRL's: under PP the model
  returns a tuple, and `_policy_gradient_step` calls `.float()` on it. The
  normalized_shape error before it suggests the stage split also needs the input
  contract PP expects.
- **cp=2** is a placement bug -- something calls `.dim` on a `Replicate`, which
  only `Shard` has.
- **the two 4-GPU combinations** share one cause and it is about how veRL calls
  `fully_shard`, so it is upstream of our model code.

## Not claimed

Only single-GPU and dp_shard=2 are verified. The other five configurations do not
run, and no number from them exists to report. The sampler is also still stubbed
throughout -- sequences are deterministic ids rather than model samples -- so
what is verified is the plumbing, not sample quality.
