# Issue draft for pytorch/pytorch: FSDP2 `post_backward` with gradient reduction off reads `_unsharded_param` of parameters that were never all-gathered

Found 2026-09-07 on torchtitan (Kimi K3, pipeline parallel over FSDP2), torch 2.14.0.dev20260802+cu130; the code is unchanged on the 2.15.0.dev20260906 nightly. Paste as a GitHub issue; one line per paragraph.

--- PASTE BEGIN ---

**Summary.** `FSDPParamGroup.post_backward` has two branches. The reduce branch skips parameters without `_unsharded_param` (`if not hasattr(fsdp_param, "_unsharded_param"): continue`); the no-reduce branch (`reduce_grads == False`) calls `fsdp_param.to_accumulated_grad_if_needed()` for every parameter, which reads `self._unsharded_param.grad` and raises `AttributeError: 'FSDPParam' object has no attribute '_unsharded_param'` for a parameter whose group never ran `unshard()`.

**How it is reached.** `FSDPState._root_post_backward_final_callback` runs `post_backward()` for every parameter group whose state is not `POST_BACKWARD`, including groups that never ran forward ("Run post-backward in case forward inputs did not require gradient so the autograd backward did not run"). Under `torch.distributed.pipelining`, `backward_maybe_with_nosync` turns gradient reduction off for the non-last micro-batches, so those groups take the no-reduce branch. An FSDP-wrapped submodule the batch does not reach (a vision tower on a text-only batch, in our case `fully_shard`-wrapped blocks of the encoder) therefore has no `_unsharded_param`, and the first backward of a non-last micro-batch raises. The same model passes without pipeline parallelism (the reduce branch is guarded) and passes with pipeline parallelism whenever the tower runs.

**Repro shape.** A model with two FSDP2-wrapped submodules where only one is used by the forward of a given batch; `fully_shard` both; run one micro-batch of `torch.distributed.pipelining` `Schedule1F1B` with two or more micro-batches (so `set_requires_gradient_sync(False)` is applied to the non-last ones) and a mixed-precision policy with `reduce_dtype=torch.float32`; the first backward raises. Names of the parameters in our run (all under the unused encoder): `vision_encoder.layers.0._checkpoint_wrapped_module.attn.wq.weight`, `...wk.weight`, `...wv.weight`, `...proj.weight`, `...mlp.linear_fc1.weight`, `...mlp.linear_fc2.weight`, `...norm1.weight`, `vision_encoder.final_norm.weight` (64 in total).

**Suggested fix.** Mirror the reduce branch's guard in the no-reduce branch, i.e. in `post_backward`:

```python
if not self.reduce_grads:
    if self.reshard_after_backward:
        self.reshard()
    for fsdp_param in self.fsdp_params:
        if not hasattr(fsdp_param, "_unsharded_param"):
            continue
        fsdp_param.to_accumulated_grad_if_needed()
    return
```

or make `to_accumulated_grad_if_needed` return early when the unsharded parameter was never created. A parameter that was never all-gathered has no gradient to upcast.

**Workaround we run with.** A process-wide guard on `FSDPParam.to_accumulated_grad_if_needed` that skips such parameters (and logs their names), installed by the training engine; on the torchtitan side we avoid the condition by not instantiating the unused submodule.

--- PASTE END ---

Related, already noted in the PP work: `PipelineStage._create_grad_recv_info` allocates backward receive buffers with `torch.empty_strided` from the strides `_backward_metadata_inference` recorded; a stage whose first op on an input is `cat` / `stack` / a slice hands back view gradients and c10d rejects the buffer (`Tensors for P2P must be non-overlapping and dense`). A dense `torch.empty(shape)` receive buffer and `.contiguous()` before send fix it; we carry an identity op with a `.contiguous()` backward on the model side. Worth a second issue.
