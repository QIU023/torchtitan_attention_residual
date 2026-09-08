# Issue: [pipelining] a stage whose forward starts with a view op cannot exchange its input gradient (receive buffer sized by non-dense strides)

## Repro

`torchrun --nproc_per_node=2 pp_dense_repro.py` (attached; 2 GPUs, NCCL). Stage 1's module is `Linear(2d, 1)(cat([x, ones_like(x)], -1))`: the gradient of its input `x` is `grad.narrow(-1, 0, d)`, strides `(2d, 1)`, not dense. Stage 0 sizes its gradient receive buffer from that gradient's metadata with `torch.empty_strided(shape, stride)` (`_utils._make_tensor_from_meta`) and the batched P2P rejects it:

```
rank 0: ValueError: Tensors must be contiguous
```

(older builds: `Tensors for P2P must be non-overlapping and dense`). Any stage that begins with `cat` / `stack` / a slice of its input hits this; model code has to wrap the input in an identity autograd function whose backward returns `.contiguous()` to get around it.

## Where

- `torch/distributed/pipelining/_utils.py`: `_TensorMeta.from_tensor` records the tensor's own strides; `_make_tensor_from_meta` allocates `empty_strided` with them.
- `torch/distributed/pipelining/stage.py`: `get_bwd_send_ops` sends the input gradient as produced; `get_fwd_send_ops` sends the output as produced.

## Proposed fix

Make the wire format contiguous: record contiguous strides in the metadata, allocate receive buffers with `torch.empty(shape)`, and send `.contiguous()` tensors (a copy only when the tensor is not already contiguous). Patch attached (`contiguous_wire.patch`, against the 2026-09-06 nightly); with it the repro runs and the pipelined stage-0 weight gradient is bitwise the single-process one (`pp_dense_check.py`).
