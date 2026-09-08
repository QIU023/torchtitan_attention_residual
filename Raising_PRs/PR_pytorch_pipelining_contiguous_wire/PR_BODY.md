# PR title: [pipelining] exchange activations and gradients as contiguous tensors

--- PASTE BEGIN ---

### Summary

Before this change a pipeline stage sized its P2P receive buffers from the peer's tensor strides (`_TensorMeta.from_tensor` records `tensor.stride()`, `_make_tensor_from_meta` allocates `empty_strided` with it) and sent activations and input gradients with whatever layout autograd produced. A stage whose forward starts with a view op (`cat`, `stack`, a slice of its input) hands back a strided input gradient, the previous stage allocates a non-dense receive buffer, and the batched P2P raises `Tensors must be contiguous`. After this change the wire format is contiguous: metadata records contiguous strides, receive buffers are `torch.empty(shape)`, and senders pass `.contiguous()` (a copy only when the tensor is not already contiguous).

### Changed files

    torch/distributed/pipelining/
      _utils.py    metadata strides are the contiguous ones; dense receive buffers; `_contiguous_strides` helper
      stage.py     `get_fwd_send_ops` and `get_bwd_send_ops` send contiguous tensors

### Test

Two-stage GPipe on two ranks where stage 1 is `Linear(2d, 1)(cat([x, ones_like(x)], -1))`: fails before, runs after, and the pipelined stage-0 weight gradient is bitwise the single-process gradient (the repro and the check are in the linked issue; a unit test in `test/distributed/pipelining/` follows the same shape).

--- PASTE END ---
