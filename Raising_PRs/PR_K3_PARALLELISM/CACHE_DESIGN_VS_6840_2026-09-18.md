# Explaining 4312's cache design against Megatron PR 6840 (2026-09-18)

For a maintainer who reads 6840 and wonders why 4312 routes gradients differently. Source references are from `torch/distributed/pipelining/_backward.py` as of torch `2.15.0.dev20260906+cu130`; the function names are stable even if line numbers move.

--- PASTE BEGIN ---

Both designs face the same problem and solve it in the only way their framework allows, so the difference is not a design preference.

## The problem

A block committed on one rank is read again by later stages on that rank. The store holds it detached, so autograd cannot carry a gradient back to the graph that produced it. Something has to move that gradient explicitly.

6840 does it inside autograd: `_AttnResGradTap` sits in the producing chunk's graph and adds `leaf.grad`. 4312 does it in the stage: `backward_one_chunk` splits the stack's gradient into the columns that travel back on the wire and deposits for the columns held on the rank.

## Why the `leaf.grad` form cannot be used here

`torch.distributed.pipelining` offers two backward paths and neither leaves a usable `.grad` on a leaf outside the stage inputs.

The full path, `stage_backward`, calls `torch.autograd.backward` and then walks the stage inputs:

    for val in input_values:
        if isinstance(val, torch.Tensor):
            grad_inputs.append(val.grad)
            val.grad = None

It reads only `input_values` and clears each one immediately. The comment above that line gives the reason: the gradient must return to the allocator rather than persist in GPU memory for the whole of `step_microbatches`.

The split path used by the zero bubble family, `stage_backward_input`, never writes `.grad` at all. It calls `_autograd_grad_for_inputs`, which is `torch.autograd.grad(outputs=..., inputs=inputs_requiring_grad, ...)` over the stage inputs alone, and then assigns the results only to those same inputs before detaching the output side graph.

So a leaf held in a rank local store is not a gradient target on either path, and anything written to its `.grad` on the full path would be cleared before the stage could read it. The explicit deposit is not a workaround, it is the only seam the framework exposes.

The reverse is equally true and worth saying, because it explains why 6840 did not choose the stage level form. A torch pipelining stage is an object with overridable forward and backward entry points and an explicit `bwd_cache`. Megatron's schedules are functions over model chunks with a single `backward_step`, so an autograd Function inside the model is the only place a tap can live there.

## What `leaf.grad` would and would not buy

These are two independent questions and it is easy to run them together.

The copy in `assemble_stack` exists because the model's contract is a single `[T, N, D]` tensor, so stored columns and received columns have to be brought into one buffer. Handing the layers a list of references removes that copy, and that is what 6840 does. It is a property of the model interface, not of the gradient mechanism.

The gradient hand back exists because the store is detached. `leaf.grad` and deposits are two answers to that, and neither removes a copy by itself.

Taking the list form would cost one copy per receiving stage per micro-batch, which at the 93 layer model with block size 12 under pp8 with vp4 is 28.0 MiB for a stage holding one block and 224.0 MiB for a stage holding eight. Against that, 6840 pads every hop to a uniform payload width because the interleaved schedule carries one `tensor_shape`, which its own helper puts at 12.0% to 27.1% more payload slices depending on the shape, while 4312 sends the exact width per hop. The net is not obviously in favour of either.

## What the stage level form gives that the tap does not

Routing comes from a one micro-batch simulation of the schedule, so it accepts any stage to rank map and any split. 6840 derives the delta in closed form assuming stage `s` runs on rank `s % P`.

The cache lives outside the model, so the model knows nothing about pipelining and activation checkpointing inside a stage never re-runs cache work. 6840 taps inside the layer loop and full recompute is not wired there.

Each stage splits its input stack's gradient once. 6840 adds one autograd node and one `leaf.grad` buffer per tapped source, received sources included.

The deposits are counted. A slot must collect exactly the number of deposits the routing tables predict, none may be left when the rank's first stage finishes backward, and a payload carrying the stage's own blocks must receive a gradient. 6840 has no such count, so a consumer chunk whose backward never ran contributes nothing and the run continues silently.

--- PASTE END ---
