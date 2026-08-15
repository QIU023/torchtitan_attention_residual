# Swapping our modules for the upstream K3 tree's: what to take and what to keep

The instruction was to replace our modules with the upstream tree's. Having read
both class by class, the answer is not uniform, and the split falls along one line:

**Take their names and their block-level layout. Keep our internals wherever ours
composes a core torchtitan primitive and theirs re-implements it.**

Their tree supports FSDP2 only -- its `parallelize.py` raises `NotImplementedError`
for both TP and EP -- so several of their internals are simpler because they never
had to carry a parallelism. Adopting those would cost us capability.

## Taken

| what | slice | status |
| --- | --- | --- |
| `attn_res_proj/norm/alpha` -> `attention_res_*`, `mlp_res_*` -> `ffn_res_*`, `final_attn_res_*` -> `output_res_*` | A | 54/54 |
| one `self_attn` -> `attention` XOR `delta_attention` | B | 54/54 pending, suite green |
| `ffn` -> `moe` XOR `feed_forward` | C1 | next |

The naming argument stands on its own, independent of convergence: titan's own
convention is `attention` / `feed_forward` internally with `self_attn` / `mlp` on
the HF side, which deepseek_v3 follows too. Their spelling is the idiomatic one and
ours was the HF one.

## Kept, with reasons

**The SiTU grouped experts.** Same math. Ours calls `self._grouped_mm(...)`, the
inherited seam; theirs calls `torch._grouped_mm(...)` directly. The MXFP8 converter
installs its quantized GEMM by overriding that seam, so their form bypasses
quantization entirely. Keeping ours.

**The gated output norm.** Ours is fla's `FusedRMSNormGated`, a fused Triton
kernel; theirs is the same math in eager PyTorch. Theirs is declarable (a
torchtitan `Module`, so it can hold a `ShardingConfig`, which fla's cannot) and
needs no shim -- a real advantage on their side -- but swapping a fused kernel for
eager code on every KDA layer's critical path is a performance regression we should
not take silently.

**The latent MoE's internals.** This is the load-bearing one. Ours is
`KimiMoE`, which composes core's `MoE` as `self._moe` and wraps it with the latent
projections; theirs is a flat `KimiLatentMoE` that holds the router, dispatcher and
experts directly and re-implements the MoE forward.

Our entire expert-parallel support is `_moe.parallelize(parallel_dims)` -- core's
`MoE` distributes its `GroupedExperts` states over the `ep` mesh and wires the token
dispatcher's ep/tp meshes for all-to-all dispatch and combine. That is 16 lines on
our side because the machinery is core's. Their class holds a plain
`LocalTokenDispatcher` and has no EP path at all, and their `parallelize.py` refuses
EP, so there is no reference implementation to copy. Flattening would mean writing
EP dispatch wiring from scratch to replace machinery we currently get for free.

It is also what CLAUDE.md's third core principle asks for: reuse over duplication.
Their flat form is not worse engineering for a tree that only runs FSDP2; it is
just not a tree that has to shard experts.

## The pattern worth remembering

Three of the four "keep ours" decisions have the same shape: their version is
simpler because it does not carry a parallelism or a quantization seam that ours
does. Their version is not wrong -- it is scoped to what their tree supports. So
the check for each remaining class is not "which reads better" but "does theirs
still work under TP, EP and MXFP4".

Where ours is better and the difference is small, it is worth proposing upstream:
the `_grouped_mm` seam is one line at three call sites and would let their experts
quantize.
