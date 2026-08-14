# Reply to the maintainer's question

He asked:

> sorry I couldn't really understand the PR summary which seems to be written by AI.
>
> Before vs. after this change, what ops that are used to performed in bf16 are not
> performed in fp32?

Two reductions, and nothing else. Post this, and nothing more -- the rest of the body he
already has. See `../PR_WRITING_RULES.md` for why the body failed to say this.

---

## Text to post

Fair -- here it is directly. Two reductions move, both inside `get_total_norm`:

1. The per-tensor norms. `torch._foreach_norm(grads, 2)` returned them in the gradients'
   dtype, so with bf16 gradients each one was accumulated and returned in bf16. Now they
   are requested with `dtype=torch.float32` (and for DTensors, which upstream's foreach
   check excludes, the equivalent `torch.linalg.vector_norm(..., dtype=torch.float32)`).
2. The norm-of-norms -- `vector_norm` over the stack of those per-tensor norms. It has no
   dtype argument here; it is fp32 because its input now is.

Nothing else changes. The gradients themselves stay bf16, the clip multiply is unchanged,
and float32 runs are bit-identical before and after.

The reason it matters is that the second reduction is over `len(grads)` values, so its
error grows with how the parameters were grouped -- under PP or EP that means the total
norm depends on where the model was cut. On llama3_debugmodel at dp_shard=2 x pp=2 the
bf16 total was 1.4453 against 1.4509 in fp32 at step 1.

---

## If he asks why not just cast the gradients

Because that doubles gradient memory for the duration of the clip and changes what gets
clipped; only the reduction accumulators need the range, which is what
`torch.linalg.vector_norm(..., dtype=torch.float32)` gives for free.

## If he asks about cost

The per-tensor norms are already memory-bound reads of the gradients; promoting the
accumulator does not change the read volume. The norm-of-norms is over one scalar per
tensor.
