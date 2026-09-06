# The four MLA CP kernels at cp2: same forward, backwards within bf16 rounding (2026-09-06)

Why the cp2 rows of the CP body's dp = 1 table share step 1 (12.53972) and differ by percents at
steps 3 / 10. Step-1 gradients of every parameter (rank 0, own dtype, before clipping) under the four
kernels, same seed checkpoint and batch as the CP matrix, Attention Gym b19162e, one compile cache,
on `61a73ca6c` + the dump-and-exit hack (`cp_kernel_grad_dump.sh`, worktree `/tmp/wt_cpdump`).

| pair | parameters bitwise | sign flips | elementwise rel diff (median) | per-param norm rel diff (median / 90th) |
| --- | --- | --- | --- | --- |
| packed Ulysses vs generic Ulysses (4450) | 24 / 750 | 0.202% | 1.0e-2 | 1.6e-4 / 1.4e-3 |
| packed all-gather KV vs generic all-gather KV (4322) | 24 / 750 | 0.136% | 7.2e-3 | 1.1e-4 / 9.9e-4 |
| packed Ulysses vs packed all-gather KV | 22 / 750 | 0.204% | 9.9e-3 | 1.6e-4 / 1.3e-3 |
| generic Ulysses vs generic all-gather KV | 22 / 750 | 0.139% | 7.3e-3 | 1.2e-4 / 1.1e-3 |

Reading: the forward is identical (the loss), the backwards add the shared rope slice's gradient in a
different order (one reduce-scatter of the slice against per-head copies through the all-to-all or
gather), a last-bit bf16 difference at the MLA layers that this model's backward amplifies on the
way to layer 0 (the largest relative differences sit on layer 0's residual-norm and KDA parameters,
3e-2 to 8e-2); the per-parameter norms still agree to 1e-4 and 0.1-0.2% of the elements change
sign, the same order as a pipeline cut (`PP_STEP10_SPREAD`: 0.22%). The two upstream kernels are
as far from each other as the packed ones from them, so the packing adds no numerical distance
beyond what choosing an algorithm does. The steps 3 / 10 spread of the table's rows is that plus
the per-invocation compile caches (the dp1 floor: 1.3% / 2.4%).

Precision note: the body's earlier "median 1.8e-4" for cp8 was a per-parameter NORM figure
(`grad_hash_hack`, norms only); the elementwise level is 1e-2. The caption now states both.

## Located (`cp_kernel_attn_grad_probe.sh`): the rope slice's gradient, delivered two ways

Full backward hooks on the MLA inner kernels of layers 23 (the first the backward meets; nothing
downstream holds a CP kernel) and 3, first micro-batch of step 1, both ranks, packed against generic
Ulysses:

| point | rank 0 | rank 1 |
| --- | --- | --- |
| layer 23, gradient arriving at the kernel (d out) | bitwise | bitwise |
| layer 23, d q and d v | bitwise | bitwise |
| layer 23, d k nope columns | bitwise | bitwise |
| layer 23, d k rope columns | packed: on head 0 only (summed inside its collective); generic: on all 16 heads | same |
| layer 23, rope gradient after the model's `expand` backward sums the heads | rel 2.4e-3; 87% of elements within 1 bf16 ulp, 97% within 4 | rel 2.5e-3; 86% / 97% |
| layer 3, d out | rel 1.7e-2 | rel 1.7e-2 |

So the two kernels part at exactly one place, the addition order of the shared rope slice's
gradient (a last-bit bf16 difference, 1 to 4 ulp), and the backward of the 20 layers in between
amplifies it to 1.7e-2 by layer 3, 1e-2 elementwise on the parameters with per-parameter norms
still within 1e-4. Nothing else differs.
