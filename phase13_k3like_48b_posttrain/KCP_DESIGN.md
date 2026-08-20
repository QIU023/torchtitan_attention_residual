# KCP:递推与卷积的两种跨 rank 依赖

从 `kimi_k3/kcp.py` 的模块 docstring 搬出。原文 36 行。内容未改。

decomposes each rank's segment into two locally computable fragments, a
cumulative transition and a zero-started state, which compose associatively; a
prefix scan over them recovers each rank's true incoming state in one
fixed-size all-gather, independent of sequence length. fla-core 0.5.1 ships this
(``chunk_kda(cp_context=...)``) and this repo validated it bit-exact, forward
and backward, against a single-rank reference.

The short convolution. KDA runs a causal depthwise conv of width
``W = short_conv_kernel_size`` on q, k and v before the recurrence. Shard the
sequence and each rank's first ``W - 1`` outputs get computed against zero
padding instead of the previous rank's tail. fla ships this too:
``causal_conv1d_cp`` is a real ``autograd.Function`` that exchanges the tail in
the forward and the matching ``dx`` in the backward. Pass
``conv1d_kernel_size`` to ``build_cp_context`` and it is wired for you.

A hand-rolled halo used to do this with ``dist.all_gather``, which is not
autograd-aware, so the gradient owed to the left neighbour's tail was dropped while
the forward stayed bit-exact -- hence using fla's autograd.Function instead.

KCP vs the Ulysses path also in this repo: Ulysses all-to-alls the head axis and
gives every rank the full sequence for its head subset, so activation memory per
rank stays O(T/cp x D) only for the projections and the recurrence sees the whole
sequence. KCP keeps the sequence sharded end to end, which is what makes it
composable with a sharded-sequence pipeline and what the 1M-token context needs.

KCP is therefore what ``kda_cp_mode`` defaults to, and Ulysses is the A/B. Which
one runs is per-layer-kind and not a whole-model choice: a CP run is KCP on the
KDA layers AND Ulysses on the MLA layers, together, because KCP decomposes the
delta-rule recurrence and has nothing to say about softmax attention.
