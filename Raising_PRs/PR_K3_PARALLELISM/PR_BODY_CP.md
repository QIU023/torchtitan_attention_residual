Adds context parallelism to the Kimi K3 text decoder. The two attention kinds need different mechanisms, so both are here: Ulysses on the Gated MLA layers, and the report's KCP on the KDA layers -- a conv halo exchange plus a prefix scan over the recurrent state, nothing softmax-shaped. As far as I can tell this is the first Ulysses in torchtitan; if you would rather it live under distributed/ as a model-agnostic piece, say so and I will move it.

Ulysses is one all-to-all trading the sharded axis from sequence to heads, the unchanged attention backend, and a second all-to-all trading back. The rotary key slice stays outside the exchange: it is headless -- one vector per token shared by every head -- so it is all-gathered along the sequence and expanded onto local heads afterwards; packing the already-expanded key sends the same values once per head and reassembles them against the wrong head subset. The local head count is read off the projection width rather than n_heads, and k_rope crosses a Partial-gradient boundary before the gather, so a later column-parallel wq_b/wkv_b changes neither.

KCP cannot be declarative: the fla kernels take raw pointers and never dispatch through DTensor, so no ShardingConfig reaches them. Hence one core change, Decoder.Config.cp_via_sharding_config (default True) -- validate_cp_backend's docstring says it is for models declaring CP in a ShardingConfig, but it ran for every decoder; a model implementing CP itself now says so and takes on its own preconditions. Declarative models are unchanged.

CP resplits the sequence, so unlike PP and EP it is not expected to be bit-identical to dp1. kimi_k3_debugmodel_text at seq 1024 (FlexAttention's BlockMask needs Q_LEN % (cp * 128) == 0), seed 42, --debug.deterministic, one seed checkpoint loaded by every cell:

<<TABLE_CPSEQ>>

Same, on a 2D mesh with data parallel, seq 512:

<<TABLE_CPDP>>

Two boundaries raise instead of running: Q_LEN not divisible by cp * 128, and a folded microbatch wider than the context window, since the CP mask rebuild is causal-only and cannot represent a document boundary.

Not in this PR: CP inside the vision tower and the report's dynamic CP for large images -- next, on the multimodal path. Without context_parallel_degree > 1 none of this executes.

Tested: CPU contract tests for the two things that used to fail silently; a cp2 integration cell.
