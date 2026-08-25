Adds context parallelism to the Kimi K3 text decoder. The two attention kinds
need different mechanisms, so both are here: Ulysses on the Gated MLA layers,
and KDA Context Parallelism -- the KCP of the K3 report -- on the KDA layers,
which is a conv halo exchange plus a prefix scan over the recurrent state rather
than anything softmax-shaped. As far as I can tell this is the first Ulysses
implementation in torchtitan; grep finds no other. If you would rather it live
under distributed/ as a model-agnostic piece than inside the model folder, say
so and I will move it.

MLA Ulysses is one all-to-all that trades the sharded axis, sequence for heads,
then the attention backend runs unchanged, then a second trades back. The rotary
key slice stays outside that all-to-all: it is headless, one vector per token
shared by every head, so it is all-gathered along the sequence and expanded onto
the local heads afterwards. Packing the already-expanded key instead sends the
same values once per head and reassembles them against the wrong head subset,
which shows up as a forward that diverges from the same layer without CP. The
local head count is read off the projection width rather than n_heads, and
k_rope goes through a Partial-gradient boundary before the gather, so that a
later column-parallel wq_b/wkv_b does not silently change either.

KCP cannot be expressed declaratively. KDA runs fla triton kernels that take raw
pointers and never dispatch through DTensor, so no ShardingConfig can drive
them. That is why Decoder.Config grows cp_via_sharding_config, default True: the
base class calls validate_cp_backend unconditionally, and its own docstring says
that check is for models declaring CP in a ShardingConfig. Declarative models
are unchanged.

CP resplits the sequence, so unlike PP or EP it is not expected to be
bit-identical to dp1. kimi_k3_debugmodel_text at seq 1024 -- FlexAttention's
BlockMask needs Q_LEN % (cp * 128) == 0, which is what sets the length, and
FlexAttention is the backend the model already uses -- seed 42,
--debug.deterministic, one seed checkpoint loaded by every cell:

<<TABLE_CPSEQ>>

Same, on a 2D mesh with data parallel, seq 512:

<<TABLE_CPDP>>

Two boundaries raise instead of running: Q_LEN not divisible by cp * 128, which
is FlexAttention's, and a folded microbatch wider than the context window, since
the CP mask rebuild is causal-only and cannot represent a document boundary.

Not in this PR: CP inside the vision tower, and the report's dynamic CP that
splits a large image along the patch dimension. Those come next, on the
multimodal path.

Without context_parallel_degree > 1 none of this executes.

Tested: CPU contract tests for the two things that used to fail silently; a cp2
integration cell.
