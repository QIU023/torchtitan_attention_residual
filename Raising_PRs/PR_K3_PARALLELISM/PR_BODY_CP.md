# PR body: context parallel for Kimi K3

按 maintainer 要求写成 terse 工程笔记:无小标题、无加粗结构、无证据表格,
精确的 op 级事实在最前。数字内联。详细证据留给 logbook 或后续 comment。

---

Adds context parallelism to kimi_k3. The two attention kinds need different
mechanisms, so both are here: Ulysses for the Gated MLA layers, and KDA Context
Parallelism (the report's KCP) for the KDA layers, which is a conv halo exchange
plus a prefix scan over the recurrent state rather than anything softmax-shaped.
As far as I can tell this is the first Ulysses implementation in torchtitan --
grep finds no other -- so if you would rather it live under distributed/ as a
model-agnostic piece than inside the model folder, say so and I will move it.

MLA Ulysses is one fused all-to-all that trades the sharded axis, sequence for
heads, then the attention backend runs unchanged, then a second trades back. The
rotary key slice is deliberately outside that all-to-all: it is headless, one
vector per token shared by every head, so it is all-gathered along the sequence
and expanded onto the local heads afterwards. Packing the already-expanded key
instead sends the same values once per head and reassembles them against the
wrong head subset, which shows up as a forward that diverges from the same layer
run without CP.

The head count is derived from the projection width rather than read off
n_heads, because wq_b/wkv_b are column-parallel and each rank holds n_heads/tp
of them. And k_rope goes through a Partial-gradient boundary before the gather:
wkv_a is replicated on the TP axis, so its gradient is the sum across TP ranks,
and without that the sum is dropped while the placement still reads Replicate.

KCP cannot be expressed declaratively. KDA runs fla triton kernels that take raw
pointers and never dispatch through DTensor, so no ShardingConfig can drive it.
That is why Decoder.Config grows cp_via_sharding_config (default True): the base
class calls validate_cp_backend unconditionally, and its own docstring says it is
for models that declare CP in a ShardingConfig. Declarative models are unchanged.

The vision tower gets the report's 5.2.3 dynamic CP: a large image is split
along the patch dimension with attention gathering key-value pairs across ranks,
and the CP group is divided into sub-groups with the large images balanced
across them, which is what keeps the communication fraction from growing with
scale. Images below the threshold stay replicated.

Files: new kcp.py, sharding.py, dtensor_ops.py, vit_cp_plan.py; kda.py gains the
two CP paths; model.py gains MLA Ulysses and the vision CP splice;
parallelize.py gains apply_cp_kimi_k3; one line in common/decoder.py.

Tested: CPU contract tests for the two things that used to fail silently, and
cp2 runs on the debug flavor. Numbers on request.

---

补充说明(cu_seqlens,回应可能的 maintainer 疑问):

KCP path passes cu_seqlens into chunk_kda. That is the CP shard boundary the KDA
prefix-scan runs over, not a packed-document boundary. It does not touch and does
not change the existing KDA rejection of packed-document masks / MMSamplePacking
-- sample packing stays unsupported exactly as upstream leaves it. The name
cu_seqlens is shared with the sample-packing convention, hence this note.
