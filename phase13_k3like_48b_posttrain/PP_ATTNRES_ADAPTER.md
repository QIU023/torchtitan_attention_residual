# AttnRes 跨 stage adapter 的设计

从 `kimi_k3/pipeline_adapter.py` 的模块 docstring 搬出。原文 48 行。内容未改。

by concatenating its cached prefix with the incoming delta.

Per-block backward grads ride the normal autograd + PP SEND_B path:

* Blocks this rank received from an earlier PP hop are slices of a
  ``recv_delta_tensor``; their grad flows back through that tensor
  and PP's built-in ``SEND_B`` ships it to the previous rank.
* Blocks this rank committed in an earlier virtual stage are bridged
  via a *rank-local* slot. At producer emission a tensor grad hook
  (:func:`_install_augment_hook`) is registered on the new block, and
  a DETACHED copy is written into the rank cache. At consumer read
  the detached cache entry is wrapped in :class:`_LocalCacheCapture`,
  whose backward deposits the grad in the slot and stops. When the
  producer's own backward runs, the hook pops the slot and SUMS the
  captured grad into the incoming grad before propagating into the
  producer's wrapped model. Detach is what guarantees the consumer's
  backward physically cannot traverse into the producer's forward
  graph -- a tensor-grad hook + ``Function.backward returning None``
  alone did NOT suffice under real PP + FSDP + AC rerun (PP8xVP4
  pressure-test reports: https://github.com/QIU023/torchtitan_attention_residual).

Both bridge mechanisms are pure local Python + a dict -- zero NCCL,
zero cross-rank state. The grad walks backwards hop-by-hop along the
same PP stage chain that forward uses (PP owns all NCCL, so no
deadlock risk), and each stage's forward graph is traversed exactly
once per mb so peak memory equals the naive-PP baseline.

:func:`pipeline_kimi_k3_with_cache_adapter` is a ``pipelining_fn`` plugged
into the experiment's ``ModelSpec``; it delegates to core
``pipeline_llm`` and (when ``TORCHTITAN_ATTNRES_CACHE=1``) wraps each
stage's submod. Schedule must be Interleaved1F1B; otherwise we warn.

Microbatch keying: the adapter keys its per-microbatch cache by the
schedule-owned integer chunk id. ``forward_one_chunk`` /
``backward_one_chunk`` are monkey-patched to stash the index on a
thread-local keyed per-adapter; forward and backward run synchronously
on the same thread, so autograd hooks that fire during backward can
read the mb index. The integer key is stable across P2P crossings
(unlike ``id()`` of a tensor).

See ``adapter_design.md`` at the project root for the full state
machine and invariants.
