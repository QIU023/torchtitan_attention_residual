# AttnRes 如何织入解码器栈

从 `kimi_k3/attn_res_model.py` 的模块 docstring 搬出。原文 38 行。内容未改。

:func:`torchtitan.models.kimi_k3.attn_res.block_attn_res`
(shared with the existing ``AttnResLlama3Model`` / ``AttnResModel`` so
that a single implementation is used across both experiments).

Per-layer AttnRes (matching ``AttnResTransformerBlock`` in the
``attn_res/`` experiment):

* Two AttnRes applications per decoder layer — one before attention,
  one before the FFN — each contributing an RMSNorm + zero-initialized
  pseudo-query ``w_l ∈ R^d``.
* At a block-start layer the running ``partial_block`` is committed
  into ``blocks`` and reset; subsequent layers accumulate into the new
  ``partial_block`` until the next block start.
* On the last stage the final aggregation (one extra pseudo-query +
  RMSNorm) runs before ``norm`` + ``lm_head``, mirroring the reference.

``_return_only_new_blocks`` flag is respected on forward so the
cross-stage cache adapter (``pipeline_adapter.py``) can drive this model
unchanged at multi-node scale. Local FSDP-only training leaves the flag ``False`` and passes
the full accumulated block stack forward.

Paper (Kimi Linear tech report §5):

> "AttnRes introduces only one RMSNorm and one pseudo-query vector
> wl ∈ R^d per layer, amounting to a negligible fraction of the total
> parameter count. Crucially, all pseudo-query vectors must be
> initialized to zero."

We use the two-per-layer pattern from the ``attn_res/`` experiment to
stay consistent with the validated PP adapter + CPU tests. The
paper's "one per layer" count treats each (attention, FFN) pair as a
sub-layer pair; the two-per-layer count here is the sub-layer-level
view and is equivalent.
