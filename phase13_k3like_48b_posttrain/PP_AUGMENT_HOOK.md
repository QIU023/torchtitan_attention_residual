# 跨虚拟 stage 的梯度捕获与求和

从 `kimi_k3/pipeline_adapter.py` 的 `_install_augment_hook` docstring 搬出。原文 37 行。内容未改。

Register a tensor grad hook on ``block_tensor`` that, when the
tensor receives its incoming grad during the producer stage's own
backward, pops the matching captured-grad slot from ``rank_cache``
and SUMS the captured grad into the incoming grad.

``expected_captures``, if provided, is the number of same-rank
later virtual stages that SHOULD deposit into this slot during the
mb's backward window (from
:meth:`BlockLayoutTables.expected_same_rank_captures`). The hook
RAISES when the observed count diverges -- the failure this catches is
a consumer's backward not firing, which drops its grad contribution
into the bit bucket, and the step that follows would apply an
incomplete gradient with nothing anywhere saying so. There is no
configuration in which continuing is the right answer, so this is not
a warning: the count is derived from the static layout, delta mode is
gated on Interleaved1F1B, and under that schedule every same-rank
consumer's backward provably precedes the producer's own.

Replaces the prior ``_LocalCacheAugment`` autograd.Function pattern:
under real PP scheduling the Function's output-view trick did not
reliably sever the consumer->producer autograd chain under the
PP8xVP4 pressure tests, so the producer's own backward
was being triggered during the CONSUMER's backward traversal,
freeing the producer's saved tensors before the producer's own
backward ran.

A tensor grad hook fires exactly once per backward call on the
tensor's accumulated grad, strictly DURING the containing stage's
own backward. No Function layer is interposed, so the consumer's
backward can not reach ``block_tensor.grad_fn`` at all (the consumer
only ever sees the detached cache entry — see ``RankLocalCache.append``).

Eval / no_grad path: when this is called inside a no_grad / eval
context (e.g. ``pp_schedule.eval()`` in torchtitan's Validator),
``block_tensor.requires_grad`` is False and ``register_hook``
raises. There is no backward in eval, so the augment-on-backward
semantics are vacuous — silently skip hook installation.
