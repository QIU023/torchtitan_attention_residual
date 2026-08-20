# 视觉 stage 为什么同时持有 embedding

从 `kimi_k3/multimodal_model.py` 的 `KimiK3ViTStage` docstring 搬出。原文 36 行。内容未改。

The vision PP stage under DEP (report 5.2.3): tower + embed + splice.

Report 5.2.3 requires ViT and text to be separate stages. This owns the tower,
and it also owns ``embed_tokens`` and the splice -- which needs saying, because
the obvious alternative does not work.

**Why the embedding lives here.** The splice needs both the features and
``input_ids``, and torchtitan passes positional args only to the FIRST stage. The
two ways to get ids to a later stage both fail: sending them over the pipe breaks
because PP's metadata inference pushes DUMMY values through it during
initialisation, and using them as embedding indices then asserts out of bounds in
the gather kernel (measured, not predicted); and putting them in the batch dict
would mean editing the shared collator, which is core. So the ids never leave
this stage. What crosses the hop is the spliced embedding stream -- one float
activation, which is exactly the homogeneous contract PP already expects, and
dummy values in it are harmless because nothing indexes with them.

The compute split the report asks for is unaffected: this stage does the vision
encode, the text stages do every transformer layer. An embedding lookup is not
text training compute.

**Spanning several stages** (report 5.2.3's other clause, "balances vision forward
and backward passes across PP stages") is selected by :meth:`set_dep_role`. With
one vision stage the role is ``"both"`` and this class behaves exactly as above --
that path is left untouched on purpose, since its numerics are already pinned.
With n > 1 the roles are:

* ``head`` -- ``patch_embed`` + its block share, AND ``embed_tokens``. Emits
  ``(patches_padded, text_embeds, sentinel_mask)``.
* ``body`` -- its block share on the patch stream, the other two passed through.
* ``tail`` -- its block share + final norm + merge + projector, then the splice.

Every pipe payload is a float activation, which is what makes this safe: dummy
values from PP's metadata inference are harmless because nothing indexes with them.
The patch stream is padded to a capacity derived from ``dep_max_*`` because PP
sizes its buffers once rather than per step.
