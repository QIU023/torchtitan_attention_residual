# 模块级 LoRA 的语义与约束

从 `kimi_k3/lora.py` 的模块 docstring 搬出。原文 54 行,
是 reviewer 打开文件第一眼撞上的东西;两条仓库规矩都要求注释精简、WHY 进文档。
内容未改。

``apply_lora`` swaps target ``nn.Linear`` projections for
:class:`KimiLoRALinear` wrappers after build.

Semantics:

* ``lora_b`` zero-init -> the wrapped model is BIT-IDENTICAL to the
  base model at step 0 (composes with the alpha graft gate: gated
  graft + LoRA both preserve the pretrained function exactly).
* Base-freeze walks the whole model, EXCEPT the AttnRes graft params
  (pseudo-queries, norms, alphas): those are new zero-init params with
  no pretrained value -- LoRA-ing them is meaningless, they must train
  full-param (the "alpha-fullparam exception").
* ``trainable_state_dict`` gives the LoRA-only checkpoint payload
  (adapters + AttnRes params), the unit veRL weight-sync ships.

Plain input against a DTensor base weight
-----------------------------------------
``forward`` has a branch for a plain input meeting a DTensor ``base.weight`` --
NoParallel descent puts one there, e.g. MoE shared experts running in
plain-tensor land, because the MoE ``to_local``s direct-child weights but not one
nested under a wrapper's ``.base``. That branch bypasses ``self.base`` entirely,
so the TP style's own collective never runs and cannot cover for a wrong choice.
WHICH AXIS is sharded therefore decides correctness:

* **Colwise or Replicate**: the local product IS this rank's output shard (or the
  full output), exactly what a ``use_local_output=True`` style produces. Nothing
  to reduce.
* **Rowwise** (``Shard(1)``, the CONTRACTED in_features axis): the local product
  is only this rank's PARTIAL contribution and must be summed across the mesh.
  ``to_local()`` drops the DTensor that would have summed it, so a partial value
  escapes as a plain tensor -- and everything downstream assumes a plain tensor
  is replicated. Measured cost of getting this wrong: rank-dependent values in
  the residual stream from MLA's ``o_proj`` onward, layer 0 ``o_proj`` differing
  by 3.5e-01 against a magnitude of 2.3e-01, every later activation diverging,
  while the same architecture without LoRA stayed bit-identical.

``o_proj`` is the only site that reaches this branch with a Rowwise base: the MLA
attention output is built in plain-tensor land, whereas the dense FFN's
``down_proj`` receives a DTensor from the Colwise gate/up pair and so takes the
``self.base(x)`` path.

TP for LoRA IS wired (parallelize.apply_tp_kimi_k3): a Colwise/
Rowwise style on a wrapped projection is redirected to ``.base`` and the
adapters are distributed to match (Colwise -> lora_a Replicate / lora_b
Shard(0); Rowwise -> lora_a Shard(1) / lora_b Replicate); any remaining
adapters (NoParallel MoE shared experts) go Replicate so clip_grad_norm_
sees one mesh. The forward aligns adapters with the input's tensor kind.
LoRA composes with FSDP / TP / EP / PP / CP and their 4D combos.
