# TP 下的 DTensor 约束:KDA 与 MoE 为何那样包

从 `kimi_k3/parallelize.py` 的模块 docstring 搬出来。搬的理由:那段有 77 行,是 reviewer 打开
文件第一眼撞上的东西,而两条仓库规矩都要求注释精简、WHY 进文档
(logbook CLAUDE.md "inline comments: one line max; move WHY to PR body/docs";
torchtitan `.claude/CLAUDE.md` "minimize comments").

内容本身没有改动 —— 它记的是三条硬约束和四条事实,都还成立。

## Why KDA is ``NoParallel`` on the tp axis
----------------------------------------
The whole ``delta_attention`` is registered ``NoParallel``, so every child param
(q/k/v/o projections, conv1d weights, ``A_log``, ``dt_bias``) becomes a
``DTensor(Replicate)`` on ``tp_mesh``. Three consequences worth stating once:

* **It is required for FSDP+TP composability.** ``clip_grad_norm_`` stacks
  per-param grad norms across the parameter list, and the stack fails if some
  norms live on ``(fsdp, tp)`` and others on ``(fsdp,)`` alone.
* **The linears compose, the kernels do not.** Inside the KDA forward every
  linear receives a DTensor input from ``input_layernorm`` and DTensor weights
  from this wrap, so it produces DTensor. fla-core's triton kernels
  (``causal_conv1d`` inside ``ShortConvolution``, ``fused_kda_gate``,
  ``chunk_kda``, ``FusedRMSNormGated``) do not dispatch through DTensor, so
  ``ShortConvolution.forward`` and ``FusedRMSNormGated.forward`` are shimmed to
  ``to_local`` weight and input at the kernel boundary and ``from_local`` the
  output. ``fused_kda_gate`` and ``chunk_kda`` are called explicitly, and their
  to_local-and-call wrappers live in ``model.py``.
* **The exit is plain.** ``local_output_grad_placements=Replicate`` ``to_local``s
  the output at the module exit, matching MLA's ``o_proj`` and the dense MLP's
  ``down_proj``, so ``attn_out`` is a plain tensor everywhere and both
  partial-block accumulation and ``block_attn_res`` see one tensor kind.

## Why the MoE TP plan wraps LEAVES with ``NoParallel`` rather than the container
--------------------------------------------------------------------------------
Four facts, all load-bearing, kept here rather than inline at the call site:

1. **The mechanism.** Registering each leaf with ``NoParallel`` puts its params on
   ``tp_mesh`` as ``DTensor(Replicate)``. The shared MoE forward ``to_local``s its
   DTensor input at entry, so gate/experts/shared_experts receive PLAIN x; each
   leaf's ``prepare_input`` hook converts back to DTensor, runs on DTensor
   (params are DTensor too, so the matmul composes), and returns plain via
   ``local_output_grad_placements=Replicate``.
2. **Why the experts need nothing extra.** ``GroupedExperts.forward`` and
   ``KimiMLP``'s gate/up/down projections dispatch normally on DTensor because
   their ops are standard ``F.linear``; ``GroupedExperts`` additionally
   ``to_local``s its DTensor params before the grouped_mm kernel.
3. **Why NOT the ``moe`` container.** The MoE forward strips x with ``to_local``
   AFTER the parent's ``prepare_input``. Wrapping ``moe`` or ``moe._moe`` would make
   the input a DTensor at MoE entry, ``to_local`` would make it plain, and the
   gate would then receive plain x while holding DTensor weights -- which errors.
   Wrapping the gate ITSELF keeps the boundary right.
4. **Required for FSDP+TP composability.** ``clip_grad_norm_`` stacks per-param
   grad norms across the whole parameter list, which fails if MoE params live on
   ``(fsdp,)`` while MLA params live on ``(fsdp, tp)``. ``NoParallel`` promotes
   the MoE params to ``(fsdp, tp)`` after the FSDP wrap.

## `_patch_fla_for_dtensor` 的原 docstring(35 行)

Build DTensor-safe forwards for ShortConvolution + FusedRMSNormGated.

Returns ``{class: forward_fn}`` for :func:`_bind_fla_dtensor_shims` to bind
per instance. Nothing here mutates the fla classes.

Both classes wrap fla-core triton kernels (``causal_conv1d``,
``fused_norm_gated``) that take raw tensor pointers and don't
dispatch through DTensor. Under TP, KDA's delta_attention is
NoParallel-wrapped — child params are DTensor(Replicate) on tp_mesh
and inputs arrive as DTensor too. Calling the triton kernel with
DTensor crashes.

The patch wraps each class's ``forward`` to:
1. ``to_local`` the input ``x`` (preserving DTensor mesh+placements
   for re-wrap at the output) and any positional / keyword args
   that happen to be DTensors.
2. Run the original forward unchanged. The forward calls the
   kernel with ``self.weight`` (still a DTensor on the module, but
   autograd-traced via to_local under the hood — see below) — so
   we additionally redirect ``self.weight`` access to its local
   view via a transient property override on the instance during
   the call.
3. Re-wrap the output as DTensor on the same mesh+placements so the
   parent's (NoParallel) prepare_output hook receives a DTensor.

Critically we do NOT swap ``self._parameters`` to a fresh
``nn.Parameter(local)`` — that would break autograd (the new
parameter has no link back to the DTensor original, so kernel
backward would write into a temporary local tensor and the
DTensor's grad would never see the contribution). Instead we mask
the attribute lookup so the kernel sees a plain ``self.weight`` for
the duration of forward, but autograd's saved-tensors reference the
local-view of the DTensor (which is differentiable through to_local).

Patch is idempotent: ``cls._fla_orig_forward`` is set on first patch.
