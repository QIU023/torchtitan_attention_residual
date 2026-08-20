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
