# 上游 AC 默认值的两点观察(2026-08-24)

针对上游 commit `f925dad99 "fix activation checkpointing"`(JavaZeroo, 2026-08-19)。
用户可能会去 PR 下评论,这里存事实与草稿。

## 事实

diff 的两处关键改动:

    -        activation_checkpoint=None,
    +        activation_checkpoint=SelectiveAC.Config(),

    -    if ac_config is not None:
    -        raise NotImplementedError(
    -            "Kimi K3 FSDP2 does not support activation checkpointing yet.")
    +        ac_policy.apply(model)
    +        if model.vision_encoder is not None:
    +            ac_policy.apply(model.vision_encoder)

1. **debug flavor 的默认从 `None` 翻成了 `SelectiveAC`**,所以 debug 模型不再覆盖
   AC 关闭这条路径。
2. **`ac_policy.apply(model.vision_encoder)` 让视觉塔的块也被 checkpoint。**
   `SelectiveAC.apply()` 用 `ptd_checkpoint_wrapper` 包住 `layers` 的每个子模块,
   反向重算时用**保存的入参**重放 forward。因此:任何在塔里保存"每次前向状态"、
   并在块循环后清空的特性,都活不过重算。

我们踩到的具体形态:per-image 的 CP patch plan 挂在模块状态上,重算时读到 `None`,
静默走回复制分支,撞 `attention_masks must be instance of BlockMask, got NoneType`。
改成把 plan 作为参数经 `VisionTransformerBlock` / `VisionAttention` 透传后,
AC 开与关五格逐位相同。

## 评论草稿(terse,无小标题、无加粗结构,符合 maintainer 的要求)

    Two observations on the activation checkpointing change. It flips the debug
    flavor default from activation_checkpoint=None to SelectiveAC.Config(), so
    the debug model no longer exercises the AC-off path anywhere -- might be
    worth keeping one flavor on each side. Second, ac_policy.apply(model
    .vision_encoder) checkpoints the tower's blocks, which means any tower
    feature that keeps per-forward state and clears it after the block loop does
    not survive recompute: the wrapper replays the block from the arguments it
    saved, so the state is already gone by backward. We hit this adding a
    per-image CP patch plan for the vision encoder -- on recompute it silently
    took the replicated branch. Threading the plan as an argument through
    VisionTransformerBlock/VisionAttention instead of module state fixes it and
    leaves other models unchanged (the parameter defaults to None); happy to
    send that if it is useful.

注意:提交信息里不要写第三方 `owner/repo#N` 或完整 PR URL,写 `PR-4025`。
