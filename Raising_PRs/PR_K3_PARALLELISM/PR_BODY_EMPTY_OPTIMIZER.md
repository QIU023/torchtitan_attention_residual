# PR title: [optimizer] A model part with nothing to train gets an empty optimizer and scheduler

Fork branch `k3_empty_optimizer` = `1c075f212` (two commits on main `ac10ca48f`, independent of the parallelism PRs; the first is the change, the second its tests). Verified on this box with the CPU tests below; no GPU run is needed for it.

--- PASTE BEGIN ---

## Summary

A pipeline stage that owns only frozen weights has no parameter to optimize; LoRA under pipeline parallelism produces such a stage (a vision-tower stage with no adapter targets, or an all-KDA stage when the targets are MLA-only). Today `OptimizersContainer` raises `param_groups pattern matched no parameters` on it, and `LRSchedulersContainer` asserts on an optimizer list of length zero.

- The param-group matcher raises on an empty match only when the model part has trainable parameters and none matched (the pattern is wrong); a part with no trainable parameter at all is warned about and skipped.
- The container still initializes torch's `Optimizer` with one empty param group (`all_params or [{"params": []}]`): torch rejects an empty params list but accepts an empty group, so the container's hooks and `param_groups` exist and `step()` / `zero_grad()` are no-ops.
- The scheduler container accepts zero schedulers, reports an empty state dict for them (rather than a zero `last_epoch` a later load would apply as progress; DCP accepts a rank contributing no keys), and restores nothing on load.

## Implementation

`torchtitan/components/optimizer/optimizer.py`: the empty-match branch of `_build_param_groups` checks `any(p.requires_grad for p in model.parameters())` before raising, otherwise logs and `continue`s; `_post_init` passes the empty-group fallback to `Optimizer.__init__`. `torchtitan/components/optimizer/lr_scheduler.py`: the `len(optimizers) > 0` assert is gone, `state_dict()` returns `{}` and `load_state_dict()` returns early when there are no schedulers.

## Limitations

A part with trainable parameters that no pattern covers still raises, as before (`test_error_on_zero_matches`, `test_uncovered_params_raises`). Nothing changes for a model whose every part has something to train.

## Tests

```text
pytest -q tests/unit_tests/cpu/test_lr_scheduler.py tests/unit_tests/cpu/test_optimizer_param_groups.py
```

Result: 33 passed (29 existing, 4 new): a fully frozen part yields no param groups; the container built on it is an `Optimizer` with one empty group and no-op `zero_grad` / `step`; a frozen part next to a trainable one leaves the trainable part's parameters covered; a scheduler container over zero optimizers steps, saves `{}` and loads `{}`. pre-commit passes on the touched files; `pyrefly check` reports the same error set as main `ac10ca48f`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1
