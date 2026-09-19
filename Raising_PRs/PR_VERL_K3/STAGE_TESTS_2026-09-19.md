# Running a test at every stage of the veRL split, for the first time

The kit's own manifest records that it had never done this: every stage was checked with `ast.parse` and `ruff`, and no stage had ever run a test. This is that gap closed, and it found a defect that only a per-stage test run can see.

## How the stages were run

Base is upstream veRL `3efe38c7`, which is still today's upstream main, so the no-intersection condition holds without re-checking. Each patch is applied on the previous result and the cumulative set of test files that patches 01 to N introduce is run.

Two local accommodations, both named so that nothing here reads as upstream-clean:

- `verl/third_party/vllm/__init__.py` is overlaid from patch 00. The vLLM on this box is a source build reporting `0.1.dev1+g6dc76a9ad`, which upstream's `>= 0.18.0` gate rejects; the shim reads `VERL_VLLM_VERSION`, set to `0.18.0`.
- `torchtitan` resolves to upstream `6c2dadbb3` rather than to the integration tree. That is not a convenience: the engine PRs target upstream and a reviewer runs them there. It also matters because the integration tree at `42691735c` **lacks** `prepare_context_parallel_input`, which upstream main has, so resolving to the tree fails stages 01 to 03 for a reason that has nothing to do with the split.

## The result

    stage                          cumulative test files   result
    01_engine_tp_packed_and_compat            4            collection error
    02_engine_pipeline                        6            collection error
    03_engine_ep_lora_qat_sync                9            collection error
    04_engine_context_parallel               11            19 passed, 6 skipped
    05_kimi_k3                               12            24 passed, 6 skipped
    06_metrics_logprob_diff                  12            24 passed, 6 skipped

The full stack is green. The first three stages are not, and they fail for one reason.

## The defect: patch 01 ships context-parallel tests that patch 04 implements

`tests/workers/test_torchtitan_engine_cp_config.py` is introduced by patch 01 and rewritten by patches 04 and 05. Patch 01's version imports unconditionally:

    ImportError: cannot import name 'ContextParallelLoadBalancerConfig' from 'torchtitan.config'

That name exists on the integration tree, at `torchtitan/config/configs.py:130`, and not on upstream torchtitan main. Patch 04's version of the same file imports the K3 model modules instead and carries a skip guard, which is why stage 04 reads 6 skipped rather than an error: the guard does its job when the K3 model is not importable, and patch 01's version has no guard to do it with.

The same shape appears one level down. `tests/workers/test_torchtitan_engine_config_fields.py`, also introduced by patch 01, carries `TestContextParallelBackendIsCheckedByTheConfig`. Patch 01 contributes two lines mentioning `context_parallel_backend`; patch 04 contributes fourteen, which are the validation those tests exercise. Run on their own with upstream torchtitan, stage 01's config-field tests are 7 passed and 2 failed, and the two failures are that class.

So the first upstream-bound PR, as the split stands, ships context-parallel tests whose implementation is three PRs later. A reviewer checking out PR 1 alone sees a collection error and two failures. Nothing in `ast.parse` or `ruff` can see that, which is why it survived every previous verification.

## What it does not mean

The split is not wrong about where the *code* goes. Patches 04 and 05 hold the CP implementation, and the full stack is green at 24 passed. What is misassigned is test content: a line-level split put an early form of two CP test files into the first patch. The fix is a reassignment inside the kit, moving those test lines from patch 01 to patch 04, not a change to the implementation.

## Fixed, and every stage is green

Two changes, one to the engine and one to the split's assignment.

**The engine change.** `_parallelism_compat_kwargs` built `ContextParallelLoadBalancerConfig(load_balancer_type=None)` unconditionally under CP. The two torchtitans model that field differently: upstream `torchtitan/config/configs.py:243` has `context_parallel_load_balancer: str | None = "headtail"`, while the fork defines a dataclass for it at `configs.py:130`. The import was already function-local, so importing the engine on upstream worked and only the CP path would have raised; it now pins the balancer off in whichever shape the tree exposes, which is the same defensive pattern the function already used for `spmd_backend`. Committed to `kimi_k3_integration_rebased` as `b5a79e15`.

**The assignment change.** `TestContextParallelBackendIsCheckedByTheConfig` exercises `TorchtitanEngineConfig(context_parallel_backend=...)`, a field patch 04 adds, and it sat in patch 01. It is now split out of the whole-file assignment: items 40 to 53 of `test_torchtitan_engine_config_fields.py` travel with 04, the rest stays in 01, with the file's trailing blanks and `if __name__` footer kept in 01 so stage 01's file is still valid.

Regenerating the kit on the new head needed a remap, which is the step the manifest describes for every head move. The engine file's item list went from 2170 to 2178; a full alignment including context items, so that `MOVES` anchors map too, carried 2165 of them, the eleven new items are the compat rewrite and take patch 01's ordinal from the lines they replace, and `VARIANTS` and `MOVES` were remapped with the same table. The builder's own check is what makes this safe to do mechanically: it reconstructs the final stage and compares it to HEAD, and a first attempt that remapped only `ASSIGN` and left `MOVES` on the old indices was caught by exactly that, 31 lines out of place.

Patch sizes move the way conservation says they should: patch 01 went 858 to 868 with the engine change, then 868 to 854 when the test class left; patch 04 went 801 to 826 taking it. Every other patch is byte-identical in size to the pre-change build.

    stage                          result
    01_engine_tp_packed_and_compat   17 passed
    02_engine_pipeline               22 passed, 1 skipped
    03_engine_ep_lora_qat_sync       22 passed, 4 skipped
    04_engine_context_parallel       19 passed, 6 skipped
    05_kimi_k3                       24 passed, 6 skipped
    06_metrics_logprob_diff          24 passed, 6 skipped

So each upstream-bound patch now passes its own tests standing alone on upstream torchtitan, which is the tree a reviewer runs them against.

## The capability-probe item is already done

The plan lists, as a step before any draft PR, replacing three flags keyed on `torchtitan_name == "kimi_k3"` with capability probes. Nothing records it as done, so it was on tonight's list. It is done, and has been for some time: the branch carries no `torchtitan_name == "kimi_k3"` anywhere, and `torchtitan_name` survives only to pick the model module.

The three behaviours are each probed now:

- the folded `[T]` token stream, at `transformer_impl.py:461-466`, as `type(model_parts[0]).preprocess_inputs is not BaseModel.preprocess_inputs`, with the comment saying in as many words that it is "probed on the model, not on its name", and read back at the two use sites through `getattr(module, "folded_token_stream", False)`;
- the context-parallel transform, chosen by `_context_parallel_transform(model_config, backend)` from the configured backend string;
- the contiguous rank-ordered CP shards, through `_parallelism_compat_kwargs` pinning the balancer off whenever CP is on, for any model.

Three mentions of `kimi_k3` remain in the engine file and none is a branch on the name: two are imports of the KDA classes inside the CP-KDA backend, and one is a comment. So the plan's item is stale rather than outstanding, and patch 05 is already only the model map, the processor and the patch-size read.

## The six branches are cut

Item 1 of the remaining-work list, the part that survives after the diagnostics question: the split is now six stacked branches on the fork, each on upstream veRL main `3efe38c7`, in application order.

    verl_engine_tp_packed          ac6dd230   tensor parallel on the packed stream, plus the compat pieces
    verl_engine_pp                 f0e9dff3   pipeline parallelism
    verl_engine_sync_ep_lora_qat   8b7c95cb   expert stacks, LoRA and QAT in the weight sync
    verl_engine_cp                 f26c72d1   context parallel over the packed stream
    verl_kimi_k3                   5e7d4497   the Kimi K3 and Kimi Linear surface
    verl_metrics_logprob_diff      4f0a255b   the rollout log-probability difference metric

Stripping the diagnostics needed no separate step. They live entirely in patch 00, which is applied last precisely so that 01 to 06 never carry them, and each of the six greps zero for the local markers (`VERL_LOCAL`, `partial_dtensor`, `BFX9`). Building the branches from the patches therefore produces trees that are clean by construction rather than by removal, which is also why the integration branch can keep carrying them.

Every commit message is written fresh and none carries a trailer, checked by grep over the whole range.

The stack top was tested the same way the stages were, with the vllm shim overlaid for the run and then reverted so the branch stays clean: 24 passed, 6 skipped, the same as stage 06 through the patch path.

What is deliberately not done here: nothing is filed. The branches exist on the fork and the PR bodies are not written.
