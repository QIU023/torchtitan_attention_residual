# Backing commits — PR #4 `parallelize_fn` signature stability

> **OBSOLETED-BY-UPSTREAM 2026-05-17.** See `PR.md` header. Upstream
> `627f4a31` (2026-04-20) already widened the kwargs. Do not file. Fork
> reconciliation tracked separately (delete launcher-side `parallelize_fn`
> wrapper after upstream merge to avoid double-kwarg injection).

## Discovered in

**Phase 11** — RLHF / GRPO entry-point on Kimi-Linear AttnRes
(`phase11_rlhf_grpo_infra/rlhf/run_grpo_kimi_attn_res.py`). The first call to
`PolicyTrainer._build_model` with a non-Qwen3 model_spec failed at
trainer construction with `TypeError: parallelize_kimi_linear()
missing 4 required positional arguments`. Workaround at launcher
boundary; clean fix proposed for upstream `trainer.py`.

## Fork source

| Field | Value |
|---|---|
| Repo | `git@github.com:QIU023/torchtitan.git` |
| Branch | `attention_residual_dev` |
| Commit | **none** — see "Status" below |
| Files implicated | `torchtitan/experiments/rl/actors/trainer.py` (PolicyTrainer._build_model) |

Status: **No fork commit yet to cherry-pick.** PR #4 is a *propose-and-file*
PR. Our fork carries a per-launcher workaround (`phase11_rlhf_grpo_infra/rlhf/run_grpo_*.py`
constructs the missing kwargs and patches them in), but the cleaner trainer-
side fix has not been committed to the fork's `trainer.py`.

## What the fork currently does

`torchtitan/experiments/rl/actors/trainer.py` lines 283-288 still call
`model_spec.parallelize_fn` with only 3 kwargs:

```python
model = model_spec.parallelize_fn(
    model,
    parallel_dims=self.parallel_dims,
    parallelism=config.parallelism,
    compile_config=config.compile,
)
```

This works for Qwen3's `parallelize_qwen3`, fails for every other
torchtitan `parallelize_*` (which require 4 more kwargs).

## Filing path (no cherry-pick, hand-port the proposal)

```bash
# 1. Clone upstream torchtitan; branch off main.
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout -b experiments-rl-parallelize-fn-signature upstream/main

# 2. Apply the trainer.py patch from PR.md by hand. The patch is small —
#    widen the parallelize_kwargs dict in PolicyTrainer._build_model
#    to include training, model_converters, ac_config, dump_folder.
#    See PR.md "Patch" section for the exact code.

# 3. Add a smoke test in torchtitan/experiments/rl/tests/ that instantiates
#    PolicyTrainer against a non-Qwen3 model_spec (e.g. parallelize_llama
#    or parallelize_deepseek_v3) and verifies _build_model succeeds.

# 4. Commit + push.
git add torchtitan/experiments/rl/actors/trainer.py \
        torchtitan/experiments/rl/tests/test_policytrainer_model_spec_compat.py
git commit -m "[experiments/rl] PolicyTrainer: widen parallelize_fn kwargs for non-Qwen3 model_specs"
git push origin experiments-rl-parallelize-fn-signature

# 5. Open PR on github.com/pytorch/torchtitan using PR.md as the body.
```

## Why no fork commit was made

The fork's `phase11_rlhf_grpo_infra/rlhf/run_grpo_*.py` launchers solve the problem at
the launcher boundary (each launcher wraps a `parallelize_fn` adapter
that injects the missing kwargs). That works for our internal use but
is not the right shape to send upstream — the fix belongs in
`PolicyTrainer._build_model` itself so every RL launcher gets it for
free.

If you want a clean cherry-pickable commit on the fork before filing,
apply the patch from PR.md to `torchtitan/experiments/rl/actors/trainer.py`
on `attention_residual_dev`, commit, push, then come back and update this
file with the commit hash.

## Conflict surface

`PolicyTrainer._build_model` was recently touched by upstream main on
`attention_residual_dev` (e.g. `82b08e74 PolicyTrainer model-spec layout
assert → soft warning`, `fd606e68 PolicyTrainer DCP-native initial load
path`). The proposed patch is local to the `parallelize_fn` call site
(lines ~280-290 today), so conflicts with those upstream-merged commits
should be minimal — but rebase before pushing the PR.

## Notes for the PR opener

- This PR is the prerequisite for the larger PR #12 (engine-agnostic
  `Generator` abstraction + SGLang reference impl). Land this first.
- The patch is `~10 lines of trainer.py + 1 ~30-line smoke test`.
  Total work: ~1 day including writing the PR description and running
  CI locally.
- Maintainers worth CCing once filed: PolicyTrainer authors (look in
  `git log --format=%an torchtitan/experiments/rl/actors/trainer.py`).
