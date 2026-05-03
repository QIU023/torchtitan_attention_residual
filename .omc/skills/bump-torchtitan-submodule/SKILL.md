---
name: bump-torchtitan-submodule
description: Bump the torchtitan/ git submodule pointer to a new fork SHA, smoke-test, and write a phase-tagged commit.
triggers:
  - "bump submodule"
  - "update torchtitan pointer"
  - "submodule pointer"
agent: executor
model: sonnet
---

# bump-torchtitan-submodule

## Purpose
Move the `torchtitan/` gitlink to a new commit on
`QIU023/torchtitan@attention_residual_dev` (or another fork branch), run the
unit-test smoke, and produce a commit message in the style this repo already
uses (e.g. `753026c phase 6: bump submodule pointer to 84d42c9`).

Critical invariant from `.gitignore`: **`torchtitan/` is a gitlink, not a
nested working copy.** Do not `git add` the directory contents and do not
re-ignore the path.

## Workflow

1. **Pick the target SHA**
   - Default: latest on the fork branch.
     `git -C torchtitan fetch origin attention_residual_dev`
     `git -C torchtitan rev-parse origin/attention_residual_dev`
   - Or accept a SHA passed as the skill argument.
   - Print the short log range about to be incorporated:
     `git -C torchtitan log --oneline <current>..<target>`
     and ask the user to confirm before the gitlink moves.

2. **Move the gitlink**
   - `git -C torchtitan checkout <target-sha>`
   - In the parent repo: `git status` should show `modified: torchtitan` (one
     line, not a flood of file diffs). If it shows file-level diffs, abort —
     the submodule was clobbered into a worktree.

3. **Smoke test**
   - Activate `attnres` env.
   - `cd torchtitan && python -m pytest tests/unit_tests/test_attn_res.py -v`
   - On failure, do **not** revert silently; report which test broke and stop.

4. **Stage and commit (only on user confirmation)**
   - `git add torchtitan`
   - Commit message format:
     ```
     phase <N>: bump submodule pointer to <short-sha>

     <one-line description of what landed on the fork>
     ```
   - `<N>` is the current phase the user is working on; infer from the most
     recent untracked / modified `phaseN/` directory if not given.

## Usage

```text
/oh-my-claudecode:bump-torchtitan-submodule           # latest on fork branch
/oh-my-claudecode:bump-torchtitan-submodule abc1234   # specific SHA
```

## Do not

- Do not run `git submodule update --init --recursive` after the bump unless
  the user asks — it clobbers any local fork-side work-in-progress.
- Do not edit files inside `torchtitan/` from this skill; that's a separate
  repo with its own PR flow.
- Do not skip the smoke test even if "it's just a docs change on the fork" —
  the smoke is fast and catches submodule-vs-env drift.
