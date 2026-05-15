# Upstream PR drafts — ready to file

Each subfolder contains everything needed to file one upstream pull request:

- **`PR.md`** — the full PR description (title, summary, motivation, patch, test plan, filing checklist).
- **`commits.md`** — the backing commit(s) on our fork (commit hash, branch, files touched) plus a cherry-pick recipe for getting the change onto an upstream fork branch.

The English-only filing flow per PR:

1. Read `commits.md` and prepare an upstream fork branch (`git checkout -b <pr-branch> upstream/main`).
2. Cherry-pick or hand-port the backing commit(s) listed in `commits.md`.
3. Push the branch to your upstream fork.
4. Open the PR using `PR.md` body verbatim.

## Current PRs

| Folder | Target repo | Title | Backing fork commit |
|---|---|---|---|
| `PR1_sglang_disable_shm_mm/` | `sgl-project/sglang` | `SGLANG_DISABLE_SHM_MM` env to force CPU mm transport | `74083ffae` on `QIU023/sglang@main` |
| `PR4_torchtitan_parallelize_fn_signature/` | `pytorch/torchtitan` | `parallelize_fn` signature stability for `experiments.rl.PolicyTrainer` | (proposed; no fork commit yet — see `commits.md`) |
| `PR7_sglang_kda_causal_conv1d_fp16/` | `sgl-project/sglang` | KDA `causal_conv1d_triton` fp16 dtype type-join fix | bundled inside `a6c46168a` on `QIU023/sglang@attention_residual_inference` — needs isolation |

## Suggested filing order

1. **PR #1** first — smallest possible upstream contribution, builds reviewer credibility.
2. **PR #4** next — required prerequisite for the future engine-agnostic Generator PR (`UPSTREAM_PR_LIST.md` PR #12).
3. **PR #7** in parallel with #4 — independent kernel fix, no cross-dependency.

See [`../UPSTREAM_PR_LIST.md`](../UPSTREAM_PR_LIST.md) for the full inventory of 12 PR candidates and the long-form filing order across all of them.

## What goes here vs. what stays in `UPSTREAM_PR_LIST.md`

- **`UPSTREAM_PR_LIST.md`** is the inventory + design discussion across all 12 PR candidates. Lives at repo root for visibility.
- **`Raising_PRs/<PR>/`** is the filing kit for each PR that is *currently being prepared*. Once a PR is filed, this folder gets a link to the upstream PR number; once merged, the folder can be deleted or archived.
