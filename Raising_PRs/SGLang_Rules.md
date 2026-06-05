# SGLang_Rules — conventions for filing PRs to `sgl-project/sglang`

Shared rules for every PR in this folder (PR1, PR3, PR7, PR8, PR9, PR11, PR13, PR15, …).
These are the things upstream reviewers / CI / pre-commit actually enforce — getting them
right up front avoids the predictable review bounce. Source: the fork's own
`sglang/.claude/rules/` + `sglang/.claude/skills/` and `.pre-commit-config.yaml`.

---

## 0. Golden path (TL;DR)

1. Branch off **current** `upstream/main` (fork moves fast — re-rebase if your base is stale).
2. Make the change **idiomatic** (env vars → `environ.py`; touching a big-class `__init__` →
   `init_*` helper; see §2).
3. **One commit**, imperative title `[srt/...] ...`, body in the official template (§4).
4. `py_compile` + `pre-commit run` clean (§3). No changes under `docs/` (§3).
5. Prefer a **GPU-free, stdlib-only reproducer/test** over prose; verify it on **Linux**
   (`/psm_*`, `resource_tracker`, SHM, NCCL, etc. behave differently on Windows) (§5).
6. Record the **base SHA** and your **commit SHA** in the PR's `FILING.md` so the next
   re-rebase is auditable.
7. Push to the fork, open the PR on GitHub (`gh` not always installed → manual web step).

---

## 1. Know the base; re-rebase before filing

Upstream `sgl-project/sglang` merges ~hundreds of commits/week. A branch prepared even two
weeks ago is stale.

```bash
cd sglang
git fetch upstream
git rev-list --count <recorded_base>..upstream/main      # how far behind are we?
git checkout <prN-branch>
git rebase upstream/main                                 # single-commit branches replay clean
git diff --stat upstream/main..HEAD                      # confirm the diff is what you expect
```

- Before rebasing, **verify the target symbol still exists upstream with the same shape**
  (`git show upstream/main:<path> | grep <fn>`). If upstream refactored it, hand-port — don't
  blindly accept a clean-looking 3-way merge.
- Record in `FILING.md`: base SHA + date, your commit SHA, the prior commit SHA (so a
  force-push history rewrite is traceable). Pushing a rebased branch needs
  `git push --force-with-lease` (history rewrite — get explicit OK first).

## 2. Make it idiomatic — the conventions reviewers enforce

The fork ships machine-readable rules under `sglang/.claude/`. **Read the matching skill
before touching the component** (`sglang/.claude/rules/modify-component-must-read.md`):

| Touching… | Read first | Rule in one line |
|---|---|---|
| any `SGLANG_*` env var / `srt/environ.py` | `env-var-conventions` | Register as an `EnvField` on the `Envs` class; read via `envs.SGLANG_FOO.get()`. **Never** add a raw `os.getenv("SGLANG_...")` / `get_bool_env_var(...)`. |
| `Scheduler` / `TokenizerManager` / `ModelRunner` `__init__` | `large-class-init-style` | `__init__` is an orchestrator; new construction logic → a new `init_<thing>` helper, not an inline block (forks override one piece). |
| speculative-decoding code | `speculative-naming` | Follow the established field/flag naming. |
| adding a test | `write-sglang-test` | Use the templates/fixtures; register under `test/registered/`. |
| sgl-kernel / jit-kernel | `add-sgl-kernel` / `add-jit-kernel` | Kernel add/build conventions. |

### Env-var specifics (most common case — see PR1)

- **Define** in the `Envs` class in `python/sglang/srt/environ.py`, grouped under an existing
  section comment (don't drop it at the bottom of an unrelated block).
- **Typed descriptor**: `EnvBool` / `EnvInt` / `EnvFloat` / `EnvStr` / `EnvTuple`. Use `None`
  default when "unset" must be distinguishable from any concrete value.
- **Access**: `from sglang.srt.environ import envs` then `envs.SGLANG_FOO.get()` (`.get()` is
  mandatory — the bare descriptor raises on `if envs.SGLANG_FOO:` on purpose).
- **Naming verb** carries intent: `ENABLE_` / `DISABLE_` / `USE_` / `FORCE_` / `LOG_` /
  `TEST_` / `DEBUG_` / `OPT_`. **Forbidden**: `DISABLE_FOO = EnvBool(True)` (double-negative
  at the call site). `SGL_*` is a deprecated alias — never add a new one.
- **Tests** override via `with envs.SGLANG_FOO.override(value):` — never mutate `os.environ`
  directly; child processes spawned inside the `with` inherit the override.
- Env var vs CLI flag: user-facing/per-deployment knob → `server_args.py` CLI flag; expert
  toggle / kill-switch / vendor integration → `environ.py`. Don't duplicate across both.

## 3. Format & pre-commit gates (will block the merge)

Run `pre-commit run --all-files` (or at least on changed files) before pushing. Active hooks
(`sglang/.pre-commit-config.yaml`):

- **black** (`black-jupyter`), **isort**, **ruff** (`--select=F401,F821` — unused imports /
  undefined names), **codespell**, **clang-format** (c++/cuda).
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-ast`,
  `check-merge-conflict`, `detect-private-key`, `debug-statements`.
- `check-added-large-files` — **max 1500 KB**. Never commit datasets/checkpoints (gitignore
  big artifacts; the AttnRes super-repo already does this for the 1 GB TASKMIX set).
- `no-commit-to-branch` — don't commit on `main`; always work on a PR branch.
- **`check-no-docs-changes` — changes under legacy `docs/` are REJECTED.** Don't add a docs
  note in the same PR (the old PR1 draft suggested one — would have been blocked). Put serving
  notes in the PR body or a follow-up, not `docs/`.
- `check-chinese-characters` — no CJK chars under `python/sglang/multimodal_gen/` (incl.
  comments). Keep code comments ASCII/English.
- Minimum static check if pre-commit isn't installed: `python -m py_compile <changed .py>`.

## 4. PR shape & body template

- **One commit**, imperative subject prefixed with the area, e.g.
  `[srt/managers] SGLANG_DISABLE_SHM_MM env to force CPU multimodal IPC transport`.
- Base = `sgl-project/sglang:main`; head = `QIU023/sglang:<prN-branch>`.
- Body uses the **official template** (don't leave the HTML-comment sections empty):

```markdown
## Motivation
<the problem; for a crash, paste the actual traceback + one-line root cause>

## Modifications
<the change in 2-5 lines; show the key hunk; state "+N/-M, K files"; note default unchanged>

## Accuracy Test
<N/A for transport/infra-only — say so + why; else real numbers>

## Speed / Profiling
<N/A or benchmark numbers; call out any opt-in slow path>

## Checklist
- [x] pre-commit formatted
- [x] tests (or reproducer) — see below
- [ ] docs (skip — legacy docs/ is rejected by CI)
- [x] code-style guidance followed
```

- **Keep it terse**: lead with crash → root cause → fix → before/after. No marketing prose.
- **Backwards-compat**: if env-gated/opt-in, state "default behaviour unchanged" explicitly.

## 5. Evidence: self-contained reproducer > links > prose

- Best evidence is a **GPU-free, stdlib-only** script the reviewer can run in one command.
  Put it under `examples/runtime/...` or as a registered test. Make it **deterministic** —
  understand the real mechanism (e.g. each framework worker has its *own* `resource_tracker`;
  reproduce it with a standalone-interpreter producer, not a shared one).
- **Verify on Linux** (WSL2 is fine): `/psm_*`, SHM, `resource_tracker`, NCCL, fork/spawn all
  differ on Windows. From this repo:
  `wsl -e bash -lc 'cd /mnt/f/.../sglang && python3 <script>'`.
- Our fork is **public** — you *may* link it for the full stack (the patched branch + the
  downstream harness, e.g. `github.com/QIU023/torchtitan_attention_residual`). Links are
  optional depth; the self-contained repro is the argument.
- **Decouple the justification from our downstream.** Frame the fix by its *general* failure
  mode (e.g. "any parent-managed process tree: Ray / SLURM / Monarch"), not "needed for our
  torchtitan generator". Downstream callers are not upstream dependencies — the PR must stand
  on its own even if our integration never lands upstream.

## 6. CI (after the PR is open)

CI is **label-gated** — it won't run until a maintainer/authorized user adds `run-ci`.
Comment-driven triggers (from `ci-workflow-guide`):

| Command | Effect |
|---|---|
| `/tag-run-ci-label` | add `run-ci` (kick off CI) |
| `/tag-run-ci-label extra` | add `run-ci` + `run-ci-extra` |
| `/rerun-failed-ci` | rerun failed jobs |
| `/tag-and-rerun-ci` | add `run-ci` + rerun failed |
| `/rerun-test <test_*.py>` | rerun specific test file(s) |

- Stages run sequentially A → B → C with fast-fail; a green stage-B/C job whose steps were
  *skipped* means an **earlier** job failed (`check-pr-test-health`) — find the real red X.
- A **label-skipped** run can't be revived by "rerun"; add the missing label to fire a fresh
  event.
- `check-changes` only runs the suites whose package you touched (`main_package`,
  `sgl_kernel`, `jit_kernel`, `multimodal_gen`).

## 7. Filing checklist (per PR)

- [ ] Branch rebased onto **current** `upstream/main`; base SHA + date recorded in `FILING.md`.
- [ ] Target symbol verified unchanged upstream (or hand-ported).
- [ ] Change is idiomatic (matching `.claude` skill applied).
- [ ] `pre-commit` / `py_compile` clean; no `docs/` changes; no large files; comments ASCII.
- [ ] Single commit, area-prefixed imperative title.
- [ ] Body in official template, terse, "default unchanged" stated if gated.
- [ ] Reproducer/test included and **verified on Linux**.
- [ ] `FILING.md` updated: open-PR URL, base/head, commit SHA, prior SHAs.
- [ ] Pushed (`--force-with-lease` if rebased) → PR opened on `sgl-project/sglang`.
