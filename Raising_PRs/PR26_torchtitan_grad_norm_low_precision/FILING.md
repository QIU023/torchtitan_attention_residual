# PR-26 filing: execution doc

**Target**: `pytorch/torchtitan` -- new PR, no dependency on any other PR in this batch.
This one is not gated on the upstream K3 PR: it touches core only, has no model code, and
reproduces on `llama3_debugmodel`.

## Freshness check, 2026-08-12

Re-verified against `upstream/main` at `65fa556be`, three days newer than the
`681fd4b50` the kit was written against:

| check | result |
|---|---|
| `git apply --check grad_norm_fp32_PR26.patch` | clean, no offsets |
| `python -m compileall torchtitan/distributed/utils.py` | ok |
| `pytest tests/unit_tests/test_distributed_utils.py` | 4 passed |

That last file is NEW upstream (landed in the 17 commits since the kit was cut) and worth
knowing about: it is 63 lines and covers none of `clip_grad_norm_`, so it neither conflicts
with this patch nor makes it redundant. Upstream's own changes to `distributed/utils.py` in
that window (+60/-9) were to `_dist_reduce` / `init_fake_mode`; `clip_grad_norm_` is
untouched.

**Re-measured 2026-08-13 on `f4e78188e`**, four arms, and the table in `PR.md` now
carries those numbers. The float32 baseline moved (1.4485 -> 1.4509) because `#4075`
(document-boundary loss masking) and `#4099` (valid-token counts on device) both landed
since `681fd4b50` and both touch `llama3_debugmodel`'s `c4_test` path -- upstream
regenerated its own golden loss files in `#4075` for the same reason. bf16-before did NOT
move (1.4453 both times), which is itself a symptom: three to four significant digits
absorb the change.

The 394-tensor partition-dependence figures need no re-measurement -- they norm tensors
directly and never touch the loss.

## Branch preparation

    cd <a fresh clone or worktree of upstream/main>
    git checkout -b grad-norm-fp32 upstream/main
    git apply <this folder>/grad_norm_fp32_PR26.patch
    # commit as a FRESH commit -- do not cherry-pick from the fork
    git commit -am "clip_grad_norm_: compute the total norm in float32 so it does not depend on the parallelism partition"

The fresh-commit rule is not cosmetic: 8 of 56 fork commits in this area carry
`pytorch/torchtitan#N` / `Refs:` trailers from before the no-cross-reference rule, and
cherry-picking one would fire a timeline event on a third-party PR (finding #11).

## Body

`PR.md`'s "Suggested PR body" section verbatim. Title:

    clip_grad_norm_: compute the total norm in float32 so it does not depend on the parallelism partition

## What a reviewer will push back on, and the answer

* **"This changes reported numbers."** Yes, and only when `training.dtype=bfloat16`.
  Default is float32, where the patch is a no-op -- state that in the first paragraph.
* **"Is the cost real?"** The norm-of-norms is over a few hundred scalars; the cast is on
  per-tensor norms, not on gradients. Include the timing if asked, do not lead with it.
* **"Why not fix `get_total_norm` in pytorch?"** Reasonable, and worth offering: the torch
  API returning the input dtype is defensible on its own terms. The torchtitan-side fix is
  the one that makes runs partition-independent today.

## Pairs with

The `pp_mesh` dtype fix in the same function (`commits.md`). Ship both in one PR -- they
are the same bug seen from two directions, and splitting them makes each look arbitrary.

## Cross-links

None. Do NOT reference the upstream K3 PR or any third-party issue in the commit message;
if a link belongs anywhere it is a PR comment we choose to post.
