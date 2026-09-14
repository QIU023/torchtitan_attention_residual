# How to write a PR body for these upstream repos

Written 2026-08-13, after a maintainer replied to PR26 with:

> sorry I couldn't really understand the PR summary which seems to be written by AI.
>
> Before vs. after this change, what ops that are used to performed in bf16 are not
> performed in fp32?

He is right, and the second sentence is the lesson: **the thing he had to ask for was the
one thing the PR was about**, and it was not findable in the body we gave him.

## The rule

A PR body answers, in this order, and stops:

1. **What changes.** The ops, the call sites, the flag. Concretely enough that a reviewer
   can predict the diff before reading it.
2. **Why it is wrong today.** One mechanism, one sentence.
3. **Evidence.** A table of numbers, or a repro command. No prose around it. One row per
   configuration, and report steps 1, 3 and 10 -- a step-1 column alone shows that a
   configuration started, not that it trains the same model (user, 2026-08-25).
4. **What does not change.** One line, when there is a default configuration people will
   worry about.

The body describes the branch being filed and nothing else. A paragraph about work that
lives on a different branch is the same failure as a paragraph about design history.
English only; a Chinese note to ourselves at the top of a paste-ready body ships.

Everything else -- design history, alternatives considered, pre-empted objections,
reassurance about scope -- belongs in the linked issue, a review comment when asked, or
nowhere.

## The failure mode, concretely

PR26's body had: a status line with three clauses, a summary, a numbered list of
consequences, an evidence table, a partition-dependence experiment, a repro command, a
"what a reviewer will push back on" section with three anticipated questions, and a
harness note. Every paragraph was true and individually defensible.

The maintainer's question -- **which ops move from bf16 to fp32** -- has a two-line answer
that was never stated as such:

    per-tensor norms:  torch._foreach_norm(grads, 2)              bf16 -> fp32
    norm of norms:     vector_norm over the stacked per-tensor norms   bf16 -> fp32

Everything else in that body was context around an answer it never gave.

## Tells that a body reads as machine-written

Each of these was present in PR26's:

* **Section per thought.** Nine headings for a 45-line patch.
* **Every claim carries its justification inline.** A reviewer reads a claim and decides
  whether to ask; supplying the defence up front doubles the length and signals that no one
  expected to be asked.
* **Pre-empting objections.** "Expect the question X. Answer: ..." Wait for the question.
* **Symmetry and hedging.** "Not X -- Y", "real but moderate", parallel clause pairs.
  Natural in analysis, conspicuous in a bug report.
* **Restating the patch in prose.** The diff is on the page.
* **Numbers embedded in sentences** instead of a table with a one-line caption.

## What to do instead, for the same patch

    clip_grad_norm_ computes the total norm in the gradients' dtype, because
    torch.nn.utils.get_total_norm returns the input dtype. With
    training.dtype=bfloat16 both the per-tensor norms and the norm-of-norms are
    bf16, so the total is wrong by a few tenths of a percent, and by an amount
    that depends on how the tensors are grouped, which under PP or EP means it
    depends on where the model was cut.

    This computes those two reductions in fp32. Gradients stay bf16; the clip
    multiply is unchanged; float32 runs are bit-identical before and after.

    llama3_debugmodel, 4 GPUs, dp_shard=2 x pp=2, seed 42, deterministic:

    | dtype    | patch  | step 1 | step 2 |
    |----------|--------|--------|--------|
    | float32  | before | 1.4509 | 1.6336 |
    | float32  | after  | 1.4509 | 1.6336 |
    | bfloat16 | before | 1.4453 | 1.6328 |
    | bfloat16 | after  | 1.4508 | 1.6343 |

Ten lines. Same information a reviewer needs to decide.

## No dashes (user, 2026-09-13)

Paste-ready text (PR bodies, review replies, comments) uses no dash as punctuation: no em dash, no en dash, no spaced double
hyphen (` -- `). Write a period, comma, colon, semicolon or parentheses instead. Before a draft is handed over, grep its paste
section for ` -- ` and the two dash characters. Hyphens inside words, flags and numbers (`pre-norm`, `--training.steps`, `-0.24%`) are fine.

## Reference format: #4577 (user, 2026-09-14)

The maintainers' own quantile-balancing PR (pytorch/torchtitan#4577) is the format and the code style to copy; our #4412 for the same feature was closed in its favour. Its body, in full structure:

    ## Summary
    Implement <feature> for <model> as described in <report section>.
    - Add <reusable component> to <common module>. <one sentence on what it does>.
    - Add <hook / coordination> next to <the existing hooks of that kind>.
    - Wire the shared components into <model> for all registered flavors.
    - Add a <n>-rank GPU test covering <what the real collective exercises>.

    ## Design
    <The mechanism, two or three sentences.>
    <One paragraph: where each piece lives and why it lives there.>

    ## Relation to #N          (only when it replaces or overlaps another PR)

    ## Test plan
    - `pytest <file> -q` (`1 passed`)
    - Scoped pre-commit checks, including formatting, lint, and Pyrefly (`passed`)

No changed-files block and no numbers table unless the claim is numerical (then the numerics rules in CLAUDE.md apply). The code side of the same standard (survey the module structure first, reuse titan's seams, no hooks or patches without a missing seam, one-line docstrings, comments only for invariants; #4577 has 18 comment and docstring lines per 398 code lines, #4412 had 111 per 381) is in CLAUDE.md, "#4577 is the reference for code and body".

## Applies to every kit in this folder

Before filing anything from `Raising_PRs/`, cut the body to the four items above. The long
version stays in the kit as `commits.md` / `FILING.md`, which is where discovery, measured
percentages, and the reasoning belong -- those files are for us.
