# Filing notes: attention residual recompute (PR 32)

Branch `k3_attnres_recompute` = `9f6bae06f`, two commits on `upstream/main` `68c97b0c5`. Fork remote `origin`. Not filed yet.

    1436053ea  aggregate the block residual without retaining FP32 copies
    9f6bae06f  zero initialise the attention residual projections

The second commit is the initialisation correction. The residual projections were drawn from `trunc_normal_` with std 0.02, so the initial depth weights were arbitrary; the report requires them uniform at initialisation. It is a correctness fix against a stated requirement, so it carries no performance evidence and none is owed. Its behaviour is pinned by tests: at zero the aggregation returns the mean of its sources exactly, the projection still receives a gradient, and the norm weight does not until the projection leaves zero.

## Why this is not part of PR 4312

The aggregation arrived with #4025 and sits on main today. Its three call sites carry no pipeline guard and the model file imports nothing from the pipeline, the stage or the layout, so the two retained FP32 tensors exist at dp1 on a single card, under TP and under EP. 4312's own change to `model.py` is +40/-23 and is a contract adaptation: `prefix_sum_TD` becomes an optional `partial_block_TD`, the concatenation becomes a ternary, the block boundary concatenation moves, and call site variables are renamed. The softmax, the upcast, the inverse RMS and the matmul are untouched. Filing this inside 4312 would move a review diff that is waiting on a reviewer and would add an argument unrelated to pipelining.

## Diff audit

Run against `git diff upstream/main..HEAD` before calling the draft ready.

- No logbook path, no cross repository reference form and no `.md` link in any source file.
- No measured value, benchmark result or experiment record in any docstring. An earlier grep appeared to find some; those were substring matches on `Platforms` and `inverse_rms`, not numbers.
- Added lines in `torchtitan/`: 94 code, 1 comment, 4 docstring, so docstrings are 4.0% of the change. #4577, the house reference, is 2.4%. The class docstring was cut from six lines to four after the first count read 5.9%.
- One new private definition, `_AttentionResidualAggregation`, with its forward and backward.

## Reuse check

Searched `models/common`, `components`, `distributed`, `quantization` and the sibling models for an existing depth softmax aggregation or a reusable recomputing wrapper. There is none. What core does have is the pattern this follows: `torch.autograd.Function` appears in `components/loss.py`, `models/common/linear.py`, `models/common/dist_gemm.py`, `models/common/aux_loss.py`, `models/gpt_oss/moe.py` and the quantization modules, and `@once_differentiable` is already used by `models/common/linear.py`, `overrides/fused_mla.py` and the MXFP8 linear. The SPMD registration follows `models/common/linear.py`, which is the one of those that registers.

## Evidence

Results in the body are the H100 sweep, one shape per process, 10 warmups and the median of 7, from the debug flavor's shape up to the released model's hidden size and block count at four context lengths. The 5060 was used for development and agrees in direction; none of its numbers are quoted, per the rule that body numbers come from the H100.

## No regression against unmodified main

The same CPU suite was run on a detached worktree at `upstream/main` with no changes, as the only honest baseline. Numbers on the left are that baseline and on the right the branch as it stood with three tests added:

    passed 1012 against 1015, failed 19 against 19, errors 7 against 7, subtests 83 against 85

The nineteen failures are the same files on both sides (varlen attention, qwen3_5 mrope, quantization, model td layout, config manager, aux loss, rope, embedding, cp attention) and none is a Kimi K3 test. The seven errors are the same on both sides and are missing packages in this environment, six for `transformers` and one for `torch_checkpointing`. Collection was confirmed directly rather than inferred from counts: the baseline collects 1041 and the branch 1044, and the branch's new file is named in the collection list while the baseline's is not.

## Where the zero init has been before

Added 2026-08-24 as `d54d327a9`, reverted the same day as `53b613d80`. The revert was on scope, not on merit: the modules are upstream's, so a pipeline or context parallel branch had no business changing how they initialise, and the revert message says the observation should reach the maintainers as a question instead. This PR is that occasion. `d54d327a9` now survives only on `k3_on_4025_pre_rebase_0824`.

## The context ceiling, and what it does not say

On one H100 of 79.18 GiB, `dim` 7168 with a stack of 8, one process per point: the naive form completes at 32768 tokens with a 52.06 GiB process peak and runs out of memory at 65536; this change completes at 65536 with 29.77 GiB and at 131072 with 59.53 GiB, and runs out at 262144. So the reachable context per rank goes up four times.

Two things that must not be claimed from it. The released default is 262144 tokens per micro-batch per dp rank and neither form reaches it on one card, so this is not "the real context now fits"; that configuration is sharded by context, tensor and pipeline parallelism and `tokens` is counted before any of it. And these peaks are process totals, unlike the increment-over-baseline peaks in the two sweep tables, so the same shape reads 52.06 GiB here and 48384 MiB there. Both are correct and they are not interchangeable.

## Step counts

Earlier runs in this work used 3, 5, 8 and 10 steps with no reason for any of them. The repository rule is steps 1, 10 and 20, and no step past the point where the reference starts memorising the debug set. Nothing in the body reports a step count now; if one is ever needed it follows that rule.

## Open

- A model level step 1 gradient comparison. The Attention Gym KDA kernel accepts only CUDA capability 10.0 and 10.3, so a full Kimi K3 step runs on neither the H100 (SM90) nor the 5060 Ti (SM120) without relaxing that guard, and a number produced under a relaxed guard is not reproducible from an unmodified tree. The body says so rather than omitting it.
- The full CPU suite on this branch, for the test plan.

## Filed

#4780, 2026-09-18. Branch `k3_attnres_recompute` at `9f6bae06f`; the two commits are the aggregation and the zero initialisation. Numbers in the body are the H100 ones, per the rule that the 5060 only smokes.

## 2026-09-24: tianyu's review answered on `attnres_review1` (not pushed to the PR branch)

tianyu-l (issuecomment-5807507932): what problem does the PR solve, in simple terms; the autograd Function's many
small ops look slow; use something like fla's `fla/ops/attnres/fused.py` through the override mechanism. The bot
review (5807512687) agreed and spelled the route out: keep the eager form as the default, add a `Configurable`
node the override can target, put the fused kernel behind `torchtitan/overrides/`, land the zero init on its own,
and show training-level peak memory and throughput.

Review branch `attnres_review1` = `e0d441c62` on main `b64103072`, pushed to the fork only (`k3_attnres_recompute`
stays `9f6bae06f`; the user force-pushes when the H100 numbers are in). Three commits:

1. `f6d78ee24` kimi_k3: zero initialise the attention residual projections (the PR's commit, unchanged).
2. `0a16e90f0` kimi_k3: the attention residual aggregation is a Configurable. `AttentionResidual(Function[Tensor])`,
   parameter-free, default `__call__` = main's eager FP32 form; the block gets `attention_res_fn` / `ffn_res_fn`
   and the model `output_res_fn`, all `field(default_factory=AttentionResidual.Config)` so `__init__.py` is
   untouched. The weights stay `attention_res_norm` / `attention_res_proj` etc., so the state dict adapter,
   `sharding.py` and every PP split in the 4312 stack are untouched (the bot's variant, wrapping norm + proj in a
   module, would have renamed every FQN). The hand-written `_AttentionResidualAggregation` is gone.
3. `e0d441c62` overrides: Kimi's attention residual fused with flash-linear-attention.
   `torchtitan/overrides/fused_attnres.py`: `FusedAttentionResidual` calls `fla.ops.attnres.fused_attnres(query,
   [*stack.unbind(1), prefix_sum], norm.weight, rms_eps=eps)`; fla optional (helion_rope's import-sentinel
   pattern), falls back to eager without fla or off CUDA; `@override(target=AttentionResidual.Config, exact=True)`.
   `pyproject.toml` lists `fla`, `fla.*` in pyrefly's replace-imports-with-any.

fla's op is Kimi's aggregation exactly: `naive_attnres` = RMSNorm(v, w) scored against `query`, softmax over
sources in FP32, weighted sum of the raw sources, one downcast. `fla.ops.attnres` exists only on fla main
(0.6.0 dev; PyPI 0.5.2 has no `fla.ops` at all); installed with
`pip install --no-deps --target <dir> git+https://github.com/fla-org/flash-linear-attention.git@main`.
The kernel reads sources as contiguous `[T, D]` rows (`_build_ptr_table`, `D` constexpr), so the `[T, N, D]`
stack's per-block views are copied once per call; a stride-aware kernel would be a fla-side follow-up.

SPMD type checking: three attempts. (a) `register_local_autograd_function(FusedAttnresFunction)`: the checker's
local rule runs the Function's forward on META tensors to infer types, and fla raises "Triton attnres requires CUDA
tensors". (b) `spmd.register_decomposition(FusedAttnresFunction, _eager_attnres)`, the eager math with the op's
signature and output tree: the checker traces the decomposition, the kernel runs for real; but the decomposition
returned `o.new_empty(0)` for the unrequested probabilities and a fresh tensor types as R against the local V
("Local SPMD produces V on axis mesh_dp but DTensor produces R"); returning the probabilities transposed instead
failed with "PartitionSpec length 2 doesn't match tensor ndim 1" because the real op's placeholder is 1-D.
(c) final: the second output is `probs.T` when `return_weights` else `scores.logsumexp(-1)` (1-D, derived, never
fresh). Registered for the Triton and, when importable, the Gluon Function classes. The 4-GPU
`kimi_k3_debugmodel_mm` cell (typechecking=True) runs with the override: 49 nodes replaced, 3 steps.

Numbers on the 5060: fla fused vs eager on random inputs, forward and all four gradients: bf16 rel 2e-5 / 3e-5
(max abs diff 3.9e-3 = one bf16 ulp at magnitude 1), fp32 rel 1e-7; zero init gives the mean and no norm grad.
`test_fused_attnres_override.py` 4 passed on CUDA with fla; CPU suite 43 passed 2 skipped (incl. test_override,
test_no_new_cli_options, all K3 CPU tests); pyrefly 0 errors on the four changed files; ufmt clean.
mm cell, seed 42 deterministic, 3 steps: eager loss 12.58970 / 11.03258 / 9.29013, fused 12.58621 / 11.05843 /
9.47570; peak memory step 1 7.75 vs 6.99 GiB. Not an identity claim: 49 bf16 roundings per step plus this box's
MoE near-tie sensitivity (see memory k3-step10-spread); the H100 kit runs eager twice for the noise floor.

H100 kit `kit_h100_2026-09-24/`: `probe_attnres.py` (flavors `attnres_debug` = stock debug on AdamW, `attnres_wide`
sized by ATTNRES_DIM/LAYERS/BLOCK, e.g. 4096/16/2 for a stack of 8), `run_attnres_h100.sh` (eager, eager2, fused
per token count, 20 steps seeded), `tables_attnres.py` (loss 1/10/last, peak GiB, tps). Drafts:
`REPLY_4780_TIANYU_2026-09-24.md` (one comment, PASTE markers, NUMBERS to fill), `PR_BODY_v2_2026-09-24.md`.

## 2026-09-24 round 2: tianyu's review r4090059193 (CHANGES_REQUESTED), handled locally, PR branch untouched

tianyu-l: "fix the init in its own PR"; "use torch_remat to solve the recomputation issue ... instead of introducing an autograd function"; "make `_apply_attention_residual` a configurable function ... with this reference impl as default, but maybe use torch.compiled version (or even call / copy FLAs' fused impl) as override ... (can leave to @acisseJZhong)". The user: leave the compile part untouched, re-evaluate how it depends on AC reuse (#4656), push nothing to a published PR branch.

Branches (fork only; `k3_attnres_recompute` stays `9f6bae06f`, #4656's `k3_ac_reuse_attention` stays `7e9622a22`):

- `k3_attnres_zero_init` = `db483314a`, new, one commit on main `9e159aed7`: the zero init plus `tests/unit_tests/cpu/test_kimi_k3_attention_residual_init.py` (3 tests; the init test fails on main as a control). Commit message now cites Section 5 of the Attention Residuals technical report (arXiv 2603.15031, "all pseudo-query vectors must be initialized to zero"); the earlier "Kimi Linear report" attribution was wrong (that paper's Section 5 says its architecture is identical to Kimi Linear). Body draft `PR_BODY_zero_init_2026-09-24.md`; the PR is not opened.
- `attnres_review1` = `4e4baa4f3` on main `9e159aed7` (was `e0d441c62` on `b64103072`), three commits:
  1. `83f05cf98` the Configurable (init tests moved out; the config-node test stays).
  2. `38fcdde4a` torch_remat recompute: #4656's `remat.checkpoint` wrapping ported onto the Function (`_checkpointed_attention_residual(name, aggregate, ...)`, block flag `checkpoint_residual`, output aggregation always checkpointed), marking moved from `parallelize_kimi_k3` (gone after #4810) into `KimiK3Model.parallelize` before the policy is built. Tests in `test_kimi_k3_attention_residual.py`: #4656's three plus RegionAC composition and the parallelize marking (fake PG, meta model), 6 passed; controls: without the port 5 fail, without the marking the parallelize test fails.
  3. `4e4baa4f3` the fla override, patch byte-identical to `e0d441c62`'s (compile part untouched).

Checks: CPU 50 passed, 2 skipped (override, residual, test_override, test_skip_dp) in `/venv/main` with fla 0.6.0 on `PYTHONPATH` and in `/workspace/venv_bfx9` without fla; override test 4 passed on CUDA with fla; `tests/unit_tests/gpu/test_kimi_k3.py` 3 passed 1 skipped on main and head alike; pre-commit on the changed files passes (pyrefly and lychee skipped there); pyrefly 0.45.1 (the pinned hook version, `scratchpad/pyrefly045`) gives the same 2 errors as main (`torch_checkpointing` in this env) and `--remove-unused-ignores` changes nothing. Local pyrefly 1.2.0 rewrites 25 files repo-wide on main as well: environment, reverted file by file (never the KDA guard lift).

Dependency evaluation: `DEPENDENCY_override_vs_ac_reuse_2026-09-24.md` (verdict, per-call bytes, the 13-cell 5060 smoke, torch_remat facts, options). Short form: complementary; the checkpoint belongs at the call site; fla's `ctx.res` makes fused-under-checkpoint pure overhead until fla rebuilds the table in backward; #4656 superseded by commit 2 (option A recommended, the user decides).

5060 smoke (13 cells, stock `kimi_k3_debugmodel`, one shared warm cache lineage): head bitwise to main 10/10 steps under none, selective, full and region AC; the AC-off peak drops 13.22 to 12.79 GiB; RegionAC runs (the flag works end to end). The fla override rows differ from eager from step 1 (loss 8e-4, grad norm two bf16 steps), identically across AC modes: not located, held back from every draft.

Diff audit (`git diff upstream/main attnres_review1`): no logbook path, no measured value in code, no private helper with a docstring in commits 1 and 2 (the block helper's docstring was dropped). New private defs: `_checkpointed_attention_residual` and `KimiK3TransformerBlock._attention_residual`, both #4656's, nothing in `models/common` or `distributed` wraps a call in a remat checkpoint conditionally. Left as is in the untouched override commit, for its owner: the module docstring says the op "saves per-token statistics instead of the FP32 stack" but it also keeps a bf16 copy of the stack (`ctx.res`); `_eager_attnres` is a private helper with a two-line docstring; the import sentinel catches only `ImportError` (in `/venv/main` fla-core 0.5.1 raises `AttributeError` from tilelang, so the module fails to import there).

H100 kit, rewritten (`kit_h100_2026-09-24/`): `run_attnres_h100.sh` runs main against the head, AC off and selective, plus the fused rows, one GPU per cell, every cell of a token count on a copy of one warm cache lineage, `main_none_rerun` as the noise floor; `lift_kda_guard.py` admits the local capability in both checkouts; `tables_attnres.py` counts equal steps against the reference; `probe_saved_bytes.py` (per-call bytes, `--time` for the op timing, `--fla-res-from-saved`). Dry run on the 5060 (STEPS=3, TOKENS=512): all 9 cells rc=0, head 3/3 equal, AC-off peak 14.61 to 14.17 GiB with the kit's AdamW flavor, the same pair #4656 measured on the H200. On the H100 box (see memory h100-box-217-18-55-200-env; `~/tt` holds uncommitted MoonEP work, so only fetch into it and add worktrees; add an `upstream` remote first if it has none):

    git -C ~/tt fetch origin attnres_review1 && git -C ~/tt fetch upstream main
    git -C ~/tt worktree add --detach ~/tt_attnres origin/attnres_review1
    git -C ~/tt worktree add --detach ~/tt_main upstream/main
    ~/venv_k3/bin/pip install --no-deps --target ~/fla_main git+https://github.com/fla-org/flash-linear-attention.git@main
    MAIN=~/tt_main HEAD=~/tt_attnres VENV=~/venv_k3 FLA=~/fla_main OUT=~/results/attnres_r2 FLAVOR=attnres_debug \
      TOKENS="512 4096" STEPS=20 GPUS="0 1 2 3" nohup bash <kit>/run_attnres_h100.sh > ~/results/attnres_r2.log 2>&1 &
    ~/venv_k3/bin/python <kit>/probe_saved_bytes.py --time   # from ~/tt_attnres, PYTHONPATH=~/fla_main:.

Drafts: `REPLY_4780_TIANYU_r4090059193_2026-09-24.md` (one PASTE block, `#INIT_PR` and `NUMBERS` to fill, #4656 sentence conditional), `PR_BODY_v3_2026-09-24.md`, `PR_BODY_zero_init_2026-09-24.md`. `REPLY_4780_TIANYU_2026-09-24.md` and `PR_BODY_v2_2026-09-24.md` are superseded.

Waiting on the user: open the init PR; force-push `attnres_review1` to `k3_attnres_recompute`; keep or close #4656; run the H100 kit; whether the fla override stays in this PR or goes to acisseJZhong.

## 2026-09-24 round 2b: the fla override leaves 4780 (the user's decision)

The user: tianyu named @acisseJZhong for the override, so this PR does not take that part. `attnres_review1` reset to `38fcdde4a` (Configurable, torch_remat) and force-pushed to the fork; `k3_attnres_recompute`, `k3_ac_reuse_attention` and `k3_attnres_zero_init` unchanged. The override commit `4e4baa4f3` survives only as `fused_attnres_override_4e4baa4f3.patch` here (`git am` onto the review branch restores it). Model code is identical between the two heads, so the round-2 smoke's main against head rows hold. Rechecked on `38fcdde4a`: CPU 48 passed (residual, test_override, test_skip_dp) in `/workspace/venv_bfx9`, `tests/unit_tests/gpu/test_kimi_k3.py` 3 passed 1 skipped, pre-commit passes on the changed files, pyrefly 0.45.1 the same 2 environment errors as main and no hook edits.

Kit: `run_attnres_h100.sh` has no fused cells and no `FLA` variable any more (five cells per token count: main, main again, head, each AC off, plus main and head under selective AC). `probe_saved_bytes.py` calls `fla.ops.attnres.fused_attnres` directly when fla imports (any import failure skips the fla rows) instead of the removed override module. Reply and body v3 updated: the override is left to @acisseJZhong; the one sentence on `ctx.res` in the reply is optional.
