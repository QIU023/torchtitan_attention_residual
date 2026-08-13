# Overnight plan, 2026-08-13

**One hard constraint shapes everything below: 8 GPUs are a single resource.** Two jobs
sharing them polluted a whole line of evidence earlier today, twice. Every GPU item is
SERIAL, each one waits for `pgrep -f torchtitan.train` to return nothing, and no item
starts until the previous one's verdict is recorded.

## Where things stand

Pushed: `attention_residual_dev` at `209ea8783`, `upstream/main` (`65fa556be`) is an
ancestor, three A-step commits on top. PR26's branch `grad-norm-fp32` at `5e88ff897`,
rebuilt on `f4e78188e`, kit re-measured -- ready for a human to file.

Post-merge verification against the frozen pre-merge baseline:

| arm | 13-cell | maxdeg | verdict |
|---|---|---|---|
| multimodal full | 9 SAME, 4 DRIFT 1e-5, 0 BROKE | 5 SAME | PASS |
| multimodal LoRA | 9 SAME, 4 DRIFT 1e-5, 0 BROKE | 5 SAME | PASS |
| text | trains where it trained; ~1e-2, attributed to `#4075` | 2 SAME, 2 BOTH_FAIL, 2 at 5e-4 | drift explained |

TP migration: step A is three quarters done and every piece verified byte-identical.
Step B -- the flip that actually closes the gap -- has not started.

## The queue, in order

### 1. Finish A: the vision tower and the model boundary (GPU, ~40 min)

The only model-side pieces left. `moonvit.py` has 4 plain `nn.RMSNorm` and
`_apply_tp_moonvit_mlp` (92 lines imperative). Swap the norms to the `RMSNorm` wrapper and
declare them; leave `_apply_tp_moonvit_mlp` for step B.

`embed_tokens` / `lm_head` stay OUT of A. Established today at the cost of 29 test
failures: upstream's `Embedding` switches to VOCAB-PARALLEL execution once `parallelize()`
sets `tp_group`, which is a different mechanism from the `RowwiseParallel` the plan applies,
so swapping the class is not behaviour-free. It belongs to B.

**Gate**: `tp2` and `fsdp2_tp2_cp2` byte-identical to `verify_post_merge/mm_full_13.txt`;
CPU suite 381.

### 2. Re-run the 8-GPU cell I interrupted (GPU, ~30 min)

`ep2_fsdp2_tp2_cp2` on the KDA-as-Module tree. It was killed mid-run to free the GPUs for
PR26, so the third A-step piece is verified on `tp2`/`fsdp2` only. Give it `timeout 2400` --
the first attempt reported `rc=124` at 800s, which was my budget, not a hang.

**Gate**: matches `12.04774 11.98894 11.81008 11.50621`.

### 3. Step B: the flip (GPU, ~3 h including the three arms)

Remove the imperative plan and let the declarations drive. All of it at once, because the
migration unit is a **residual stream**, not a module -- verified today: migrating the dense
FFN alone dies with `aten.add.Tensor got mixed torch.Tensor and DTensor`, since declarative
rowwise leaves a DTensor where the still-imperative attention hands out a plain tensor.
AttnRes widens that stream further by injecting two more sources into it.

Order inside the flip, all in one commit:

1. delete the 19 plan entries in `apply_tp_kimi_k3` and `_apply_tp_moonvit_mlp`;
2. swap `embed_tokens` to the `Embedding` wrapper and declare it (vocab-parallel is the
   correct end state, and it is what upstream's own models do);
3. drop the 28 `use_local_output` sites -- the three constraints that justified them are all
   retired (`DECLARATIVE_MIGRATION_2026-08-13.md`);
4. keep `_to_local_if_dtensor` and `_patch_fla_for_dtensor` EXACTLY as they are. They sit at
   the kernel call site, which is the right level, and they are why fla was never the
   blocker.

**Gate**: all three arms via `run_postmerge_gate.sh`, judged by
`compare_to_dev_baseline.py` against `verify_post_merge/` (not `baseline_pre_merge/` --
the text arm's forward baseline moved). Criterion: **no BROKE**. Numbers may move where
a boundary conversion was removed, since that changes summation order; what may not move
is which cells train.

If it fails, the first diagnostic is which cells: a residual-stream mismatch hits every
cell with tp>1, a placement error hits specific combinations.

### 4. Zero-GPU work, for while the above runs

* **The two unjustified core changes.** `common/moe.py`'s DTensor scatter branch and
  `pipeline_parallel.py`'s `local_batch_size` division. Both are either upstream bugs --
  in which case they are clean standalone PRs, like PR26 -- or they do not belong in shared
  files. Decide by reading whether upstream's own models can reach the same failure. This is
  the item most likely to matter for review, because a reviewer sees these before they see
  our model.
* **Three changes that should move out of core**: `distributed/fsdp.py`'s
  `add_zero_valued_dependency`, `tools/grouped_mm_empty_shim.py`, and the
  `optimizer.py` / `lr_scheduler.py` frozen-stage relaxations. The last pair is a real core
  semantic extension with a good motivation (LoRA plus PP), so it wants its own PR rather
  than a move.
* **PR-A's body is now wrong.** It argues a declarative approach "cannot express a
  plain-tensor boundary". After step B that argument is void and the PR shrinks from a
  577-line imperative plan to a declaration set plus the fla shims. Rewrite after B lands,
  not before.
* **`#12` remainder**: 62 comment blocks, ~780 lines.

### 5. Do NOT do overnight

* File any PR. PR26 is ready and that is a human's call.
* Touch `/workspace/vllm_k3` beyond its local branch.
* Start the K3 declarative work that would restructure modules -- we are waiting to rebase
  onto the upstream K3 PR, and restructuring now would conflict with the structure we have
  to rebase onto. Step B does not restructure: it removes an imperative layer and declares
  placements, with no module renamed.

## Two process rules that earned their place today

**Before every GPU verification, prove the tree under test contains the change.** Three
times today a result described the wrong tree: `run_maxdeg.sh` hard-coded `TITAN` while its
sibling honoured the override, edits landed in the dev worktree while `PYTHONPATH` pointed
at the merge worktree, and five maxdeg cells "matched byte-identically" because they were
literally the same tree. A `grep -c` for the new symbol in the tree the run will import
costs nothing.

**A control arm is only a control for what you have checked it shares.** The GRPO merged
arm was used four times as proof about a LoRA-enabled `params_dict` it does not have; a
smoke cell was compared against a matrix cell that passes different batch arguments. Write
down what differs before treating one run as the other's oracle.

---

# Where the TP migration actually stands, end of 2026-08-13

## Branches

| branch | commit | state |
|---|---|---|
| `attention_residual_dev` | `108e55f4b` | **clean**. Step A complete, and its last commit is now verified on `ep2_fsdp2_tp2_cp2` too (`12.04774 11.98894 11.81008 11.50621`, byte-identical). |
| `tp_declarative_refactor` | `88f92d15e` | two modules migrated, `tp2` passes, **`ep2_fsdp2_tp2_cp2` fails**. |

Step A's verification gap is closed: `108e55f4b` had only been checked on `tp2` and
`fsdp2_tp2_cp2`, neither of which has an EP axis, and "step A is verified including the
most sensitive cell" was a claim carried over from `209ea8783`. It has now been measured
directly and it holds.

## What migrated, and what it took

`lm_head` and `embed_tokens` are fully declarative. Getting the FIRST one across required
removing three obstacles that were invisible while the imperative plan covered everything:

1. the driver skipped any subtree already holding a DTensor param, making "weight
   distributed, activation contract never installed" a stable dead state;
2. `KimiMLP` hard-coded one tp intent for two opposite roles (dense FFN is genuinely
   tensor-parallel, a MoE layer's `shared_experts` is replicated) -- 78 of 79 conflicts;
3. a catch-all sweep at the end of `apply_tp` promoted every remaining plain param to
   Replicate, claiming any module removed from the plan before the driver saw it.

Obstacle 2 says something about step A that matters: its cells were byte-identical because
the declarations were INERT. That proves they were not in use -- it does not prove they
were right, and the moment the skip rule changed, 78 of them were wrong.

## The open failure

`ep2_fsdp2_tp2_cp2` fails in BACKWARD with

    RuntimeError: the local_tensor argument only accepts torch.Tensor but got DTensor

`--debug.detect-anomaly` places the forward at
`multimodal_model.py:529, _encode_images_dynamic_cp` -- the vision tower's dynamic-CP
path, which does its own `to_local` + `all_gather` and whose backward now receives a
gradient that is already a DTensor.

Established by measurement, not inference:

* dev at `108e55f4b` PASSES this cell, so the vision-norm declarations are not the cause
  on their own;
* `a96333064` (lm_head plus the three obstacle fixes) already fails it, so `embed_tokens`
  and vocab-parallel are NOT the cause -- an earlier hypothesis with a detailed mechanism
  attached, and wrong;
* the obstacle-3 exclusion cannot be tested in isolation: switching it off makes the
  catch-all promote `lm_head` to Replicate against its Shard(0) declaration, so the run
  dies before it reaches the vision tower. Obstacles 1 and 3 are coupled.

## What to do next, and what NOT to do

**Do not bisect this with 8-GPU training runs.** Five were spent here; one was a port
collision that reads exactly like a failure (`run13_flav.sh` warns about precisely this and
retries -- hand-written torchrun commands do not), one tested a hypothesis that could not
be isolated, and one tested a mechanism I had described in detail before measuring it.

Write a probe instead. `probe_declaration_conflicts.py` listed all 79 conflicts with module
names in a single run, while each guess advanced one layer. The probe this needs prints the
tensor KIND (DTensor or plain) at each step of `_encode_images_dynamic_cp`'s forward on both
trees and diffs them. A first attempt at it fails with "Unknown c10d backend type FAKE" when
run from a detached worktree -- the ParallelDims construction needs the same backend the
trainer uses.

**Take the vision tower out of this migration.** Its TP is a separate 92-line
`_apply_tp_moonvit_mlp`, its CP is a third mechanism with hand-written collectives, and the
text side needs neither. The two are only entangled because step A declared the vision
norms as a convenience.
