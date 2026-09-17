# The two stale local branch refs, aligned onto origin (2026-09-17)

Both forks carried a local branch ref weeks behind its origin counterpart, which is what sent an earlier pass in this session chasing work that had never been lost. Both are now aligned, and the old generation is preserved rather than discarded.

## Why the operation is a reset and not a rebase

A rebase would replay the local commits onto origin. Every one of those commits already exists on origin in content, under a different hash, because the integration trees are rebuilt and rebased repeatedly. Replaying them would produce duplicates and a long conflict chain and would end at the same tree. The operation that achieves the intent is to point each local ref at its origin counterpart, after proving that nothing unique is lost.

## verl: `kimi_k3_integration_rebased`, `3dfd0430` to `7db90d7a`

Proof before moving: the local tip holds no file that the origin head lacks, and the head's log contains the local tip's own subject exactly once, so `3dfd0430` is the pre rebase copy of a commit already on origin's line. The branch was checked out in `/tmp/wt_verl_new`, whose only dirty entry is an untracked directory that a hard reset leaves alone. Zero tracked changes were at risk.

## torchtitan: `k3_on_4025`, `9f87e0891` (2026-09-03) to `caa7c014a`

This ref is not a branch that fell behind. It is an older generation of a tree that has been rebuilt several times (0910, 0912, 0915, 0916, 0917), so the comparison needs care.

The first classification pass was wrong and is worth recording as a trap. Restricting the diff to `torchtitan/` reported 25 files as absent from the current tree, including the whole `models/kimi_k3/tests/` directory. Model local tests have since moved to `tests/unit_tests/cpu`, which sits outside `torchtitan/`, so every migrated test necessarily read as lost. Redoing the comparison against the whole tree, and falling back to a symbol search when the basename changed, resolved 34 of the 46 local only files as moved:

    torchtitan/components/lora.py              -> config/transform/lora.py and models/common/lora.py
    torchtitan/components/metrics.py           -> observability/metrics.py
    torchtitan/components/quantization/*       -> torchtitan/quantization/*
    torchtitan/models/kimi_k3/pipeline_adapter.py -> pipeline_stage.py (RankLocalCache survives)
    torchtitan/models/kimi_k3/vision_preprocess.py -> hf_datasets/multimodal/utils/image.py
    torchtitan/models/kimi_k3/tests/*          -> tests/unit_tests/cpu/test_kimi_k3_*.py

Of the rest, `components/quantile_balance.py` is the closed #4412 implementation that #4577 replaced, `experiments/forge/` and the mxfp8 doc are upstream files upstream removed, and the attention residual checkpoint test's successor lives on the `k3_ac_reuse_attention` PR branch.

## What only the tag holds

The 09-03 tip is preserved as the annotated tag **`k3_int_20260902_pre_rebuild`** (`737298052`), pushed to the fork, because the standing rule reads missing pieces from the old tree instead of rewriting them. Nothing else pointed at that line. Five items live there and nowhere else:

    models/kimi_k3/tests/test_cp_contracts.py
    models/kimi_k3/tests/test_pp_fqn_injection.py
    models/kimi_k3/tests/test_vit_stage_shares.py
    the kimi_k3_debugmodel_rl_mx_qat_vit1 flavor (9fde3e5ad, 8 lines in config_registry.py)
    distributed/minimal_async_ep/kernels.py

The flavor is the one with a live consumer: `matrix_scripts/verl_grpo_qat_ladder.sh` still names `kimi_k3_debugmodel_rl_mx_qat_vit1`, and the current tree has no vision on one stage flavor under any name. Restoring it is an eight line cherry pick from the tag. I left that decision alone, since adding a flavor to the upstream bound tree is a separate call from aligning a ref. The `minimal_async_ep` backend is gone from the current tree entirely (zero files, against eleven for `deepep`), so its kernels file is part of a backend that was dropped, not an accidental loss.

## Two scripts whose behaviour changed

Each branch was checked out in a worktree, so moving the branch moved the worktree's tree with it. Two scripts read those paths in active lines rather than in comments:

    matrix_scripts/verl_grpo_newtree.sh   PYTHONPATH and cd on /tmp/wt_verl_new, now 7db90d7a instead of 3dfd0430
    matrix_scripts/run_4025_matrix.sh     TITAN defaults to /workspace/tt_4025/torchtitan, now caa7c014a instead of the 09-03 tree

Both now run against the current tree, which is the direction the alignment intends, but the change is silent, so it is recorded here. Either worktree can be put back on its old commit with a detached checkout if an old run has to be reproduced: the tag for torchtitan, `3dfd0430` from the reflog for verl.
