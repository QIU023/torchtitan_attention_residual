# The released-format branch, run where its artifacts are

`k3_released_format` = `65badf428` has been written and unfiled, recorded as blocked on "a box with the disk for the released checkpoint". This box has the artifacts, so the tests were run: worktree `/tmp/wt_x_released` at that commit, `/venv/main`.

    KIMI_K3_RELEASED_DEBUG_DIR=/workspace/k3qat_mm_hf
    pytest -q tests/unit_tests/cpu/test_kimi_k3_released_checkpoint.py \
              tests/unit_tests/cpu/test_kimi_k3_quant_scope.py \
              tests/unit_tests/cpu/test_kimi_k3_vision_preprocess_parity.py

    13 passed, 1 skipped

The skip names itself: `no released index at KIMI_K3_RELEASED_INDEX=None`.

## The recorded blocker is not the real one

Pointing `KIMI_K3_RELEASED_INDEX` at the debug artifact makes that test fail rather than skip, with `AssertionError: the released index carries no packed weights`, which is correct of it: the body says in as many words that the debug artifact is unpacked, 658 plain keys, and the test exists to check that packed keys are refused.

Every index on this box was checked for packed or scaled keys, and none has any:

    k3mini_hf          0 packed/scale of 2333 keys
    k3mini_hf_weights  0 of 946
    k3mini_hf_vllm     0 of 2333
    k3mini_text_hf     0 of 1032
    k3qat_mm_hf        0 of 658

So what is missing is not disk, it is the released **quantized** checkpoint itself, which is not on this machine. The distinction matters because the recorded blocker sends the next session looking for a big box when what it needs is an artifact.

## It does not block filing

The one artifact-dependent test skips with a named reason when the index is absent, which is the shape upstream takes, and the body's Limitations section already says the artifacts are not in the repository. Thirteen of the fourteen tests pass here against the released-layout debug checkpoint.
