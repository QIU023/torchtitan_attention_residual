# The integration tree's K3 GPU tests could not run on this box

Found while pricing what the local 8 x RTX 5060 Ti can do overnight, on `/workspace/tt_4025/torchtitan`, which is a live checkout of `k3_on_4025` at exactly `42691735c`, the 09-18 head that carries the AttnRes recompute and the LoRA stack.

`tests/unit_tests/gpu/test_kimi_k3.py` did not fail, it did not even collect:

    ImportError: cannot import name 'ContextParallelRouting'
      from 'attn_gym.linear.context_parallel'

The control says the environment is fine: `tests/unit_tests/gpu/test_activation_checkpoint.py` is 4 passed on the same tree with the same interpreter, torch `2.14.0.dev20260802+cu130` on driver 580.173.02, so the BFX9 gate that bites Blackwell on newer mains is not what this is.

Two causes, both environment rather than code, and both now fixed.

**attention-gym was on the fork's own branch.** The checkout sat on `pr453` at `7c83f6c` of 2026-09-02, which is upstream's tree at an older base plus one commit of ours, "Add staged delta-rule primitives and a context-parallel recipe for KDA and GDN". Upstream is 45 commits past that base and introduced `ContextParallelRouting` on 2026-09-05 (`9c6913e`, `7418529`). The tree is not ambiguous about which it wants: `pyproject.toml:34` pins `attn-gym[linear] @ git+https://github.com/meta-pytorch/attention-gym.git@main`. Our own commit's content appears upstream as #473, #492 and #499, so the fork branch is behind its own merged work. Every symbol the tree imports resolves on upstream main, checked one by one; `short_conv` became a package there rather than a module, which is why a first pass looking for `attn_gym/linear/short_conv.py` wrongly reported `causal_conv` missing.

**A pinned dependency was not installed.** With the import fixed the suite collected and three tests still failed, on `ModuleNotFoundError: No module named 'renderers'` from `torchtitan/hf_datasets/text_datasets.py:14`. `renderers == 0.1.11` is in `requirements.txt`, `.ci/docker/requirements.txt` and `pyproject.toml:22`; it was simply absent from `/venv/main`.

With upstream attention-gym on the path and `renderers==0.1.11` installed:

    tests/unit_tests/gpu/test_kimi_k3.py      7 passed, 1 skipped
    tests/unit_tests/gpu/test_kda_attention.py 7 passed, 1 skipped

So the K3 GPU surface of the current integration tree passes here, and until tonight nothing on this box could have told anyone that, because the suite stopped at import.

The verification used `git archive upstream/main` into `/tmp/ag_up` and `PYTHONPATH`, deliberately, so that no submodule pointer or branch state changed while the diagnosis was still open. Making it permanent is a separate decision: the fork's `attention-gym` checkout is on `pr453`, the superproject records a different commit again, and moving either is state the user tracks.
