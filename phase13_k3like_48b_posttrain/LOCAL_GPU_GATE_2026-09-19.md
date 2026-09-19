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

## Resolved by removing the submodule

The user's call: drop attention-gym from this repo entirely and take upstream from the venv, which is what `torchtitan/pyproject.toml` asks for in the first place.

Done in that order, so the box was never left with neither working: upstream installed first (`pip install "attn-gym[linear] @ git+https://github.com/meta-pytorch/attention-gym.git@main"`, which replaced the editable `.pth` that pointed into the submodule), the K3 GPU tests re-run against it with no `PYTHONPATH` at 14 passed and 2 skipped, and only then `git submodule deinit` plus `git rm` and the `.gitmodules` section.

Two checks before the removal. `/venv/vllm_k3`, the veRL environment, has no `attn_gym` at all, so nothing there depended on the path. And the branch the checkout sat on existed **only locally**: the fork carried `main` and `kda-fla-backend` and not `pr453`, so it was pushed to the fork first. It turns out not to be this line's work at all, `7c83f6c` is authored by drisspg and `git cherry` puts it outside upstream main, so `pr453` is an unmerged upstream PR branch that had been checked out and left there.

After the removal the suites still read 14 passed, 2 skipped. `.git/modules/attention-gym` is deliberately left on disk: it costs a little space and holds the clone, and deleting it buys nothing.

## The whole CPU suite on the integration tree

With the two gates above cleared, the full suite runs for the first time on this box: **1211 passed, 18 failed, 34 skipped** in 16.5 minutes on `k3_on_4025` at `42691735c`.

Getting it to collect at all took two more environment fixes, both the same shape as the first two.

`verl` was editable-installed into `/venv/main`, and `verl/scripts/__init__.py` makes it a regular package, so `import scripts` resolved to veRL's rather than torchtitan's and `test_download_hf_assets.py` and `test_tokenizer.py` could not import. veRL's own work lives in `/venv/vllm_k3` and `/workspace/venv_verl`, so the install in `/venv/main` was a leftover and was removed.

`torch_checkpointing` is pinned in `pyproject.toml` and was not installed. Installing it is not enough: the tree imports `torch_checkpointing.default_resharder`, and PyPI's only release, 0.1.0, does not have it, so `test_torch_checkpointing.py` is excluded from the run. That dependency is also absent from `requirements.txt` and from `.ci/docker/requirements.txt`, so the same file would fail in CI unless CI takes it from somewhere else.

The 18 failures are all optional dependencies or access, not the tree: `flash_attn` is not installed and accounts for 31 of the import errors in the log, `torchao` is 0.17.0 and lacks the 32x32 swizzle cast kernels the MXFP8 tests import, and one tokenizer test needs the gated `meta-llama/Llama-3.1-8B` repo.
