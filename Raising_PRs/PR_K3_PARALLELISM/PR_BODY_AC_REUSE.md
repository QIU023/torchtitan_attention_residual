# PR title: [Kimi K3] Recompute the attention-residual math in backward

For PR 4656, repurposed (user, 2026-09-15, option a): retitle 4656 to the line above, replace its body with the paste section, and sync `k3_ac_reuse_attention` to review branch `ac_review3` = `37d595a47` (the user re-authored `be311737f` + `7ec30e5a9` as `cc2031a37` + `37d595a47`, one line less: the `from spmd_types import SpmdType` line the import-hunk resolution had kept is gone, main's `model.py` no longer imports it and nothing used it; two commits on main `810e62786`, rebased 2026-09-16 on the user's word after TP and quantile balancing merged: the recompute `be311737f` = the old `fd0a9b6c6`, one import hunk resolved against the TP merge, and its CPU test `7ec30e5a9`). The region declarations (`615908fcd`, MLA/KDA `torch_remat` regions for RegionAC) are out: the report's paragraph (sec 5.2.2) says nothing of the kind, and reading "AC reuse" as reusing attention activations under AC is what put 4656's current body off course. Verdict table against the paragraph: `phase13_k3like_48b_posttrain/REBASE_CPMM_DEP_AC_2026-09-15.md` section 1 (the block stack's per-block re-cat, sentence 1 of the paragraph, is not in this PR either; a separate decision).

Numbers: the table below was measured on the pre-rebase head (main `b21f7d43e`) on one RTX 5060 Ti; the recompute commit is unchanged by the rebase, but per the H100 rule the body's table is rerun on H100 on `7ec30e5a9` before filing, and the 5060 only smokes (`int0915_logs/run_ac4656r.sh`: dp1 with AC none / selective / full, seeded, 3 steps). Smoke on `7ec30e5a9` (2026-09-16): rc 0 in all three modes, loss `12.62200` / `10.81623` / `8.22700` identical across none, selective and full (the step-1 value moved from `12.63048` with the base, main now carries quantile balancing), peak memory 14.17 / 12.68 / 12.53 GiB. `ac_review3` and `k3_ac_reuse_attention` both at `7ec30e5a9` (force-with-lease from `ea1316606`, the user's standing word for the draft). The RegionAC row of the old table is gone with the regions. Test plan names the new test file. Not run on this head: the 33-layer and pipeline cells.

--- PASTE BEGIN ---

## Summary

Recompute Kimi K3's attention-residual math in backward, as described in section 5.2.2 of the Kimi K3 technical report ("The AttnRes computation is entirely wrapped with checkpointing, so the activation saved for the backward pass at each layer is identical to that of the standard residual architecture").
- Run `_apply_attention_residual` under a `torch_remat` checkpoint in `kimi_k3/model.py` whenever autograd records it and no activation-checkpointing policy already covers the block, for the two residuals of every block, and always for the output aggregation.
- Mark the blocks in `parallelize_kimi_k3` when selective, full or region AC checkpoints them, so their residuals run as plain calls inside that checkpoint.
- Add a CPU test of the recompute: gradients bitwise against the unwrapped residual, the residual math run once more in backward, no stack-shaped saved tensor.

## Design

The residual upcasts the whole block stack and the prefix sum to fp32 and keeps the (N+1)-entry intermediates for backward, so each layer's saved activations grow with the stack. Under a checkpoint, backward recomputes them from the stack and the prefix sum, which are kept anyway, so the per-layer saved set matches a standard residual block at the cost of re-running the residual math. Values are unchanged.

The saved set matches a standard residual block in every mode. With activation checkpointing off, the model guarantees it: each residual is a `torch_remat` checkpoint. Selective and full AC wrap the whole block in a torch checkpoint, which already keeps these intermediates out of the saved set, so `parallelize_kimi_k3` marks the block (`checkpoint_residual = False`) and its residuals run as plain calls; a second checkpoint inside would only recompute them again. Under RegionAC the block is a `torch_remat` checkpoint too, so the same flag applies. The output aggregation on the head stage sits outside every block checkpoint and always takes its own.


## Results

Main `b21f7d43e` against this PR on the debug model, dp1, bf16, `seed=42`, deterministic, 2048 tokens per step in 512-token micro-batches, 10 steps, one GPU, one inductor cache shared by every cell and warmed by a 1-step run of each. Loss and grad norm are compared at all ten steps.

```bash
PYTHONPATH=<attention-gym main>:. torchrun --nproc_per_node=1 -m torchtitan.train \
  --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
  --training.steps 10 --training.num-tokens-per-train-step 2048 \
  --training.num-tokens-per-microbatch-per-dp-rank 512 activation-checkpoint:none
```

| activation checkpointing | loss and grad norm, main vs PR | peak memory main / PR | tps main / PR |
| --- | --- | ---: | ---: |
| none | bitwise (`12.63048` / `20.6250` at step 1, `3.64293` / `4.9062` at step 10) | 14.61 / 14.17 GiB | 838 / 788 |
| selective (the flavor default) | bitwise | 12.68 / 12.72 GiB | 501 / 501 |
| full | bitwise | 12.52 / 12.52 GiB | 608 / 616 |

With activation checkpointing off the recompute saves 0.44 GiB of peak memory for about 6% of throughput. Selective and full AC run as on main, because their block checkpoint already excludes the intermediates and the residual is not checkpointed a second time.

## Test plan
- `pytest tests/unit_tests/cpu/test_kimi_k3_attention_residual_recompute.py -q` (`3 passed`)
- Scoped pre-commit checks, including formatting and Pyrefly (`passed`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WBy1d9YVu44nYCVqykRqL1

--- PASTE END ---
