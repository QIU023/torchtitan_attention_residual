# Reply to Tianyu on PR 4312: activation save / release policy (comment 3976976576, `model.py` line 242)

For the user to post as the reply to 3976976576 (Tianyu's follow-up, 2026-09-10; nothing posted in that thread yet). Checked on 2026-09-13 against the PR head `k3_pp_text` = `pp_review4` = `dbc425403`: `KimiK3TransformerBlock.forward` and `_apply_attention_residual` in `kimi_k3/model.py`, `SelectiveAC` / `_get_default_save_ops` in `distributed/activation_checkpoint.py` (the PR base already has torch_remat, #4497; the per-op SAC is unchanged), `parallelize_kimi_k3` (`ac_policy.apply(model)`: one region per layer), `PPRankLocalCache` use in `kimi_k3/pipeline_stage.py`. Derived from the code, not measured; a memory trace of the 2 x 4 example can follow if asked. The attention-residual checkpoint wrapper on the integration branch is NOT in this PR and is not described.

--- PASTE BEGIN ---

Close, with one correction: the redundancy is per block, not per layer. Each block starts a new stack holding all previous blocks' results (so consecutive stacks overlap -- that is the N(N+1)/2), but the layers of a block share that one tensor, and the per-layer residual reads are recomputed in backward: the default `SelectiveAC` makes each layer one checkpoint region, and the reads' ops (cat, fp32 cast, norm, softmax, bmm) are not in its save set. This is our implementation; Kimi hasn't published theirs.

2 blocks x 4 layers, one micro-batch:

| forward | kept until backward | freed right after use |
| --- | --- | --- |
| layer 0 | `S1 = [e]`, `[T, 1, D]` | FFN read `[S1, h]` |
| layers 1-3 | nothing (they share `S1`) | 2 reads each, `[T, 2, D]` |
| layer 4 | `S2 = [e, x4]`, `[T, 2, D]` | reads `[T, 2, D]` and `[T, 3, D]` |
| layers 5-7 | nothing (they share `S2`) | 2 reads each, `[T, 3, D]` |

Extra memory: `S1` + `S2` = 3 rows of `[T, D]` (N(N+1)/2 for N blocks). Backward recomputes each layer's reads from `S` and the layer's input; `S2` is freed after layer 4's backward, `S1` after layer 0's. Without activation checkpointing, autograd would keep every read's fp32 copy -- the per-layer redundancy you describe.
