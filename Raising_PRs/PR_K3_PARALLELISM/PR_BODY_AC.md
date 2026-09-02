# PR title: [Draft] [Kimi K3] the attention residual under activation checkpointing: recompute the residual math, keep attention outside the wrap

Branch `ac_review1` (`ca9e497f8`, two commits on upstream/main `1dcb14a0c`). CPU test passes. Results are placeholders until the measured rows are pasted in. Paste between the markers into the PR body.

--- PASTE BEGIN ---

### Summary

Two changes to how Kimi K3's attention residual meets activation checkpointing, both non-computation. Before, `_apply_attention_residual` saved its fp32 intermediates -- the whole (N+1)-entry block stack, upcast twice per layer -- for backward; after, the residual math runs under `torch.utils.checkpoint` and those intermediates are recomputed from the stack and the prefix sum, so the activations saved per layer match a standard residual architecture. Before, selective AC wrapped a whole block, and the KDA kernel -- a custom op outside the per-op policy's save set -- was recomputed in backward; after, with `ac_reuse_attention` set, the policy wraps only each block's MoE/feed-forward, and attention and the residual math keep their activations.

### Design

- `_attention_residual_math` is the existing function unchanged; `_apply_attention_residual` wraps it with `use_reentrant=False` when a grad is required, and calls it plainly otherwise (inference, no-grad).
  - The inputs the recompute needs -- the block stack and the prefix sum -- are alive in the graph regardless, so the wrap adds no saved tensor of its own.
- `ac_reuse_attention` is a model config field, off by default; `parallelize_kimi_k3` routes to `_apply_ac_outside_attention`, which uses the policy's own `_wrap_block` on `layers.{i}.moe` or `layers.{i}.feed_forward` so the selective policy's mm save/recompute balance stays where the parameter memory is.
  - The whole-block wrap is unchanged when the field is off; the default is the upstream behavior.
- The test asserts the wrap is value- and gradient-identical to the unwrapped math and that the body runs a second time in backward rather than reading intermediates back.

### Results

Both changes are non-computation, so the bar is bitwise-identical loss against main with the same seed.

<placeholder: dp1 rows on `kimi_k3_debugmodel`, steps 1 / 3 / 10 -- main, this branch with the field off, this branch with the field on; peak memory for the three>

```
torchrun --nproc_per_node=1 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.data_parallel_shard_degree 1
```

### Changed files

    torchtitan/models/kimi_k3/
      model.py                       +40/-2  the residual checkpoint wrap; the ac_reuse_attention field
      parallelize.py                 +26/-1  _apply_ac_outside_attention
    tests/unit_tests/cpu/
      test_kimi_k3_attn_res_checkpoint.py  +92  (new)

### CI/CD Coverage

The three tests are CPU and run in the default suite; no GPU cell is added.

--- PASTE END ---

Notes for us, not for the body:

- Origin: the report reading of 2026-08-29/30 -- AttnRes saved its fp32 block-stack intermediates per layer, and selective AC recomputed the KDA kernel whole. Both were found on the integration tree (`k3_on_4025` 2be96d23e, c67f5b2e1) and ported here as two commits; the comment wording moved from "fla's opaque autograd.Function" to "a custom op outside the per-op policy's save set", which is what attn-gym's `chunk_kda` is on main.
- The third fix from that reading, `pp_balance` (parking saved activations on a peer through the Mooncake Transfer Engine), is held: it adds a third-party dependency to the model folder and needs a decision before it goes near a PR.
