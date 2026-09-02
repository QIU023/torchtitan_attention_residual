# PR title: [Draft] [Kimi K3] multi-token prediction: one KDA-typed mirror layer and the composite loss

Branch `k3_mtp` (`5ce30dbe1`: the two MTP commits plus a merge of upstream/main `1dcb14a0c`, one import resolution -- the flops helpers split upstream). 2 CPU tests pass on the merged tree. Evidence: `phase13_k3like_48b_posttrain/MTP_2026-08-29.md`. Paste between the markers into the PR body.

--- PASTE BEGIN ---

### Summary

Adds multi-token prediction for Kimi K3 (report sec 3.3). Before this change the model trains next-token only; after it, `model_registry("debugmodel", num_mtp_layers=k)` appends k MTP layers -- each mirroring a backbone block -- and `KimiMTPLoss` adds 0.3x the per-depth cross entropy of predicting the token k+1 ahead to the main loss. The released config ships zero MTP layers, so the default model is unchanged; `kimi_k3_debugmodel_mtp` enables one.

### Design

- The mirror layer is `enorm`/`hnorm` -> `eh_proj` (2d -> d) -> one KDA-typed block with `layer_id=0`: it opens its own (empty) attention-residual stack rather than joining the backbone's depth, and KDA consumes the depth-shortened sequence directly (an MLA mirror would need per-depth mask rebuilds).
  - The depth-k input fuses the backbone's final PRE-norm hidden state with the embedding of the token k+1 ahead: the reference feeds `hnorm` the unnormalised state, and normalising twice is not an identity, which would break parity against official MTP weights.
  - The last k+1 positions have no target and are dropped rather than padded -- padding would invent supervision. The mirror block's residual projection starts zero-initialised, so depth k opens as an identity contribution.
- The loss runs on the folded stream (one token axis); the multimodal splice is length-preserving, so shift-by-k stays aligned and visual positions already carry `IGNORE_INDEX` in the labels.
- Two compositions raise rather than silently skipping depths, each with the reason in the message: chunked loss (MTP needs full-vocab logits per depth, exactly the allocation chunked loss exists to avoid -- combining them is a loss change, not a guard) and a pipeline split that separates `tok_embeddings` from `lm_head`.

### Results

```
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_mtp \
  --debug.seed 42 --debug.deterministic --training.steps 10 \
  --parallelism.data_parallel_shard_degree 2
```

Training loss on `kimi_k3_debugmodel_mtp` (composite: main CE + 0.3x depth-1 CE; step-1 value checks against main-only 12.45 + 0.3 x 12.6), one seed, warmed compile cache:

| config | step 1 | step 3 | step 10 |
|---|---|---|---|
| dp1 | 16.36625 | 10.76353 | 4.92550 |
| fsdp2 | 16.33981 | 9.80768 | 4.65185 |
| fsdp4 | 16.34217 | 9.53120 | 4.68098 |

### Changed files

    torchtitan/models/kimi_k3/
      mtp.py                       +163  the mirror layer and the composite loss (new)
      model.py                    +56/-1  mtp_layers config, pre-norm capture, the
                                         depth loop, the two guards
      __init__.py                 +44/-2  num_mtp_layers through the registry
      config_registry.py            +19  the mtp flavor (plain CE)
      tests/test_mtp_layer.py       +32  (new)

### CI/CD Coverage

CPU tests cover the mirror layer's shapes and the loss composition; the flavor trains in the debug integration set.

--- PASTE END ---
