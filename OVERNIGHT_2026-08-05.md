# Overnight run: anchors and task order

## Restore points (recorded before any of tonight's work)

| repo | branch | commit |
|---|---|---|
| logbook | main | ac7496b |
| fork torchtitan | attention_residual_dev | b99ff970c |
| fork verl | kimi_k3_integration | cb9ac9e1 |

PR bookmarks `k3_pr_{a,b,c,d}` are all at `a146d1bf2` and are NOT touched tonight.

To roll back anything from tonight: `git reset --hard <commit above>` in the
repo concerned. Every commit tonight is separate and revertable on its own.

## Task order (from the user)

1. ViT real TP
2. ViT dynamic CP
3. ViT DEP -- vision compute scheduled into PP bubbles
4. 32-layer multimodal PP8xVP4 stress test (completes the PP line)
5. veRL GRPO
6. full MTP support
7. LoRA TP debug

Rule for tonight: if a task blocks, try a few approaches, then move to the next
one and record why. Do not leave the tree half-edited.

## Log

