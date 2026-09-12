# Short reply to Tianyu's numerics question on PR 4312 (2026-09-12)

For the user to post. CLAUDE.md numerics-acceptance rule: the residual is located (layer, op, first tensor, magnitude; probe `matrix_scripts/pp_step1_0912/`, tree `pp_review4` + probe patch) but its mechanism is not explained, which the rule says blocks posting -- the user's call. Numbers: 4 x H100 PCIe, `pp_review4` = `dbc425403`, logs in `phase13_k3like_48b_posttrain/pp_h100x4_logs_2026-09-12/`; step-1 campaign on 8 x RTX 5060 Ti, `PP_STEP1_BITWISE_PREDICTIONS_2026-09-12.md`. The two no-pipeline controls use local probe switches (`MB_REVERSE`, `NOSYNC_GA`), not the PR. Long form: `REPLY_4312_NUMERICS_2026-09-11.md`.

--- PASTE BEGIN ---

Against our bar -- step-1 loss bitwise, step-1 gradients bitwise or every difference located -- the pipeline passes on the loss and on every gradient the transport touches; one located residual remains, starting at one bf16 ulp in 0.4% of the last layer's `attention.wq_a` gradient, inside that layer's attention backward on the last stage, before any gradient crosses a stage boundary. The later percents are this debug flavour amplifying ulp-level differences, and a single GPU does the same with no pipeline in it.

**Accumulation order, cache off / on vs no PP.** Take one block of the attention-residual stack under pp2 x vp2 (rank 0 holds stages 0 and 2, rank 1 stages 1 and 3). Autograd adds each layer's read of the block onto the gradient that has already arrived, top-down. Without PP that is one running sum. With the cache off each hop hands that running sum to the previous stage, which keeps adding, so the order is the same as one GPU. With the cache on, stages 2 and 3 read the block from their rank's store, sum their own reads from zero and deposit the subtotal, and stages 1 and 0 add it when they collect: the same terms, grouped differently.

```python
import torch
torch.manual_seed(0)
a = [[torch.randn(4096).mul(10.0 ** torch.randint(-2, 2, (4096,))).bfloat16() for _ in range(3)] for _ in range(4)]
def fold(g, reads):                                   # autograd's order: add each read onto what arrived
    for r in reads: g = r.clone() if g is None else g + r
    return g
no_pp = fold(None, a[3] + a[2] + a[1] + a[0])
g = fold(None, a[3])
for k in (2, 1, 0): g = fold(g, a[k])                 # cache off: the running sum crosses each hop
cache_off = g
cache_on = fold(fold(None, a[1]) + fold(None, a[3]) + fold(None, a[2]), a[0])   # stages 2, 3 deposit
print(torch.equal(cache_off, no_pp), int((cache_on != no_pp).sum()))           # True 1568 (of 4096); float64: equal
```

**Why it is not a bug.** Step-1 gradients of every parameter on the 24-layer debug model at pp2 x vp2, with the predictions written down before the dumps were read. Cache on vs off: layers 12-23 and the head are bitwise, and the parameters that produce a block read from a store (layers 0-11, embeddings, vision tower) move by a median of 2-3 bf16 ulps. Deleting one deposit, a real bug in that path, moves the same tensors about a hundred times further (relative L2 0.7-0.9). Against one GPU the pipeline differs in two places, neither the transport: FSDP accumulates the four micro-batch gradients in float32 under the pipeline and in bf16 after each micro-batch without it (matching that makes `lm_head` and the final norms bitwise), and a residual that starts in the last layer's attention backward on the last stage, before any gradient has crossed a stage boundary, identical with the cache on and off.

**The table.** 4 x H100 PCIe, one seed checkpoint, 1024 tokens per step (four stages need four micro-batches, and the multimodal loader needs 256 tokens per micro-batch); steps stop at 20 because the reference memorises the 32-sample debug set after that.

| cell | loss, step 1 | step 10 | step 20 | grad norm, step 1 | step 10 | step 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dp1 | `12.605700` | `3.114620` | `3.373330` | `18.625` | `5.4375` | `3.9844` |
| dp1, micro-batches accumulated as the pipeline does (no PP) | `12.605700` | +3.85% | -1.72% | `18.625` | -10.9% | -6.67% |
| dp1, accumulation order reversed (no PP) | `12.605700` | +4.27% | -2.31% | `18.625` | +22.4% | +6.67% |
| pp2 | `12.605700` | +3.61% | -2.52% | +0.67% | +4.60% | -6.27% |
| pp2 x vp2, cache on | `12.605700` | +1.17% | -0.71% | `18.625` | -31.6% | +1.96% |
| pp2 x vp2, cache off | `12.605700` | +12.85% | -2.72% | `18.625` | +10.9% | -7.84% |

The size of that sensitivity depends on the device, and part of it is KDA: Attention Gym's KDA is written for SM100/SM103 (main refuses anything else), so on H100 and RTX 5060 Ti we run it with that guard lifted, and its fused kernels pick their configuration by autotuning in each process; main has not validated these cards. The same pp2 cell reads +3.6% at step 10 here and +13.7% on RTX 5060 Ti. A pass with KDA's autotuning off is running.
