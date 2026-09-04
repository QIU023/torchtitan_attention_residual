# Audit of the PP and CP diffs pushed after the first review (2026-09-04)

中文摘要：两份 diff 逐行读完（PP：`pp_review3` `d6b1ffe47` 对 main，12 个文件 +1231/-33；CP：`cp_review4` `02392b5c9` 的 CP 层，9 个文件 +780/-16）。没有发现正确性问题：PP 的模型侧重构逐行等价（之前的梯度哈希也是逐位相同），子类的前向/反向/元数据推断与 torch 基类的契约一致；CP 的打包 kernel 拆/包/展开互逆有测试。要改的都是小事：一个死函数、一条误导的注释、两处纯格式抖动、一个缺类型的参数、一处错误信息指错原因、若干超过两行的注释。下面按分支列，每条标了建议动作；改不改由你定，我不动分支。

Both diffs: `/tmp/claude-0/.../scratchpad/audit_pp_full.diff` and `audit_cp_full.diff` (regenerate with `git diff upstream/main pp_review3` and `git diff <spmd tip> cp_review4`).

## PP: `pp_review3` (`d6b1ffe47`) against main

What was checked: the block forward refactor against the original arithmetic (first layer of a block joins the stack before attending; every other layer carries the partial sum) -- equivalent line by line, and bitwise in the earlier probe; the stage subclass against `_PipelineStageBase.forward_one_chunk` of the installed torch (it appends `output_chunks` and fills `fwd_cache` the same way; the base has no input/output validation call in this version, so nothing is skipped); the backward split (`bwd_cache` entries correspond to the stage inputs `(hidden, delta)`, dense, `None` when the delta needs no gradient); the routing tables (`delta = accumulated - held[receiver]`, readers and deposits per rank) against the uneven-split test; metadata inference (`_compute_outputs` assembles the same stack from zero placeholders; `_compute_input_grads` returns dense gradients).

| # | file | finding | action |
|---|---|---|---|
| P1 | `layout.py` | `unstack_blocks` has no caller left (the adapter used it) | delete |
| P2 | `pipeline_stage.py` `_retrieve_recv_grads` | when the payload's gradient arrives as `None` but this stage committed blocks in it, the deposits stay uncollected and the rank's first stage later raises "deposits left uncollected", which names the wrong cause | raise there, naming the block and the stage |
| P3 | `pipeline.py` `kimi_k3_module_fqns_per_model_part` | recomputes core's virtual-stage count (`ceil((L + first + last) / layers_per_stage)`); if core's `_get_pipeline_metadata` changes, the split diverges silently | reuse core's helper if it exposes the count, else a one-line comment naming the source |
| P4 | `model.py` block forward, `pipeline_stage.py`, `layout.py` | several comments run 3-5 lines (the block-forward comment, the routing-table simulation, the deposit rules); the maintainers asked for 1-2 line comments with the reasoning in the PR body | trim to the one-line "what", the "why" is in the body and the logbook |
| P5 | `pipeline_parallel.py` | `stage_class` threaded through `pipeline_llm` and `_pipeline_module_split` (+10/-2); the dropped `# pyrefly: ignore` was unused (0 errors) | none |
| P6 | tests | three CPU files cover the split, the uneven tables with cache on and off, and the carrier ops; no test runs a stage end to end on CPU (the gloo path would need a two-rank harness) | none for this round; the pp2 GPU cell is the end-to-end test |

No correctness finding. The one-line review fixes asked for on 4312 are all in (`first_layer_in_block`, the public split function, the CI cell dropped, the comment revert).

## CP: `cp_review4` (`02392b5c9`), the CP layer over the stack + TP + spmd

What was checked: the packed kernels against 4450's `UlyssesCPFlexAttention` and 4449's `AllGatherCPFlexAttention` (inheritance, `_reshard`, `reduce_dtype`, the `shard_attention_mask` / `shard_attention_heads` flags); `_split_rope` / `_expand_rope` as inverses (tested with pass-through collectives); the KCP kernel's plan and conv history; the recipe transform and the model-side KDA check; the vision splice under CP and the CP-with-SP rejection; the boundary (`set_gqa_inner_attention_local_map`, identity on cp).

| # | file | finding | action |
|---|---|---|---|
| C1 | `model.py` MLA forward | the new comment says "the inner attention is the Ulysses kernel"; it may be the all-gather kernel | say "a CP kernel that owns its exchange" |
| C2 | `model.py` MLA forward | `del positions` removed; `positions` is still unused, so the line is churn against main | restore `del positions` |
| C3 | `sharding.py` | two hunks are pure reflow of `_set_mla_sharding`'s signature and call (ufmt was happy either way) | revert the reflow |
| C4 | `kda.py` | `cp_plan=None, cp_group=None` on `KDAKernel.forward` and `_conv_and_scan` are untyped | `ContextParallelPlan \| None`, `dist.ProcessGroup \| None` |
| C5 | `model.py` `_exchange_sentinel_counts` | `torch.cuda.current_device()` as the device of the counts | `vision_embeds.device` |
| C6 | `context_parallel.py` MLA Ulysses | `q`, `k_nope`, `v` reach FlexAttention as views of the packed tensor after the split; correct, and FlexAttention may copy them to contiguous. The generic kernel hands over contiguous tensors | note in the body; measure before adding `.contiguous()` (three copies) |
| C7 | `context_parallel.py` KCP | packed-document boundaries raise `NotImplementedError`; the flex path passes none, the varlen path would hit it at the first step | stays; stated in the body |
| C8 | `model.py` splice under CP | `+ unused * 0.0` keeps the tower's collectives; tianyu-l's llm/vlm variant question decides whether this disappears | the user's call, in the reply |
| C9 | `torchtitan_recipes/kimi_k3.py` | mutates the passed config before `apply_transforms` copies it, the same pattern as `torchtitan_recipes/muse_glimmer.py` | none |

No correctness finding. The four MLA kernels agree at step 1 to the digit and the packed-vs-generic gradients differ at the bf16-rounding level (`PR_BODY_CP.md`).

## Numbers to re-measure after the sequence-parallel splice fix

The tp2 rows taken before the fix (the CP body's tp2 row and the dp1-vs-tp2 gradient control) are marked in the bodies; the corrected tp2 with SP read 12.54164 / 7.35554 / 3.16327 before the runs were stopped. tp4 and dp2 x tp2 on the corrected tip, and the tp2 gradient control, are the outstanding cells.
