# PR-PP 正文(PR-text 规则:无小标题、无粗体结构、无证据表)

--- PASTE BEGIN ---

Adds pipeline parallel to Kimi K3, and fixes three things that made it wrong rather than broken.

The block attention residual did not cross a stage boundary. Every stage started its forward with a fresh zero-width residual, so the blocks completed on stage 0 were dropped and stage 1 trained against a different model, with no shape error anywhere. On the debug flavor with the same tokens per step, no PP gives 12.46284 / 9.62380 / 7.44679 and pp2 gave 12.48449 / 11.93899 / 9.30017. The residual is now the stage's second output and second input, which is what the block sum is defined over.

The final aggregation ran on every stage. Under PP the stages that do not own the head have output_res_proj set to None, the same way norm and lm_head are, so it now runs only where the head is.

And the stage FQN injection returned early on this model and said nothing. It reads the layer count from a flat config's num_hidden_layers, and this model carries the layers themselves, so the split silently fell back to core's and left the AttnRes aggregation modules off the last stage. It also now maps its older spellings back to core's when the model uses those, rather than emitting FQNs that match no child -- core sets every non-matching child to None, so that failure is a stage missing pieces rather than an error. The vision tower is named for the same reason: it belongs with whichever stage kept the embedding, since vision features are spliced into the embeddings and nothing vision-side crosses a boundary.

What settles the model side is a probe that splits the built model the way core does -- keep a slice of layers, set the modules the stage does not own to None -- and chains the halves by hand, with no schedule, loss or microbatching involved. Against the whole-model forward: max absolute difference 0.000e+00. End-to-end pp2 still differs from a no-PP run, which is initialization: each rank inits only its own stage, so the RNG is consumed differently. Expert parallel is the control -- it leaves init order alone and matches the dp2 baseline to five digits.

    torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 \
      --config kimi_k3_debugmodel --parallelism.pipeline_parallel_degree 2 \
      --parallelism.num-pp-microbatches 2

--- PASTE END ---

## 不进正文的支撑

| 声称 | 证据 |
|---|---|
| 块残差被丢弃 | 同批次对照,step 3:7.44679 vs 9.30017 |
| stage 接口正确 | `pp_stage_parity_4025.py`,max_abs = 0.000e+00 |
| FQN 注入静默跳过 | CPU 测试 `test_pp_fqn_injection.py`,3 个用例 |
| 塔的归属 | 同上,断言恰好一段持有且是 embedding 那段 |

## 未包含

DEP(`dep_bubble_*.py` 已在树上,未接线未测)。
