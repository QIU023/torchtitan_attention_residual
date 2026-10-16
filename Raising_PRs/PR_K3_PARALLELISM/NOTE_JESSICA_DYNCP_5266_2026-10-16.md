# 给 Jessica 的说明草稿：#4380 的 dynamic CP 和 #5266（2026-10-16）

用户 10-16 的 goal："给 Jessica 的说明草稿： 写进 logbook 的 md 文件，由你决定是否发。"

- **对象：** Jessica Zhong（GitHub `acisseJZhong`），#5266 "Make the vision encoder CP-invariant, only dp_shard" 的作者。
- **发的地方：** 建议发在 #4380 上，作为评论 @ 她。#4380 是唯一改到 #5266 代码的地方，评论里引用的提交链接也只在这个 PR 里能点开。也可以私信她，正文不用改。
- **发之前要做的：**
  - 先把 review 分支 `cpmm_review1` 推上去；
  - 把下文的 `<fix commit>` 换成修复提交在 fork 上的链接。
- **依据：** `DYNCP_5266_FIX_2026-10-16.md`（探针、单测、类型检查格子、替换法、栈上的验证）。
- **写法：** 结论放第一句，每句只讲一件事；不用 we/our/us，不用破折号；最后只提一个请求，就是请她看这个 hook。

--- PASTE BEGIN ---

@acisseJZhong Dynamic CP in #4380 breaks the CP invariance that #5266 gives the vision encoder, so the K3 tower now opts out of it when dynamic CP is on.

With dynamic CP, each CP rank encodes a different row band of a large image. The output boundary's CP all-reduce then reaches each band through the feature bank's all-gather backward, which is a reduce-scatter, so split features get CP times their gradient. FSDP over dp_shard alone never reconciles the CP ranks, so their towers drift apart. On CP2 with every image split, the two ranks' tower gradients differed in 22 of 22 tensors at step 1, their sum was 2.00x the dynamic-CP-off gradient (cosine at least 0.9999), and 16 of 22 parameter tensors differed after the first update.

The fix is in <fix commit>. With dynamic CP on, `MoonViTCPEncoder._parallelize` declares CP as R instead of I in the tower's sharding configs, which is the typing before #5266, and converts `pixel_values` from I to R at the tower boundary. A new `MultimodalModel._encoder_is_cp_invariant(encoder)` hook, True by default, lets K3 shard that tower over dp_shard x cp, so FSDP sums the partial gradients. Every other encoder, and K3 with dynamic CP off, keeps the #5266 path unchanged. A new 2-GPU test goes through the real FSDP wrapper and output boundary: split and CP-invariant fp32 tower gradients agree to 1.2e-6, against 1.0 before the fix.

Could you take a look at the hook in `torchtitan/models/common/multimodal.py`? It is the only place #4380 touches the #5266 code.

--- PASTE END ---
