# Draft PR：Kimi K3 流水线 cache 的说明页和两张图（base main）

- 分支：fork `QIU023/torchtitan` 的 `k3_pp_cache_doc` = `52f9dd6d3`，在 upstream main `f35966713` 之上只有一个 commit（2026-09-27：4312 已 squash 合并为 `e033f7517`，分支从旧的 `c79066a0d` 改为直接 cherry-pick 到 main，页面引用的名字在 main 上都存在）。
- **09-29 第一轮 review（shuhuayu，5 条行内评论）已改完并推送（用户："直接推 k3_pp_cache_doc"）。** 回复见 `REPLY_4914_REVIEW1_2026-09-29.md`。分支 `k3_pp_cache_doc` 从 `52f9dd6d3` fast-forward 到 `bb4b02064`，多了两个提交：
  - `fe44d3068`：前向路径全部改成黑色；split 框补上 ∇Δ = ∇B[Δ] + deposits 和 deposits 的箭头；图例的 collect、1F1B 那句话都改了；删掉 "What the stage does" 和 "The routing tables" 两节，以及图里 tables 的注释；loop placement 的约束保留成一句话。页面现在 47 行。
  - `bb4b02064`：`_pack_outgoing_delta` 的 docstring 从 views 改成 copied。下面粘贴区相应加了 `stage.py` 那一条，GitHub 上的 body 要用它替换。
- 最初的 `52f9dd6d3` 只加三个文件，没有改任何代码：
  - `torchtitan/models/kimi_k3/pipeline_parallel/PP_ATTN_RES_CACHE.md`（57 行）：09-25 从 4312 删掉的那版说明页，加上 Overview（主图）和 Example（例子图）两节，store 统一叫 rank cache。
  - `assets/images/kimi_k3_pp_attn_res_cache.svg`：一个 virtual stage、cache 开和关、forward 和 backward、带图例。
  - `assets/images/kimi_k3_pp_attn_res_cache_example.svg`：forward only，pp4 × vp2，loop 排布，每个 stage 开一个 block，图下是每一跳的跨 rank P2P 通信量。
- 图放 `assets/images/`，和 `models/common/MOE_SHARDING.md` 放 `moe_sharding.png` 的约定一致；页面用相对路径 `../../../../assets/images/...` 引用，两个链接和四个 `.py` 链接在分支上都能解析到文件。
- 检查：仓库自己的 pre-commit（跳过会全仓改文件的 pyrefly）在三个文件上 trailing-whitespace、merge-conflict、large-files、end-of-file、codespell 全过；lychee 链接检查在这台 Windows 机器上起不来（`/bin/bash` not found），是环境问题，相对链接已手动核对。例子图的每一格都是 `layout.py` 的 `infer_block_layout_tables` 对这个配置（cache 开和关）的实际输出。
- 这台机器没有 `gh`，PR 需要你在网页上开：打开 https://github.com/pytorch/torchtitan/compare/main...QIU023:torchtitan:k3_pp_cache_doc?expand=1 ，标题和正文从下面复制，在绿色按钮的下拉里选 **Create draft pull request**。开好以后把 PR 号告诉我，我把 reviewer 回复里的链接补上。

# PR title: [Kimi K3] Document the pipeline's attention residual cache, with two figures

--- PASTE BEGIN ---

## Summary

Adds a page for the Kimi K3 pipeline's attention residual cache, `torchtitan/models/kimi_k3/pipeline_parallel/PP_ATTN_RES_CACHE.md`, with two figures in the style of `models/common/MOE_SHARDING.md`, as asked in the review of #4312.

- `assets/images/kimi_k3_pp_attn_res_cache.svg`: one virtual stage on one rank with `attn_res_cache` on and off, forward and backward, with the P2P sends and receives and a legend for every symbol.
- `assets/images/kimi_k3_pp_attn_res_cache_example.svg`: a forward-only example, four ranks with two virtual stages each in loop placement, showing the blocks each rank holds when each stage starts, the blocks each hop brings, and the per-hop P2P volume with the cache on and off.
- `torchtitan/models/kimi_k3/pipeline_parallel/stage.py`: the docstring of `_pack_outgoing_delta` says the payload is copied out of the model's stack; it said views.

--- PASTE END ---
