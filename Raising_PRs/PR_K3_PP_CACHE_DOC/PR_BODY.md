# Draft PR：Kimi K3 流水线 cache 的说明页和两张图（stacked on 4312）

- 分支：fork `QIU023/torchtitan` 的 `k3_pp_cache_doc` = `c3e95b0f7`，在 PR 4312 的 head `k3_pp_text` = `7814d1f8b` 之上只多一个 commit（相对 main 共 16 个 commit，其中 15 个是 4312 的）。
- 这个 commit 只加三个文件，没有改任何代码：
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

## Relation to #4312

Stacked on #4312, whose commits show here until it merges; only the last commit is this PR's, and it adds documentation only.

## Test plan

- The example's values are the output of `infer_block_layout_tables(stage_to_rank={s: s % 4 for s in range(8)}, n_layers=32, layers_per_block=4, layer_to_stage={l: l // 4 for l in range(32)}, cache=...)` from `layout.py`, with `cache=True` and `cache=False`.
- `pre-commit run --files` on the three files: trailing whitespace, end of file, large files and codespell pass.

--- PASTE END ---
