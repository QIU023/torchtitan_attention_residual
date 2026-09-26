# 回复 reviewer 对 cache 图的意见（2026-09-26）

reviewer 原话：图是对的；`B ≤ s − P` 这个简写不直观（意思是 stage s − P 及更早产生的所有 block）；建议加一个 loop 排布、只看 forward、能看出哪个 rank 持有哪些 block 的小例子。

改动：包说明页和两张图放在 stacked on 4312 的 draft PR 里（分支 `k3_pp_cache_doc`，见 `Raising_PRs/PR_K3_PP_CACHE_DOC/PR_BODY.md`）；PR 4312 已被两位 approve，不动它的分支。开好 draft PR 后，把下面粘贴段里的 `<draft PR link>` 换成链接。
- `pp_attnres_cache.svg`：cache 框里的 `B ≤ s−P` 改成两行文字 "blocks produced at / stage s−P or earlier"；图例 `s, P` 一条改成 "this rank last ran the micro-batch at stage s−P"，说明为什么是 s−P。
- `pp_attnres_cache_example.svg`（新）：forward only，4 个 rank、每个 2 个 virtual stage（pp4 × vp2，和 AttnRes 论文的 cache 流水线图同形状，也是 PR 结果表里的一行），loop 排布，每个 stage 开一个 block；每个 stage 框写入口时 cache 里的 block、这一跳收到的 block、它 commit 的 block；图下一行标题 "PP cross-rank P2P communication amount per hop"，对比 cache 开（1, 2, 3, 3, 3, 3, 3）/ 关（1 … 7）时每一跳带的 block。
- 没选 pp2 × vp4：P = 2 时每一跳永远只带上一个 stage 开的那一个 block，看不出 Δ 可以有多个 block、也看不出一个 block 要跨 P − 1 个 rank 接力。
- 例子里每一格都是分支上 `layout.py` 的 `infer_block_layout_tables` 对这个配置的实际输出（cache 开和关都跑了），不是手推的。
- `PP_ATTN_RES_CACHE_DRAFT.md`：包说明页加 Overview（主图）和 Example（例子图）两节，其余沿用 09-25 删除前的文本，store 统一叫 rank cache。图放 `assets/images/`（MOE_SHARDING.md 的约定）。

--- PASTE BEGIN ---

Thanks. The cache box now reads "blocks produced at stage s−P or earlier" instead of the shorthand, and the legend says why: s−P is where this rank last ran the micro-batch.

I added the example as a second figure: forward only, four ranks with two virtual stages each in loop placement (the shape of the cache figure in the Attention Residuals paper), each stage opening one block. Each stage shows the blocks its rank already holds, the blocks its hop brings and the block it commits; underneath, the P2P volume of every hop with the cache on (at most P−1 = 3 blocks) and off (1 to 7). The values are what `layout.py` computes for that split.

The page and both figures are in a draft PR stacked on this one, so this diff does not move: <draft PR link>

--- PASTE END ---
