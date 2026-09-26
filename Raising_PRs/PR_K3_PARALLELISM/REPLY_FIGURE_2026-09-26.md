# 回复 reviewer 对 cache 图的意见（2026-09-26）

reviewer 原话：图是对的；`B ≤ s − P` 这个简写不直观（意思是 stage s − P 及更早产生的所有 block）；建议加一个 loop 排布、只看 forward、能看出哪个 rank 持有哪些 block 的小例子。

改动（都在 logbook，包说明页和两张图作为单独的 change 提交，PR 4312 已被两位 approve，不动它的分支）：
- `pp_attnres_cache.svg`：cache 框里的 `B ≤ s−P` 改成两行文字 "blocks produced at / stage s−P or earlier"；图例 `s, P` 一条改成 "this rank last ran the micro-batch at stage s−P"，说明为什么是 s−P。
- `pp_attnres_cache_example.svg`（新）：forward only，3 个 rank、每个 2 个 virtual stage、loop 排布，每个 stage 开一个 block；每个 stage 框写入口时 cache 里的 block 和它 commit 的 block，每一跳标 Δ，图下对比 cache 开 / 关时每一跳带的 block。
- 例子里每一格都是分支上 `layout.py` 的 `infer_block_layout_tables` 对这个配置的实际输出（cache 开和关都跑了），不是手推的。
- `PP_ATTN_RES_CACHE_DRAFT.md`：包说明页加 Overview（主图）和 Example（例子图）两节，其余沿用 09-25 删除前的文本，store 统一叫 rank cache。图放 `assets/images/`（MOE_SHARDING.md 的约定）。

--- PASTE BEGIN ---

Thanks. The cache box now reads "blocks produced at stage s−P or earlier" instead of the shorthand, and the legend says why: s−P is where this rank last ran the micro-batch.

I added the example as a second figure: forward only, three ranks with two virtual stages each in loop placement, each stage opening one block. Each stage shows the blocks its rank already holds and the block it commits, each hop the blocks it carries, and a strip underneath compares what the same hops carry with the cache off. The values are what `layout.py` computes for that split.

--- PASTE END ---
