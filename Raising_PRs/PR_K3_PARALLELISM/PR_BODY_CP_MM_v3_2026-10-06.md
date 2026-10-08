# PR 4380 body v3（10-07 重构后：PR 分支 `k3_cp_mm` = `cpmm_review1` = `6317c5538`，在 main `948d65c86` 上，七个提交），2026-10-06

## 状态（不粘贴）

- **10-07 按 DEP 方式重构（用户："直接改，review和pr分支都做，然后改body，及时推送diff"）：** `k3_cp_mm` = `cpmm_review1` = `6317c5538`，在 `9b03b4af1` 之上快进两个提交：
  - `e094d6b9a` "kimi_k3: dynamic CP as a vision_cp package"：新包 `kimi_k3/vision_cp/`，包括 `plan.py`（原 `vit_cp_plan.py`，内容不变）、`attention.py`（`VisionCPAttention`、`VisionCPLayout`、gather）、`encoder.py`（`MoonViTCPEncoder`，`_forward_split` 拆成 `_pack_inputs`、`_encode`、`_assemble_bank`）、`__init__.py`（`build_cp_subgroups`、`install_vision_cp`）。K3 目录只剩三处改动：`KimiK3VisionEncoder(MoonViTCPEncoder)`、flavor 换用 `VisionCPAttention`、`parallelize` 里一行 `install_vision_cp`。
  - `6317c5538`：GPU 测试开头的两行注释缩成一行。
  - **数值不变的证据（本机）：**
    - 原样搬过去的 9 个函数和方法，AST 和旧 head 一致，只差改名；`plan.py` 逐字节相同。
    - 拆开的 `_forward_split` 做了新旧对比：4 个 gloo 进程（CPU、fp32，两边都换成同一个稠密注意力），跑 GPU 测试的 4 种情形，新旧两棵树的输出和全部参数梯度 368 个张量逐位相同，并确认每种情形都走了切图路径（kit `kit_cpmm_2026-10-06/local/vision_cp_equiv/`；本机用桩代替 CuTeDSL，不进任何提交）。
    - CPU 规划测试 7 passed，black 22.12 和 pyflakes 干净。
  - **GPU 会话待跑：** `test_kimi_k3_vision_cp.py`（4 卡，2 passed）；任选一个 CP=2、阈值 128 的端到端格子，在同一份 cache 上和 `9b03b4af1` 比 20 步轨迹（应逐位相同）；pre-commit 的 pyrefly 只看改动文件。
  - 粘贴区改了：Summary 的文件和类名、Design 的 Sub-CP groups 一条、新增 Placement 一条、Test plan 的测试文件名。

- **为什么重写：** #4639 已以 `3f087cf15` 合入 main。v2 描述的是叠在 #4639 旧版上的 `923f8bd47`；新版是在 main 上重写的一个提交，设计改了两处（见 `CPMM_4380_PREP_2026-10-06.md`）：
  - tower 每个 micro-batch 只调用一次（旧版按图逐张调用，不开 PP 时各 rank 的 FSDP all-gather 次数会对不上，2 卡复现会卡死）；
  - 拆分的大图特征在整个 CP 组做一次 all-gather（旧版在别的子组上把这些大图整张重编一遍）。
- **PR 分支（10-06 晚，用户：审核没问题就推）：** `k3_cp_mm` = `9c648c4e2`，force push 前的 `923f8bd47` 备份在 `backup/k3_cp_mm_pre_20261006`。在 `7202a40e7` 上有四个单独的提交：`fcaaeb25f`（gather 放到 SPMD 类型检查外）、`f53f65a18`（删私有 helper 的 docstring）、`2d93f8016`（子组改为 unflatten CP mesh）、`9c648c4e2`（删规划模块没用到的部分，修报错信息，删复述断言的测试注释）。要粘贴本 body，并把标题里的 "[DO NOT review, pending rebase]" 去掉。
- **5060 上的结果**（不进 body）：在 `CPMM_4380_PREP_2026-10-06.md` 里；body 的 Results 等 H100。
- 粘贴区已检查：没有 we/our/us，没有破折号。
- **Results（10-07 H100）：** box 115.124.123.239，4 卡 H100，驱动 560，所以 torch 是 cu126 的 0906 nightly，加 kit 的 shim（`param_dtype_override_fn`、DistMoE 的两个 pipelining 名字）。测的是 PR head `6317c5538` 和 main `948d65c86`，和 GitHub 上的 PR 一致。kit `kit_cpmm_2026-10-06`（`run_matrix2.sh` 预热 100 步，`table_4380.py` 出表），结果在 box 的 `/workspace/h100_cpmm`。bank 替换：swaplog 里 200 次调用，预热和正式各 100 次。
- **10-07 Design 改成嵌套 bullet**（用户："design又是大段长段落，改成嵌套bullet points的格式"）：内容和原来三段相同，另外按代码补了三条：行带尾部的 rank 补齐、子组数取 CP 大小不超过大图张数的最大因子并按长度优先填、补齐的 query 在 projector 之后丢掉。每条一行。
- **10-07 rebase（用户："现在rebase 4380 把除了h100数值之外的其他的事情都做了"）：** 五个提交原样搬到 main `948d65c86`（中间 21 个上游提交，6 个碰 K3 文件，没有冲突），PR 自身 diff 的 patch-id 不变。旧 head `9c648c4e2` 备份在 `backup/k3_cp_mm_pre_20261007`。
- **CI 覆盖：** main 的两个 h100 K3 CP 格子在 main 上建 optimizer 就失败（DistMuon layout，10-07 nightly 证实）。修复在单独的分支 `k3_cp_muon_layout`，body 草稿 `PR_BODY_K3_CP_MUON_LAYOUT_2026-10-07.md`。那个 PR 开出来以后，可以在本 body 的 Test plan 加一句："The h100 cells `kimi_k3_mm_allgather_kv_cp` and `kimi_k3_mm_ulysses_cp` split their first 256-patch image at step 5; they train once #<修复 PR 号> lands."

- **10-08 H100 新增（用户："把昨天dynamic cp剩下的H100实验还有特殊的高分辨率测试都跑了，得看到CP维度在超清大图下"）：** Results 末尾加了 tower 的显存和时间表，六行，外加五条说明。数据出自 box 115.124.123.240 -p 30797，PR head `6317c5538`，main `948d65c86`；完整三种 AC 的表和端到端的数在 `CPMM_4380_PREP_2026-10-06.md` 的 10-08 一节。粘贴区约 1290 词（按 wc 计，含表格）。
- **10-08 压缩（用户："1确定压缩 只压缩词语 表格数据不压缩"）：** 两张表逐字节不变；文字从 1024 词压到 783 词（含表格从 1301 到 1060），其中已经算上新加的两条阈值说明（packing 下快 11% 到 23%、显存少 38% 到 58%；单张 256 到约 9000 patch 的图慢约 5%，见 logbook 10-08 阈值一节，默认 256 不改）。

--- PR 4380 body v3: PASTE BEGIN ---

## Summary

Adds dynamic context parallelism for the Kimi K3 vision tower (report sec 5.2.3): under CP a large image is split by rows over a sub-CP group that gathers its keys and values, and several large images spread over equal sub-groups.

- `vision_cp/plan.py`: which images split, over which sub-groups, and each rank's rows.
- `vision_cp/attention.py`: `VisionCPAttention`, `VisionAttention` plus the sub-group key and value gather.
- `vision_cp/encoder.py`: `MoonViTCPEncoder`, the base of Kimi K3's encoder; `dynamic_cp_min_patches` (256) keeps smaller images whole.
- `vision_cp/__init__.py`: `install_vision_cp` builds the sub-CP groups from `KimiK3Model.parallelize`.

## Design

- Main's CP path encodes every image on every CP rank; this PR changes only how that vision bank is computed.
- An image splits when it has at least `dynamic_cp_min_patches` patches and its height takes whole merge blocks.
- Bands are whole merge blocks of rows, the same rows in every frame; trailing ranks pad to the first band.
- Large images fill equal sub-groups longest first, as many as the largest divisor of the CP size not above their number.
- The tower runs once per micro-batch, as on main, so FSDP issues the same collectives on every rank.
- Only bands gather keys and values; padding keys match nothing, and padded queries are dropped after the projector.
- One all-gather over the CP group returns the split images' merged tokens, so the bank is main's; with no large image the tower runs main's path.
- `install_vision_cp` unflattens the CP mesh once per divisor of the CP size; every rank builds the groups, since building is collective.
- The classes extend only `VisionAttention` and `MoonViTEncoder`, so other MoonViT models can reuse them; Kimi K3 changes its encoder's base class, its flavors' vision attention and `parallelize`.

## Results

Kimi K3 debug model, 4x H100 (torch 2.15.0.dev20260906+cu126), main `948d65c86`, CP=2, 100 steps, seed 42, deterministic, typechecking on as in #4639's cells, AdamW (lr 8e-4; DistMuon does not build under CP on main), one warm cache per block; loss / grad norm.

| Configuration | Step 1 | Step 10 | Step 50 | Step 100 |
|---|---|---|---|---|
| main, all-gather or Ulysses | 8.19922 / 2.5781 | 3.68411 / 2.5312 | 2.47760 / 2.4531 | 3.04643 / 5.7812 |
| main, all-gather, second run | 8.19922 / 2.5781 | 3.68411 / 2.5312 | 2.47760 / 2.4531 | 3.04643 / 5.7812 |
| this PR, all-gather or Ulysses | 8.19922 / 2.5781 | 3.68293 / 2.5625 | 2.47868 / 2.4531 | 3.02490 / 5.7500 |
| this PR, every image split (threshold 128) | 8.19896 / 2.5781 | 3.67482 / 2.5469 | 2.49460 / 2.4844 | 3.01145 / 5.5938 |

- All-gather and Ulysses are identical for 100 steps, on main and on this PR, as is main's second run.
- This PR matches main bitwise through step 4; step 5's 256-patch image is the first it splits.
- Splitting every image moves step 1 by 2.6e-4, the split tower's bf16 rounding: with main's vision bank swapped in, the run matches main bitwise for 100 steps.
- In fp32 the split tower matches the whole tower within 2e-6 forward and 1e-5 in gradients, in the GPU test and at CP 2 and 4 on 2016 and 4032 px images and 16 video frames.
- With CP=1 this PR matches main bitwise for 100 steps.

Vision tower alone (Kimi K3's: 27 layers, dim 1024, random bf16 weights), forward and backward over one micro-batch's images on 4x H100, with the recipe's selective checkpointing unless noted; peak memory per GPU / time. Every CP rank holds all the images, so main measures the same at CP 1, 2 and 4.

| Images (patches) | main | this PR, CP=2 | this PR, CP=4 |
|---|---|---|---|
| one 2016 px image (20736) | 15.1 GiB / 0.81 s | 10.5 GiB / 0.44 s | 7.5 GiB / 0.27 s |
| one 4032 px image (82944) | 57.6 GiB / 11.1 s | 39.3 GiB / 5.6 s | 27.1 GiB / 2.9 s |
| one 4032 px image, no checkpointing | out of memory | 47.8 GiB / 5.5 s | 31.4 GiB / 2.8 s |
| one 4032 px image, full checkpointing | 9.4 GiB / 14.0 s | 5.8 GiB / 7.0 s | 3.9 GiB / 3.6 s |
| four 2016 px images (82944) | 57.6 GiB / 3.2 s | 29.7 GiB / 1.6 s | 15.8 GiB / 0.81 s |
| 16 frames of 448 px in 4-frame items (16384) | 12.0 GiB / 0.22 s | 6.5 GiB / 0.15 s | 3.7 GiB / 0.15 s |

- Figures include 1.7 GiB of weights and gradients; flex compiles without max-autotune on both sides.
- Several images or video items go whole to one-rank sub-groups, so memory falls as 1/CP.
- A split image keeps its whole keys and values on each rank for the attention backward, so its memory falls as 1/CP only under full checkpointing.
- Training the debug text model with this tower (recipe settings, AdamW) on 2016 px images peaks at 19.1 and 17.6 GiB on main at CP=2 and 4, and 15.1 and 10.8 GiB with this PR.
- With samples packed into 8192-token micro-batches, this PR trains 11 to 23% faster than main with 38 to 58% less memory (cc12m-test at native size and at 1008 px, CP 2 and 4).
- A micro-batch holding a single image of 256 to about 9000 patches trains about 5% slower, since its per-layer gathers cost more CPU time than the split saves.
- On cc12m-test resized to 1008 px, this PR differs from main from step 1 (8.20263 against 8.20187); with main's vision bank swapped in, it matches main bitwise for 20 steps.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_vision_cp_plan.py -q` (7 passed).
- `pytest tests/unit_tests/gpu/test_kimi_k3_vision_cp.py -q` on 4 GPUs (2 passed): split against whole tower in fp32 for one image over the CP group, two images over two sub-groups, one image per rank and a video whose last rank holds only padding; and two data-parallel groups splitting different numbers of images under FSDP.

## Relation to earlier revisions of this PR

The earlier revision sat on the first #4639, called the tower once per image and re-encoded other sub-groups' large images; this one calls it once per micro-batch.

--- PASTE END ---
