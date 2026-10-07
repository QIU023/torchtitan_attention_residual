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


--- PR 4380 body v3: PASTE BEGIN ---

## Summary

Adds dynamic context parallelism for the Kimi K3 vision tower (report sec 5.2.3): under context parallelism a large image is split by rows across the ranks of a sub-CP group whose attention gathers keys and values, and several large images are spread over equal sub-groups longest-first.

- `vision_cp/plan.py`: the pure planning (which images split, the sub-group layout and balance, each rank's rows, the key layout after the gather).
- `VisionCPAttention` (`vision_cp/attention.py`): the shared `VisionAttention`, gathering the split images' keys and values over their sub-group when the tower hands it a `VisionCPLayout`.
- `MoonViTCPEncoder` (`vision_cp/encoder.py`): a `MoonViTEncoder` that plans the micro-batch, encodes it, and returns the vision bank of main's CP path; Kimi K3's encoder subclasses it.
- `MoonViTCPEncoder.Config.dynamic_cp_min_patches` (256): images below it stay whole.
- `install_vision_cp` (`vision_cp/__init__.py`): builds the sub-CP groups, called from `KimiK3Model.parallelize`.

## Design

- Scope: main's CP path encodes every image of the micro-batch on every CP rank and gathers each rank's rows from that vision bank; this PR changes only how the bank is computed.
- Which images split:
  - An image with at least `dynamic_cp_min_patches` patches whose height takes whole merge blocks.
  - Every other image is still encoded whole on every rank.
- How a large image splits:
  - By rows into bands of whole merge blocks, every frame of a video at the same rows, so the projector's spatial merge and temporal pooling stay local to a band.
  - Trailing ranks of a sub-group hold fewer real rows and pad their band to the first rank's length.
  - Several large images go to equal sub-CP groups: as many groups as the largest divisor of the CP size not above the number of large images, filled longest first.
- One tower call per micro-batch:
  - Every rank calls the tower once, as on main, so FSDP issues the same collectives on every rank whatever images a data-parallel rank holds.
  - The packed stream carries the whole images first, then this rank's bands of its sub-group's images.
- Attention:
  - Only the band part gathers keys and values, over the sub-group.
  - One block mask keys every query to its own image; padding keys match nothing, and padded queries are dropped after the projector.
- Vision bank:
  - After the projector, one all-gather over the CP group returns every split image's merged tokens, so the bank, and everything after it, is main's.
  - With no large image in the micro-batch, the tower runs main's path exactly.
- Sub-CP groups:
  - A process group cannot be built per batch, so `install_vision_cp` unflattens the CP mesh once per divisor of the CP size, at parallelize time.
  - Building a group is collective, so every rank builds them, including pipeline stages without the tower.
- Placement:
  - The mechanism lives in `kimi_k3/vision_cp/`, and its classes extend only `VisionAttention` and `MoonViTEncoder`, so another MoonViT model can use it once it supports context parallelism.
  - Kimi K3 changes in three places: its encoder's base class, the flavors' vision attention, and one `install_vision_cp` call in `parallelize`.

## Results

Kimi K3 debug model on 4x H100 (torch 2.15.0.dev20260906+cu126), main `948d65c86`, CP=2, 100 steps, seed 42, deterministic, SPMD typechecking on as in #4639's cells, AdamW (lr 8e-4) because the recipe's DistMuon does not build under CP on main; each block of cells shares one warm cache; loss / grad norm.

| Configuration | Step 1 | Step 10 | Step 50 | Step 100 |
|---|---|---|---|---|
| main, all-gather or Ulysses | 8.19922 / 2.5781 | 3.68411 / 2.5312 | 2.47760 / 2.4531 | 3.04643 / 5.7812 |
| main, all-gather, second run | 8.19922 / 2.5781 | 3.68411 / 2.5312 | 2.47760 / 2.4531 | 3.04643 / 5.7812 |
| this PR, all-gather or Ulysses | 8.19922 / 2.5781 | 3.68293 / 2.5625 | 2.47868 / 2.4531 | 3.02490 / 5.7500 |
| this PR, every image split (threshold 128) | 8.19896 / 2.5781 | 3.67482 / 2.5469 | 2.49460 / 2.4844 | 3.01145 / 5.5938 |

- All-gather and Ulysses give identical numbers for all 100 steps, on main and on this PR; main's second run is identical for all 100 steps.
- This PR matches main bitwise through step 4. Step 5's micro-batch holds the first 256-patch image, the first one it splits.
- Splitting every image moves step 1 by 2.6e-4, the split tower's bf16 rounding: a probe that hands main's vision bank to the language model in that run reproduces main bitwise for all 100 steps, and in fp32 the GPU test matches the split tower to the whole tower.
- With CP=1 this PR matches main bitwise for all 100 steps.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_vision_cp_plan.py -q` (7 passed).
- `pytest tests/unit_tests/gpu/test_kimi_k3_vision_cp.py -q` on 4 GPUs (2 passed): the split tower against the whole tower in fp32, forward and parameter gradients, on one image over the CP group, two images over two sub-groups, one image per rank, and a video whose last rank holds only padding; and two data-parallel groups that split different numbers of images under FSDP.

## Relation to earlier revisions of this PR

The earlier revision sat on the first version of #4639, called the tower once per whole image, and re-encoded other sub-groups' large images whole; this revision runs the tower once per micro-batch and gathers the split images' features over the CP group.

--- PASTE END ---
