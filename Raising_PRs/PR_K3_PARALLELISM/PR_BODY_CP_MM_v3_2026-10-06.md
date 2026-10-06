# PR 4380 body v3（`cpmm_review1` = `f53f65a18`：`7202a40e7` + `fcaaeb25f` + `f53f65a18`），2026-10-06

## 状态（不粘贴）

- **为什么重写：** #4639 已以 `3f087cf15` 合入 main。v2 描述的是叠在 #4639 旧版上的 `923f8bd47`；新版是在 main 上重写的一个提交，设计改了两处（见 `CPMM_4380_PREP_2026-10-06.md`）：
  - tower 每个 micro-batch 只调用一次（旧版按图逐张调用，不开 PP 时各 rank 的 FSDP all-gather 次数会对不上，2 卡复现会卡死）；
  - 拆分的大图特征在整个 CP 组做一次 all-gather（旧版在别的子组上把这些大图整张重编一遍）。
- **PR 分支：** `k3_cp_mm` 还是 `923f8bd47`，同步要等用户的话。review 分支 `cpmm_review1` 已推（旧的备份在 `backup/cpmm_review1_pre_20261006`），10-06 白天在上面加了两个单独的提交：`fcaaeb25f`（gather 在 SPMD 类型检查外运行，CI 的 h100 格子开 typecheck，第 5 步就会走到切分路径）和 `f53f65a18`（删掉私有 helper 的 docstring）。
- **5060 上的结果**（不进 body）：在 `CPMM_4380_PREP_2026-10-06.md` 里；body 的 Results 等 H100。
- 粘贴区已检查：没有 we/our/us，没有破折号。

--- PR 4380 body v3: PASTE BEGIN ---

## Summary

Adds dynamic context parallelism for the Kimi K3 vision tower (report sec 5.2.3): under context parallelism a large image is split by rows across the ranks of a sub-CP group whose attention gathers keys and values, and several large images are spread over equal sub-groups longest-first.

- `vit_cp_plan.py`: the pure planning (which images split, the sub-group layout and balance, each rank's rows, the key layout after the gather).
- `KimiK3VisionCPAttention` (`kimi_k3/vision_encoder.py`): the shared vision attention, gathering the split images' keys and values over their sub-group when the tower hands it a `VisionCPLayout`.
- `KimiK3VisionEncoder.forward`: plans the micro-batch, encodes it, and returns the vision bank of main's CP path.
- `KimiK3VisionEncoder.Config.dynamic_cp_min_patches` (256): images below it stay whole.

## Design

Main's CP path encodes every image of the micro-batch on every CP rank and gathers each rank's rows from that vision bank. This PR changes only how the bank is computed. Images below `dynamic_cp_min_patches`, or whose height does not take whole merge blocks, are still encoded whole on every rank. Each large image is split by rows into bands of whole merge blocks, every frame of a video at the same rows, so the projector's spatial merge and temporal pooling stay local to a band.

The tower runs once per micro-batch on every rank, as on main, so FSDP issues the same collectives on every rank whatever images a data-parallel rank holds. The packed stream carries the whole images first and then this rank's bands of its sub-group's images. The attention gathers only the band part, and one block mask keys every query to its own image and drops the padding rows. After the projector, one all-gather over the CP group returns every split image's merged tokens, so the bank, and everything after it, is main's. With no large image in the micro-batch the tower runs main's path exactly.

A process group cannot be built per batch, so `KimiK3Model.parallelize` builds one layout per divisor of the CP size, on every rank, including pipeline stages without the tower.

## Results

Pending (H100).

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_vit_cp_plan.py -q` (8 passed).
- `pytest tests/unit_tests/gpu/test_kimi_k3_vision_cp.py -q` on 4 GPUs (2 passed): the split tower against the whole tower in fp32, forward and parameter gradients, on one image over the CP group, two images over two sub-groups, one image per rank, and a video whose last rank holds only padding; and two data-parallel groups that split different numbers of images under FSDP.

## Relation to earlier revisions of this PR

The earlier revision sat on the first version of #4639, called the tower once per whole image, and re-encoded other sub-groups' large images whole; this revision runs the tower once per micro-batch and gathers the split images' features over the CP group.

--- PASTE END ---
