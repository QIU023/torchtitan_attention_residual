# MoonEP 重写：今晚 4 × H100 的 smoke 套件（2026-09-29）

分支 `moonep_review1`（fork），在 main `5dc97a3e7` 上重写的 MoonEP 集成。PR #4751 的分支 `k3_moonep_seam` 还是旧实现 `f556ab4fd`（备份 `backup/k3_moonep_seam_pre_rewrite_20260928`），smoke 通过、用户说同步以后再推。body 草稿 `../PR_BODY_MOONEP_v2_2026-09-29.md`，审计 `../DIFF_AUDIT_MOONEP_2026-09-28.md`，本地检查 `LOCAL_CHECKS.md`。

## 机器要求

- 4 张 H100（或更新的卡），挂在 NVSwitch 上，每张卡的 multicast 属性都为 1。4 卡的切片常常没有 NVSwitch，`103.60.105.169` 这类机器跑不了（记忆 `moonep-needs-switch-multicast`）。
- 开机后第一件事：`bash check_box.sh`，要看到 `MOONEP OK`；看到 `NOT USABLE FOR MOONEP` 就直接退掉机器，不要装任何东西。

## 步骤

1. `bash check_box.sh`（几秒）。
2. `bash setup_box.sh`（10 到 20 分钟）：在 `~/mep` 下建 venv，装 torch nightly cu130、仓库的依赖、cutlass DSL 全家 4.6.2（MoonEP 公开版钉的版本），编译 MoonEP `33327eb`，把 fork 的 `moonep_review1` 拉到 `~/mep/tt`。最后一行打印的 `nvidia-cutlass-dsl*` 版本必须全是 4.6.2。
3. `nohup bash smoke.sh > ~/mep/smoke.log 2>&1 &`（约 30 分钟），进度在 `~/mep/results/summary.txt`：
   - CPU 测试（`test_moe.py`、`test_integration_test_definitions.py`）；
   - GPU 测试 `tests/unit_tests/gpu/test_moonep.py`（真实 MoonEP，两卡，2 个用例）；
   - CI 的方式跑 h100 格子 `kimi_k3_fsdp+moonep`；
   - 数值格，`--debug.deterministic`，同一份预热缓存，各 20 步：
     - standard 跑两次，确认确定性；
     - standard 换成 EP=2，数据不变、只换归约顺序，作为噪声底；
     - MoonEP；
     - DeepEP（装了 `deep_ep` 才跑）；
   - 计时格：同样几个后端，不开 deterministic，各 30 步；
   - 出表：`~/mep/results/tables.txt`。
4. 把 `~/mep/results/` 拷回 logbook（本目录的 `h100/` 下），表格填进 body 的 Results。

## 判断标准

- GPU 测试 2 passed，CI 格子 rc=0。
- 数值：第 1 步 loss 和 standard 逐位相同（本机的假 MoonEP 上就是这样）；之后的差应当和噪声底那一行（standard EP=2 对 standard）同一量级，因为两者都只是换了归约顺序。差得明显更大，就先定位，不写进 body。
- cc12m-test 是调试用的小数据集，所以表格只到 20 步，表里要写明。

## 可能遇到的问题

- cutlass 版本不齐：JIT kernel 里报 `mbarrier_arrive_and_expect_tx ... is not a valid CombiningKind`，不是 MoonEP 的 bug，把全家重新装成 4.6.2（记忆 `cutlass-dsl-family-must-match`）。
- MoonEP 编译要 torch 已装好，所以用 `--no-build-isolation`。
- `Buffer` 在进程组销毁前没有显式 `destroy()`，退出时由它自己回收；如果训练结束后进程卡住不退，记下来，这是要在 PR 里补的一点。
