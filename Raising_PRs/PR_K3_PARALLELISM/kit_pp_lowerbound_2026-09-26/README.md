# PP lower bound 测量套件（2026-09-26）

用来测"PP lower bound 这条线"的三步：PR A（rank store 到下界）、统一激活存储、offload 与 balance 一起做。每一步测三样：
- cache 显存：rank store 高水位，以及每个 rank 的峰值；
- fwd/bwd 耗时；
- 通信与计算的重叠。

所有格子都是本地 recipe（`probe_lb.py`），从不提交到任何分支。

## 文件

| 文件 | 作用 |
|---|---|
| `probe_lb.py` | recipe `lb_probe`，由 09-24 的 `probe_ppmem.py` 改来。做以下几件事：①用环境变量设层数、block、dim、seq、`layers_per_stage`；②每步在 titan 重置峰值之前记录各 rank 的峰值；③统计 rank store 实际引用的字节数（按底层存储去重）及其每步高水位；④第二步起再记管理器登记的字节峰值；⑤`PPMEM_ACTION_TRACE=<step>` 时，在那一步逐个计算动作记峰值、动作后的显存和 store 字节数 |
| `run_lb.sh` | 跑一个格子：pp8、Interleaved1F1B、16 个 micro-batch、seed 42、deterministic。设 `LB_PROFILE_STEP` 时抓该步的 trace |
| `campaign.sh`、`campaign2.sh` | 一组格子依次跑：先各跑 1 步把 cache0 预热，然后每个格子用 cache0 的拷贝跑显存与一致性（N 步）和计时（抓第 8 步）。`campaign2.sh` 支持给单个格子设环境变量，写法是 `name=tree@VAR=val,...` |
| `analyze_trace.py` | 从 titan profiler 的 trace 算每个 rank 的指标：一步时长、计算、前向、反向、NCCL 驻留、重叠、暴露的通信、空闲 |
| `tab_lb.py` | 汇总：每个 rank 的峰值和 store 高水位（含最大值、均值），以及各格子的 loss 与 grad norm 是否逐位一致 |

## 切分

所有格子都用生产切分：93 层、block 12、pp8 × vp4（`PPMEM_LPS=3`，共 32 个 stage），只缩小 dim 和 seq（5060 上 dim 2048、seq 2048）。stage 划分、开 block 的位置（rank 0 和 4）、最重的 rank（4）都与 `pp_memory_model_v4b_2026-09-25.py` 的 H100 PP8×VP4 相同，每个 rank 的单位数一一对应。

## 5060 上怎么跑

```
S=<scratchpad>; K=<本目录>
LB_ROOT=$S/lb PPMEM_LAYERS=93 PPMEM_BLOCK=12 PPMEM_DIM=2048 PPMEM_SEQ=2048 PPMEM_LPS=3 \
  nohup bash $K/campaign2.sh s2 20 8 "pra=<PR A 树>" "mgr=<第二步树>@PPMEM_MANAGER=1" \
  "obl=<第三步树>@PPMEM_OFFLOAD=1,PPMEM_BALANCE=1,PPMEM_HOST_GBPS=10,PPMEM_PEER_GBPS=2" &
python3 $K/tab_lb.py $S/lb 5 s2_mem_pra s2_mem_mgr s2_mem_obl
python3 $K/analyze_trace.py $S/lb/s2_time_pra/dump
```
每个树都要打上 KDA 的 SM120 本地放宽（`ppopt_kda_lift.patch`），这个放宽不进任何提交。

## H100 上怎么跑（机器还没有）

- **8 × H100：** 直接用上面的切分（pp8 × vp4）。dim 和 seq 可以放大到接近生产：先用 `h100_probe_sizing.py`（09-24 套件）估算。
- **4 × H100：** 改成 pp4，`layers_per_stage` 取 3 时是 32 个 stage、每个 rank 8 个。这样和 5060 的切分就不再一一对应，表里必须注明。
- **环境：** 沿用 09-23 那台机器的 `venv_k3`（torch nightly、attn-gym、cutlass、torch-remat），另外要装 mooncake-transfer-engine。
  - H100 的 SM90 需要 KDA 的放宽，见 `probe_apply_h100.py`。
  - 有 RDMA 网卡时，远程池会落在对端 GPU 上；没有 RDMA 就走 TCP，池在 host 上。
- **带宽参数：** host 链路和 peer 链路的带宽（`PPMEM_HOST_GBPS`、`PPMEM_PEER_GBPS`）按实测填写，NVLink 远快于 5060 的 PCIe。
