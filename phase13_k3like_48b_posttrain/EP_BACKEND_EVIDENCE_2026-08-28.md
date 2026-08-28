# K3 EP comm-backend 实验(2026-08-28,2×H100 NVL / PCIe)

EP PR(上游 4314)的 review 问 "did you try other backends?" / "make this general instead of
hardcoding"。`ep_review1` 已把 `moe_comm_backend` 做成 spec 参数(`286d139`);本文件记录在
真机上逐后端跑 ep2 的全过程:环境怎么搭、每个后端撞上什么、改了什么代码、数字是多少。
判据同 `EVIDENCE_METHOD_2026-08-25.md`:一个 seed checkpoint、每格预热一趟、每格断言
`Loading the checkpoint from`、报 step 1 / 3 / 10;判差异只看 step-2(HANDOFF_2026-08-26 §七)。

## 先看这里:四个问题的直接回答

**本机为什么跑不了 DeepEP?** 机器有 NVLink,但**分给本容器的两张卡之间没有**:容器拿到的是主机的
GPU #0 与 GPU #5(`/dev/nvidia0`、`/dev/nvidia5`;PCI 61:00.0 / C1:00.0,分属两个 NUMA)。
`nvidia-smi nvlink --status` 每卡 12 条 26.5 GB/s 的活跃链路 —— 那是各自连到不在容器里的
桥接邻居(H100 NVL 的桥只连相邻成对的卡);`nvidia-smi topo -p2p n` 在 GPU0↔GPU1 之间是
**NS(Not Supported)**,`topo -m` 是 SYS。DeepEP v2 的节点内内核依赖 NVLink 语义:第一次 dispatch
就 `DeepEP NVLink barrier timeout` → `CUDA_ERROR_LAUNCH_FAILED (719)`;DeepEP 自带
`test_barrier.py` / `test_ep.py`(2 rank)同样失败;README 明写 "NVLink for intranode
communication"。**要一对桥接的卡(相邻编号)才能验 DeepEP / MoonEP**(§三)。

**minimal_async_ep 怎么回事?** 能跑(10 步 rc=0)但**丢 routed experts 的 `w1`/`w3` 梯度**:
逐参数探针里只有它们差(K3 上 1/500,上游 deepseek 普通专家上精确为 0),其余参数全一致;
给专家 GEMM 输入加一行 `.clone()` 即恢复。根因:专家权重梯度反向要用保存的输入 `x_RD`,
MinimalAsyncEP 的 dispatch 返回的是两槽接收 buffer 的 view,combine 反向把它盖掉了。
**上游自己的 fused_swiglu 路径同样中招**(未修 main 上 `w13` 梯度精确为 0),只是总 grad_norm/loss 上看不出来(专家梯度占总范数的万分之一),CI 又只测能跑。上游缺陷,不是 K3 的(§八.1)。

**K3 跑不起来?** 不是。K3 本身没问题;只是要选其它后端得先修两处:core 按 `model.dim`
给 dispatcher buffer 定宽而 K3 专家吃 latent 流(512 vs 1024)→ 修在 core(`20b48f5`);
`moe_comm_backend` 必填参数打挂上游单测 → 给默认值(`4f6462c`)(§四)。

**ep2 状态?** `standard` ep2×fsdp2 **step-1 与 dp2 逐位相同**(12.59885),step-2 差 4.3e-3;
full AC 对照逐位相同。**PR head 不动**:换后端在本机一个都没真正验通(用户 2026-08-28 决定),
`k3_ep` 退回 `c117ce1`(`comm_backend` 仍钉 standard),后端工作留在 `ep_review1` @ `4f6462c`;
两版 body 都备好了(§十),选项由用户定。

## 〇、一句话结论(15:25,矩阵 5 格全部落地)

| 后端 | 本机状态 | 备注 |
|---|---|---|
| `standard`(PyTorch a2a) | ✅ ep2 跑通,**step-1 与 dp2 逐位相同**(12.59885),step-2 差 4.3e-3 | 矩阵主格;full AC 对照逐位相同 |
| `minimal_async_ep` | ⚠️→✅ 修前丢 routed experts 的 `w1`/`w3` 梯度(step-1 grad_norm 低 17%,step-2 差 3.8e-1);**修后**(`maep_dispatch_owned`,§八.0)所有参数组回到同一噪声带,grad_norm 差 1.4%,step-2 差 5.2e-2 | 需 full AC;需 core 修 buffer 宽度(§四.3);根因是上游 MinimalAsyncEP 接收 buffer 被 combine 反向改写、专家 GEMM 保存的是它的 view(§八.1,deepseek 普通专家上复现,clone 即修) |
| `deepep`(v2 ElasticBuffer) | ❌ 第一次 dispatch 即 `CUDA_ERROR_LAUNCH_FAILED`,内核报 `NVLink barrier timeout` | DeepEP 自带测试同样失败:**硬要求 NVLink**,本机两卡 PCIe;需 NVLink 机器 |
| `hybridep` | 未跑 | DeepEP 另一条不兼容分支,GB200/NVL72 专属,CI 也是分开装分开测 |

## 一、机器与环境(最终态)

| 项 | 值 |
|---|---|
| 实例 | vast.ai,2 × H100 NVL 96 GB,224 核 / 1.5 TB;`nvidia-smi topo -m` 两卡间 **SYS**(PCIe + UPI 跨 NUMA,**无 NVLink 桥**);`topo -p2p r/w` 均 OK,`torch.cuda.can_device_access_peer(0,1)=True` |
| 驱动 | 580.178.04(driver_max_cuda 13.0) |
| CUDA toolkit | 镜像原装 12.8(`/usr/local/cuda`,不动)+ apt `cuda-toolkit-13-0=13.0.3-1` → `/usr/local/cuda-13.0`(编译 DeepEP/MoonEP,与 cu130 torch 同大版本;torch 的 cpp_extension 会拒绝 12.8 nvcc 配 cu130) |
| torch | **2.15.0.dev20260827+cu130**(nightly,`--index-url https://download.pytorch.org/whl/nightly/cu130`) |
| torchvision | 0.30.0.dev20260827+cu130(`--no-deps` 单独装;uv 把 torch+torchvision 一起解析会失败) |
| triton | 3.8.0+gitb252c7c4(nightly torch 自带)。**残留** `pytorch-triton==3.6.0+git8fedd49b`(我误装的),两者同一套文件,`import triton` 得 3.8.0;**不要单独 uninstall pytorch-triton**(RECORD 重叠会把 3.8 的文件删掉),要清就两个一起卸再装 triton |
| fla | flash-linear-attention 0.5.2 |
| NCCL | `nvidia-nccl-cu13==2.30.7`(nightly 自带;DeepEP v2 要 ≥ 2.30.4) |
| NVSHMEM | `nvidia-nvshmem-cu13==3.4.5`(pip;DeepEP v2 已转 NCCL GIN 后端,NVSHMEM 只是可选) |
| DeepEP | `deep_ep==2.1.0+01dc3aa`,源码 `main` @ `01dc3aa`(= 上游 `.github/scripts/install_deepep_v2.sh` 钉的提交),`TORCH_CUDA_ARCH_LIST=9.0`,内核**运行时 JIT**,所以跑的时候 `CUDA_HOME=/usr/local/cuda-13.0` 必须给 |
| torchtitan 依赖 | `.ci/docker/requirements.txt` + `requirements-vlm.txt` + `pytest==7.3.2 expecttest torch_checkpointing ninja`:grain 0.2.18, datasets 4.7.0, tokenizers 0.23.1, safetensors 0.8.0, tyro 1.0.16, spmd_types 0.2.3, av 18.1.0, einops 0.8.2 |
| lint | ufmt 2.3.0 + black 22.12.0 + usort 1.0.5 + flake8(按 `.pre-commit-config.yaml` 钉住;改动的三个文件 `ufmt diff` 干净) |
| 卸掉 | 镜像自带的 torchaudio 2.11.0+cu128、torchcodec 0.11.0+cu128(transformers 会顺手 import torchaudio,CUDA 版本不匹配直接 raise) |
| 树 | `ep_review1` worktree `/workspace/tt_ep_review1`(基点 upstream `30eb5e5`);`k3_on_4025` worktree `/workspace/tt_k3_on_4025`(MoonEP 用) |

### 重建命令(按序)

    source /venv/main/bin/activate
    uv pip install -r .ci/docker/requirements.txt -r .ci/docker/requirements-vlm.txt "pytest==7.3.2" expecttest "torch_checkpointing>=0.1.0" ninja
    uv pip uninstall torchaudio torchcodec triton
    uv pip install --pre "torch==2.15.0.dev20260827+cu130" --index-url https://download.pytorch.org/whl/nightly/cu130
    uv pip install --no-deps --pre "torchvision==0.30.0.dev20260827+cu130" --index-url https://download.pytorch.org/whl/nightly/cu130
    apt-get install -y cuda-toolkit-13-0            # 只装 toolkit,绝不碰驱动
    # DeepEP v2(约 40 s,真正的内核在运行时 JIT)
    git clone --recursive https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout 01dc3aaac82068020353dce2c302e38153c0bfaa
    CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 python -m pip install --no-build-isolation --no-deps .
    # lint
    uv pip install "ufmt==2.3.0" "black==22.12.0" "usort==1.0.5" flake8

DeepEP 格的运行时环境(来自上游 `qwen3_moe_deepep` 的 docstring,无 RDMA 网卡的机器需要):

    CUDA_HOME=/usr/local/cuda-13.0 NCCL_NVLS_ENABLE=0 EP_DISABLE_GIN=1 EP_REUSE_NCCL_COMM=0 \
    NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_DISABLE_MNNVL=1 \
    LD_LIBRARY_PATH=$SP/nvidia/nvshmem/lib:$SP/nvidia/nccl/lib     # SP = site-packages

## 二、环境踩坑时间线(全部按发生顺序,含走错的路)

1. 镜像 torch 2.11.0+cu128:装完依赖一构建配置就 `ImportError: DataParallelMeshDims`
   (`torchtitan/distributed/fsdp.py`,树 @ `30eb5e5` 要新 FSDP API)。cu128 索引最新 stable 就是 2.11。
2. 换 **2.13.0+cu130**(logbook 08 月初记录用过 2.13.0 stable)→ `DataParallelMeshDims` 有了;
   但 transformers 顺手 `import torchaudio`,镜像的 torchaudio 是 cu128 → raise。cu130 索引上
   torchaudio 只到 2.11(上游已 wind-down),直接卸掉 torchaudio + torchcodec。
3. 2.13.0 上 `minimal_async_ep` 烟测 **step 1 跑通**后死在 `torch.distributed.set_timeout`
   不存在(trainer 第一步后必调 `set_pg_timeouts`,无条件)。这是 nightly API。
4. 换 **nightly 2.15.0.dev20260827+cu130**(上游 CI 就用 nightly;logbook 08-27 那台的
   `2.14.0.dev20260802` 已不在索引)。uv 把 torch+torchvision 一起解析失败(报 torchvision 只有
   aarch64 轮子,实际 x86 cu130 轮子存在,是解析器的锅)→ 只装 torch,再 `--no-deps` 装同日
   torchvision。中间 `uv pip uninstall triton` 后误从 nightly 索引装了 `pytorch-triton 3.6.0`,
   后来 nightly torch 自己带上 `triton 3.8.0` 覆盖(见 §一 残留说明)。
5. nvcc:镜像 12.8 不能给 cu130 的 torch 编扩展,apt 装 `cuda-toolkit-13-0`(≈ 3 GB,5 分钟)。
6. DeepEP 第一次按 CI 的 `install_deepep.sh`(`hybrid-ep` 分支 @ `f725d29`)克隆 —— 那条分支
   **没有 `ElasticBuffer`**(只有 `Buffer` / `HybridEPBuffer`),而 torchtitan 的 `deepep.py` 要 v2。
   正确来源是 `install_deepep_v2.sh`:`main` @ `01dc3aa`。上游 `run_8xgpu_integration_tests.sh`
   注释明说 "DeepEP v2 and HybridEP currently live on incompatible DeepEP branches",分开装分开测。
7. DeepEP v2 编译炸在 `engram.hpp`:`ncclGinRequest_t` / `ginTrafficClass` 未定义 —— 需要
   **NCCL ≥ 2.30.4 的 GIN 设备 API**(README 明写)。stable 2.13 自带 2.29.7 没有;`uv pip install
   --no-deps nvidia-nccl-cu13==2.31.2` 后编过;切 nightly 后自带 2.30.7,重编仍过。torch 加载的
   就是 `site-packages/nvidia/nccl/lib/libnccl.so.2`(ldd 确认),所以 DeepEP 与 torch 共用一份。
8. 每次换 torch 都要重编 DeepEP(ABI),每次 ≈ 40 s。

## 三、DeepEP 在这台机器上跑不了:NVLink 是硬要求

torchtitan 侧 `kimi_k3_debugmodel_deepep` ep2 烟测在第一次 dispatch 就
`CUDA_ERROR_LAUNCH_FAILED (719)`,内核先打出 `DeepEP NVLink barrier timeout, tag: 3, nvl: 0 ...`。
为把 K3/torchtitan 排除在外,直接跑 DeepEP 自己的测试(2 rank,同一环境):

| 测试 | 结果 |
|---|---|
| `tests/elastic/test_barrier.py --num-processes 2` | `CUDA error: unspecified launch failure`(barrier 内核,GPU 100% 空转到超时) |
| `tests/elastic/test_ep.py --num-processes 2 --num-tokens 256 --hidden 512 --num-topk 4 --num-experts 32` | 同样 719,在 `ElasticBuffer.dispatch` |

DeepEP v2 README "Requirements":**NVLink for intranode communication**;"PCIe kernel support for
non-NVLink environments" 列在它的 roadmap。本机两卡 `SYS`(PCIe),P2P 读写虽 OK,但 DeepEP 的
NVLink barrier 用的是 NVLink 才有的语义。**结论:DeepEP 的验证需要 NVLink 机器(H100 SXM /
NVSwitch),与 `MOONEP_DRAFT.md` 租机验收条件(`topo -m` 两卡间须 `NV#`)一致;本文件里的
deepep 格只证明"接线到位、在 PCIe 上按 DeepEP 自己的方式失败",不证明 K3+DeepEP 的数值。**

## 四、跑之前先撞上的三件事(都进了 `ep_review1`)

1. **`_kimi_k3_config` 的 `moe_comm_backend` 必填把上游自己的单测打挂了。** `286d139` 让它
   必填,而上游 `tests/unit_tests/test_kimi_k3.py` 直接调 `_kimi_k3_config(...)` 不带这个参数
   → `TypeError`。给默认 `"standard"`。
2. **MinimalAsyncEP 要求 full AC**(`maybe_update_minimal_async_ep_config` 直接 raise:"MinimalAsyncEP
   requires full recompute")。K3 debug flavor 默认 SelectiveAC。`kimi_k3_debugmodel_minimal_async_ep`
   flavor 里设 `FullAC.Config()`,与 deepseek_v3 的 H100 recipe 同做法;矩阵多加一格
   `ep2_standard_fullac` 把 AC 的影响单独隔出来(CLI 写法是 tyro 子命令 `activation-checkpoint:full`,
   必须放在所有 `--` 选项之后,`mx3_backend.sh` 因此把 `--dump-folder` 挪到格参数之前)。
3. **core 按 `model_config.dim` 给 dispatcher buffer 定宽,K3 的专家吃的是 latent 流。**
   `update_ep_token_dispatcher_config` 里 `token_dispatcher_cfg.hidden_dim = model_config.dim`
   (debugmodel 1024),而 `KimiLatentMoE.forward` 送进 `routed_experts` 的是 `routed_down(x)`(512)。
   MinimalAsyncEP 的接收 buffer 因此 1024 宽,专家 grouped GEMM 报
   `contraction dimension of mat_a and mat_b must match`(烟测实撞)。DeepEP/HybridEP 的 buffer
   同一个来源。修在 core:按 `moe_cfg.routed_experts.inner_experts.dim`(专家输入宽度)定宽 ——
   deepseek_v3 / qwen3 的专家输入宽度就是 model dim,行为不变;K3 变成 latent_dim。真 K3 下这是
   7168 vs 3584。修完 MinimalAsyncEP 初始化日志 `hidden_dim=512`,烟测 3 步通。

## 五、代码改动(fork `ep_review1` @ `4f6462c`;**不进 PR head**)

| commit | 文件 | 内容 |
|---|---|---|
| `286d139`(已有) | `kimi_k3/__init__.py` | `moe_comm_backend` 成为 `model_registry` 参数,穿到 `make_token_dispatcher_config` |
| `20b48f5` | `models/common/token_dispatcher.py` +7/−3 | dispatcher buffer 宽度取 `routed_experts.inner_experts.dim` |
| `4f6462c` | `kimi_k3/__init__.py` +1/−1, `kimi_k3/config_registry.py` +18/−1 | `_kimi_k3_config(..., moe_comm_backend="standard")` 默认;新增 `kimi_k3_debugmodel_deepep` / `kimi_k3_debugmodel_minimal_async_ep`(后者 FullAC) |

分支状态(2026-08-28 晚):

* `ep_review1` = `4f6462c`(已推),承载全部后端工作,给下一台 NVLink 机器用。
* `k3_ep`(PR head):我曾把它快进到 `4f6462c`,用户决定**换后端全部不 work 就不更新 PR head**,
  退回 `c117ce1`。退回命令(force-with-lease 只在远端确为 4f6462c 时生效;本会话的分类器不让
  我执行,由用户跑):

      cd /workspace/tt_ep_review1 && git push origin c117ce1:refs/heads/k3_ep --force-with-lease=refs/heads/k3_ep:4f6462c16

* PR body(`PR_K3_PARALLELISM/PR_BODY_EP.md`)已按 head = `c117ce1` 重写:正文仍是 GitHub 上现行
  那份(`comm_backend` pinned to standard),尾部加 `### EP backend verify result`,只陈述在
  `ep_review1` 上试过的结果与数字,不把 ep_review1 的改动说成本分支的。

## 六、门禁

| 项 | 结果 |
|---|---|
| `tests/unit_tests/test_kimi_k3.py` | 2 passed / 1 skipped(修 §四.1 之前 1 failed) |
| `tests/unit_tests/cpu/test_config_manager.py` | 31 passed / **2 failed**:`test_cuda_graphs_reject_*`(`SystemExit: 2`)在**未改动的 `286d139` 上一模一样**,与本分支无关 |
| `ufmt diff` / `flake8` 三个改动文件 | 干净 |
| 双卡 NCCL allreduce 自检 | OK(换 NCCL 后做的) |

## 七、烟测(3 步,ep2 × fsdp2,`--debug.deterministic`,seed 42,无 seed ckpt)

| 后端 | 环境 | 结果 |
|---|---|---|
| minimal_async_ep | torch 2.13.0 + hidden_dim 未修 | step 1 前 GEMM 报 contraction dim 不匹配 |
| minimal_async_ep | torch 2.13.0 + 修后 | step 1 = 12.61903,然后 `set_timeout` 缺失 |
| minimal_async_ep | nightly + 修后 | **12.60522 / 10.28942 / 7.66797,rc=0**;step 1 约 3 分钟(triton 编译),之后 ≈ 8 s/步 |
| deepep | nightly | 719 launch failure(见 §三) |

(2.13.0 与 nightly 的 step-1 差 1.4e-2 —— 不同 torch,不可比,只说明各自跑通。)

## 八、矩阵 `mx3_ep_backends_0828_0828_145737`(5 格全部落地,15:25)

树 `ep_review1` @ `4f6462c`,flavor `kimi_k3_debugmodel` 系,全局 batch 8192、微批 256/dp rank,
一个 seed checkpoint(每格日志 2 处 `Loading the checkpoint from`),每格 warm + measure 两趟各
10 步,每格 ≈ 9 分钟。判据看 step-2(HANDOFF_2026-08-26 §七),表里同时给 step-1 grad_norm。

| cell | world | backend | AC | step 1 | step 2 | step 3 | step 10 | grad_norm s1 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| dp2 | 2 | - | selective | 12.59885 | 9.55945 | 7.58868 | 3.30412 | 13.6250 | rc=0 |
| ep2 × fsdp2 | 2 | standard | selective | **12.59885** | 9.55519 | 7.55794 | 3.32943 | 13.6250 | rc=0 |
| ep2 × fsdp2 | 2 | standard | full | 12.59885 | 9.55519 | 7.55794 | 3.32943 | 13.6250 | rc=0,**与上一行三步逐位相同** |
| ep2 × fsdp2 | 2 | minimal_async_ep | full | 12.59281 | **9.93810** | 7.56969 | 3.22479 | **11.3750** | rc=0 |
| ep2 × fsdp2 | 2 | deepep | selective | - | - | - | - | - | rc=1:`DeepEP NVLink barrier timeout` → 719(§三) |

读法:

* **standard:ep2 对 dp2 step-1 逐位相同(12.59885),step-2 差 4.3e-3**,与 08-27 那张 8 卡表
  同形(EP 只把同一组 GEMM 换执行位置)。
* **full AC 数值中性**:`ep2_standard_fullac` 与 `ep2_standard` 十步逐位相同,所以
  minimal_async_ep 那行可以直接对 `ep2_standard` 比,AC 不是变量。
* **minimal_async_ep 跑通但数值与 standard 不同,且差在反向**:同一 seed、同一权重,step-1 前向
  loss 只差 6.0e-3(bf16 combine 累加顺序量级),但 **step-1 grad_norm 11.3750 vs 13.6250(−17%)**,
  step-2 因此差 3.8e-1 —— 是 dp2→ep2(standard)那 4.3e-3 的 90 倍,不在"换并行度"的带宽内。
  warm/measure 两趟 step-1 都是 12.59281,可复现,不是偶发。反向数值路径:standard 的 combine
  是 fp32 乘 score + `deterministic_scatter_add`;MinimalAsyncEP 是 triton `reduce_topk_slots_kernel`
  / `expand_topk_grad_kernel`(`dtype=grad_out.dtype` 即 bf16)/ `topk_scores_grad_kernel`。
  归因见 §八.1:是 MinimalAsyncEP 与普通 GroupedExperts 组合的上游缺陷。
* deepep 格 seed 载入成功,第一次 dispatch 即失败,签名与 §三 一致。

### 八.0 修复后的 K3 数值格(`mx3_maepfix_0828_*`,树 `ep_review1_maepfix` @ `c3f54ba`,同一 seed ckpt)

| cell | step 1 | step 2 | step 3 | step 10 | grad_norm s1 | s2 |
|---|---|---|---|---|---|---|
| ep2 × fsdp2 standard | 12.59885 | 9.55519 | 7.55794 | 3.32943 | 13.6250 | 13.3125 |
| ep2 × fsdp2 minimal_async_ep(修前,§八) | 12.59281 | 9.93810 | 7.56969 | 3.22479 | 11.3750 | 12.4375 |
| ep2 × fsdp2 minimal_async_ep(**修后**) | 12.59281 | 9.50324 | 7.45474 | 3.22755 | 13.4375 | 13.9375 |

修后:step-1 grad_norm 与 standard 差 1.4%(修前 17%),step-2 差 5.2e-2(修前 3.8e-1);逐参数探针
(`grads_fix_kimi_k3_*.json`)全部分组落在同一噪声带(experts 组 max rel 3.1e-2 / 中位 6.9e-3,attention
中位 2.0e-2,router 中位 3.7e-2),不再有任何一组梯度丢失。step-1 前向的 6.0e-3 修前修后相同,
来自 MinimalAsyncEP 自己的 combine 内核与 standard 的 fp32 乘 + scatter_add 之间的累加差异
(K3 的 combine 之后紧接 `routed_norm`,对这一点比 deepseek 敏感:deepseek 上同一差异只有 1e-5),
不是本 bug。5.2e-2 仍大于 dp2→ep2(standard)的 4.3e-3,是 K3 debug 模型对这个前向差异的放大,
判据上应写成"与 standard 在 bf16 归约序层面一致、无梯度丢失",不写"逐位"。

### 八.1 归因:是 MinimalAsyncEP 的,不是 K3 的(逐参数梯度探针)

三步收窄,工具在 `matrix_scripts/ep_backend_probe/`(含四份原始 JSON):

1. **内核自洽**:`tests/unit_tests/gpu/test_minimal_async_ep_kernels.py` 6 passed。
2. **deepseek_v3 两后端 2 步对照**(ep2×fsdp2,full AC,`--training.disable_cuda_graphs`,
   seed 42;spmd_types 与 partial_dtensor 各跑一遍,数字完全相同):

   | 后端 | step 1 loss / grad_norm | step 2 loss / grad_norm |
   |---|---|---|
   | standard | 8.00752 / 4.2267 | 6.11044 / 4.0909 |
   | minimal_async_ep(上游 flavor,带 fused_swiglu) | 8.00778 / 4.2259 | 6.11541 / 4.0995 |

   看起来 deepseek 没事 —— **这个判据看不出这个 bug**(见下:专家梯度只占总范数的万分之一)。
   上游 `deepseek_v3_debugmodel_minimal_async_ep` 强制 `enable_fused_swiglu`(专家换成融合 override),
   K3 是 SiTU-GLU 用不了它;我一度据此推断 fused 路径没事,**错了**,见第 4 条。
3. **逐参数梯度探针**(`grad_probe.py`:同一 seed ckpt、同一个微批、一次前向+反向,dump 每个
   参数的梯度范数;`grad_diff.py` 分组比):

   | 树 / 专家实现 | 结果 |
   |---|---|
   | K3 `kimi_k3_debugmodel` vs `_minimal_async_ep` | 其它所有组(router、latent 投影、attention、vision、shared)中位数 ≤ 6e-2、多数 1e-2 内;**`routed_experts.inner_experts.w1_EFD` / `w3_EFD` 每层都是 standard 的 1/500**(如 layer 23:0.651 → 0.0012);`w2_EDF` 一致。总范数 23.508 vs 19.732 |
   | deepseek `dsv3_std` vs `dsv3_maep_plain`(**普通 GroupedExperts**,无 fused) | 其它组 ≤ 2.5e-3;**`w1_EFD` / `w3_EFD` 精确为 0.00000**(standard 0.0018–0.0092);`w2` 一致 |
   | 同上,但在上游 `GroupedExperts.forward` 临时加 `x_RD = x_RD.bfloat16().clone()` | **全部一致**:experts 组 max rel 3.1e-4,总范数 4.44737 vs 4.44740 |

4. **上游自己的 fused 路径(未修 main,`dsv3_std_fused` vs `dsv3_maep_fused`,`run_grad_probe_fused.sh`)**:
   融合权重 `w13` 的梯度在每一层**精确为 0.00000**(standard+fused 0.0054–0.0115),其它组 ≤ 6.4e-4;
   总 grad_norm 4.44490 vs 4.44487,loss 4.005814 vs 4.005869 —— 指标层面不可见。
   **结论:上游 MinimalAsyncEP 的所有训练配置都在静默地不更新 routed experts 的 gate/up 投影。**
   没被发现的原因:症状静默(其余参数照常学、loss 照常降)、专家梯度占总范数万分之一、
   唯一 CI 格(`deepseek_v3_fsdp+cp+tp+minimal_async_ep`)只断言跑完、内核单测测不到 autograd
   保存张量的生命周期。MinimalAsyncEP 2026-06-13 进上游(#3561),当前 dispatcher/experts 兄弟
   节点结构来自 07-16 的重构(#3859)。

5. **编译路径(`experiments/graph_trainer`,aot_fx_trace,`memory_policy full`,`run_grad_probe_graph.sh`)**:
   修前 `w13` 每层精确 0;修后 0.01149 / 0.00811 / 0.00638,与 eager fused standard 逐位相同。
   eager 与编译两条路径都中招,修复都生效。

6. **CI 预检**:pyrefly 0.45.1 在 `ep_review1`(含修复)与 `maep_dispatch_owned` 上均 0 errors;
   `test_integration_test_definitions.py` 11 passed。H100 上 dp2 / ep2 standard / ep2 fullac / ep2 maep
   的 step-10 显存 7.37 / 8.07 / 7.26 / 7.29 GiB(256-token 微批,tps 无性能意义,不进 body)。

**机制**:专家权重梯度 `dW1 = x_RDᵀ·dgate` 用的是 autograd 保存的 GEMM 输入 `x_RD`;MinimalAsyncEP 的
dispatch 直接返回全局接收 buffer(`_HIDDEN_RECV_BUFFER_COUNT = 2`,每次 `_copy_rows_to_peers_and_wait_cuda`
轮换一个槽)的 view。full-AC 下一层的顺序是:重算前向 dispatch(槽 k)→ combine(槽 k+1)→
**combine 反向的 `_dispatch_to_experts`(槽 k+2 ≡ k,把 `x_RD` 盖掉)**→ 专家反向读 `x_RD` → 错。
`w2` 用的是 `_situ_glu`/silu 的输出(独立张量),`dx_RD = dgate·W1 + dup·W3` 不需要 `x_RD`,所以只有
`w1`/`w3` 中招 —— 与两棵树、plain 与 fused 的探针全部吻合。

**结论**:`minimal_async_ep` 在 K3 上"能跑"但**丢 `w1/w3` 梯度**,根因在上游 MinimalAsyncEP
与普通 `GroupedExperts` 的组合;正确的修法是 dispatch 交出自有张量(或专家侧 clone),不属于
EP PR。PR body 如实写这一段;K3 的 flavor 保留(它就是复现路径)。临时 clone 补丁已撤。
**修复已实现并验证**:fork 本地分支 `maep_dispatch_owned`(基点 upstream/main `b953a3f`,commit `74a89f8`,
`MinimalAsyncEPTokenDispatcher.dispatch` 在 grad 开启时 clone 交出的行),deepseek 普通专家探针 experts 组
max rel 3.1e-4(修前 1.0);cherry-pick 到 `ep_review1_maepfix`(`c3f54ba`)后 K3 探针 experts 组 max rel
3.1e-2 / 中位数 6.9e-3(修前 0.998),总范数 23.508 vs 23.245(修前 19.73)。kit:`Raising_PRs/PR30_*`。

## 九、MoonEP 上机:同一个硬件条件把它也挡住了

| 步骤 | 结果 |
|---|---|
| 构建 | `MoonshotAI/MoonEP` master @ `2bd860b`,`pip install -e .`(CUDA 13.0,`-std=c++20`,加 `nvidia-cutlass-dsl==4.4.2`),49 s,`from moonep import Buffer` OK;torch 未变 |
| CPU 门禁 | `k3_on_4025` @ `04f73f4`:`test_moon_ep_dispatcher.py` **4 passed**(假替身) |
| MoonEP 自测 2 rank | `tests/test_planning.py` 第 2 个用例起全部 **`CUDA error nvl_shared_buffer.cuh:403 'operation not supported'`** = `cuMulticastGetGranularity`;`MOONEP_MEM_HANDLE_TYPE=fd` 无区别;其余五组同样在 Buffer 构造处失败 |
| torchtitan `kimi_k3_debugmodel_moonep` ep2 × fsdp2 | 配置 / spec(`MoonEPTokenDispatcher` + `MoonEPGroupedExperts`,dim 512)/ `check_moonep_mesh`(dp_shard 2 == ep 2)全部通过,trainer 建好、structured logging 起来,**在 MoonEP `Buffer(...)` 构造处报同一个 403** |

原因:MoonEP 的 dispatch/combine 建在 **NVSwitch SHARP multicast** 上(`buffer.py:_create_nvl_multicast_view`,
`assert nvl_multicast_supported()`),而且连普通 VMM 张量的 padding 粒度都要先 `cuMulticastGetGranularity`
(`nvl_granularity_max` 取所有句柄类型的最大值);本容器这对卡(主机 GPU #0 与 #5)之间没有 NVLink,
multicast 对象不可用。**与 DeepEP 是同一个结论:要一对桥接的卡。** 在那之前本机能给 MoonEP 的
只有 CPU 假替身测试(已过)。

准备好的东西(换到 NVLink 对上直接用):

* flavor `kimi_k3_debugmodel_moonep`(`k3_on_4025_local` 本地提交,未推,见 §十);
* `matrix_scripts/moonep_onbox/`:`build_moonep.sh`、`run_moonep_selftests.sh`(6 组自测,2 rank)、
  `run_moonep_smoke.sh`(ep2 3 步)、`run_grad_probe_moonep.sh`(standard vs moonep 逐参数梯度 =
  反向到达 + 槽内专家梯度);`matrix_scripts/moonep_matrix.sh`(dp2 / ep2 standard / ep2 moonep 数值格)。
  验证顺序不变:自测 → 烟测 → token 守恒(step-1 loss 对 standard)→ 梯度探针 → 数值格。
* 注意:`moonep` 是从 session 的 scratch 目录 editable 安装的;换机器重装,或把源码放到
  `/workspace` 下再 `pip install -e .`。

## 十、状态与待办(2026-08-28 晚,最终)

**PR head 怎么办(用户定)。** 用户不 force push 打开中的 PR 分支,所以 `k3_ep` 现在就是 `4f6462c`
(= `c117ce1` + `286d139` spec 参数 + `20b48f5` core 宽度 + `4f6462c` flavor/默认值)。两个选项,
两版 body 都在 `Raising_PRs/PR_K3_PARALLELISM/`:

| 选项 | head | 贴哪版 body |
|---|---|---|
| (a) `git revert` 三个提交再推 | 回到 c117ce1 的内容,多一个 revert commit | `PR_BODY_EP_head_c117ce1.md`(正文 = GitHub 现行,尾部只陈述在 ep_review1 上试过的结果) |
| (b) 就留在 `4f6462c`(推荐:三处代码本身正确,spec 参数正是 reviewer 要的) | 4f6462c | `PR_BODY_EP_head_4f6462c.md`(Summary 说 backend 是 spec 参数,Changed files 含 core 一行,尾部 verify 段) |

**本机还能推进什么。** 只有 `minimal_async_ep`:不需要 NVLink,bug 已定位(§八.1),可在本机修并复验
K3 ep2。deepep(NVLink)/ hybridep(GB200)/ moonep(NVSwitch multicast)本机无法推进,脚本已备好
(`ep_backend_matrix.sh`、`moonep_onbox/`),换到 `topo -p2p n` 为 OK 的一对卡上直接跑。

待办:
1. 用户选 (a)/(b),贴对应 body 到 4314。
2. MinimalAsyncEP 修复(本机可做):dispatch 交出自有张量 → deepseek 普通专家探针 `w1/w3` 回到 3e-4
   → K3 `ep2 × minimal_async_ep` 数值格与 standard 同量级 → 单独 PR(带 deepseek 复现数字)。
3. NVLink 对到手后:`ep_review1` 上重跑 `ep_backend_matrix.sh`,再跑 `moonep_onbox/`。

## 释放前清单(2026-08-28 晚,已执行)

本机只在 worktree 里、未推 fork 的三个提交已以补丁形式进 logbook:

| 提交 | 内容 | 补丁位置 |
|---|---|---|
| `74a89f8`(分支 `maep_dispatch_owned`,基点 upstream/main `b953a3f`) | MinimalAsyncEP dispatch 输出在 grad 下 clone | `Raising_PRs/PR30_*/0001-minimal_async_ep-*.patch` |
| `c3f54ba`(`ep_review1_maepfix`) | 同一补丁 cherry-pick 到 `ep_review1` 之上 | 同上,`git am` 到 `ep_review1` 即得 |
| `12bad1b`(`k3_on_4025_local`) | `kimi_k3_debugmodel_moonep` flavor | `phase13_k3like_48b_posttrain/matrix_scripts/moonep_onbox/0001-kimi_k3-a-moonep-debug-flavor.patch` |

fork 上已推的:`ep_review1` = `k3_ep` = `4f6462c`。logbook 全部推送。本机再无 EP 方向可做的事:
standard 验完、minimal_async_ep 修好并在 eager/编译两路径验证、deepep/moonep 需 NVLink 对、hybridep 需 GB200。

## 十一、复现

    matrix_scripts/ep_backend_matrix.sh <tag>   # TITAN=<ep_review1 worktree>;5 格清单与每格 env 都在里面
    matrix_scripts/mx3_backend.sh               # mx3.sh 的"按格选 flavor + env 前缀"版本;mx3.sh 本身未动


---

# 第二台机器(2×H100 SXM,NV18 真 NVLink)补记 — 2026-08-28 晚

上文四个"本机跑不了/未验"在真拓扑上全部闭合。盒:vast.ai 2×H100 80GB HBM3,
`topo -m` GPU0↔GPU1 = **NV18**;环境按上文重建命令逐条复刻(torch nightly
2.15.0.dev20260827+cu130、cuda-toolkit-13-0、DeepEP main @ 01dc3aa;新增一坑:
无 RDMA 盒需 `apt install libibverbs-dev` 供 `infiniband/mlx5dv.h`,
legacy internode 内核编译要头文件,运行时仍只走 intranode)。

## 五格矩阵(mx3_h100nvl_0828_183356,树 ep_review1 @ 4f6462c1)

| 格 | s1 | s3 | s10 |
|---|---|---|---|
| dp2 | 12.59951 | 7.45599 | 3.26481 |
| ep2_standard | **12.59951**(=dp2 逐位) | 7.43228 | 3.30036 |
| ep2_standard_fullac | 12.59951 | 7.43228(=standard 逐位) | 3.30036 |
| ep2_minimal_async_ep | 12.59768 | **7.56392** | 3.29721 |
| ep2_deepep | 12.59438 | 7.46660 | 3.24408 |

- **DeepEP v2 首次真验通过**(上台 SYS 拓扑的 719 超时确系拓扑,非代码)。
- maep 的 s3 偏离(+0.13)即未修 w1/w3 梯度丢失的宏观指纹。

## 逐参数探针裁决(std vs maep,未修/已修,同 seed 3 步)

- 未修:榜首全为 `routed_experts.w1/w3`,相对差 **0.998**
  (std 梯度 0.70–0.87,maep 0.0015–0.002 —— 近乎全丢),NVLink 真
  symm-mem 路径复现;
- 打 PR30 补丁后:榜首降至 2.3–2.5e-1 的微量参数(A_log 5e-5 量级),
  无参数族系统性归零。
- **结论:owned-dispatch fix 必要**,双拓扑复现;PR30 独立成立,EP PR
  默认 standard 不依赖之。

## review 5053724179 三点落码 + 重跑暴露的一笔

`97cb79909`:hidden_dim 配置化(工厂参数,K3 传 latent_dim,core None 守卫
回退 model dim)、set_moe_sharding TODO、set_decoder_sharding why 注释。
`18d1f182a`:工厂 maep 分支漏传 hidden_dim(重跑第一击即中,contraction
mismatch)——config-first 的完备性补全。

## 评审尖透明性重跑

std_r2 s10=3.30036、deepep_r2 s10=3.24408、maep_r3 s1/s3=12.59768/7.56392
——三格均与矩阵行**逐位同值**,取宽改动数值透明。

## 分支终态

`ep_review1` = `k3_ep` = **18d1f182a**(已快进同步,2026-08-28 用户指令)。
MoonEP 依指令在上述全部落地后开工。

## MoonEP:核心目标在 NVLink 真机闭合(2026-08-28 深夜)

链条:CPU 假世界门(4/4)→ 真库构建(master @ 2bd860b,cutlass-dsl 4.4.2,
CUDA 13)→ 官方自测 **6/6**(dispatch 12/combine 14/grad_reduce 12/
prefetch 14/planning/e2e;e2e 需 `torchrun -m tests.test_e2e` 模块方式,
pytest 收集器会双重 init PG,是测试写法非内核问题)→ titan 集成冒烟
ep2×fsdp2 **一次通过**(12.47486 → 9.19460,exit 0;盲写 dispatcher/
experts 与真内核首次接触即成)→ 逐参数梯度探针对照 standard:榜首
2.1–2.6e-1 相对差、全为 1e-4 量级噪声带参数,**无 routed_experts 族**
——与"已修 maep vs standard"同噪声轮廓;token 守恒与反向到达成立。

ON-BOX 三疑点判明:Buffer 组绑定/隐藏 dtype/cu_seqlens 布局在冒烟中
全部按本地 Claude 的真实合同重写版成立,tripwire 零触发。moonep flavor
已入集成树(`4961dec31`)。钉住的 moonep 提交本身是 "Support MXFP4
expert weights in remote prefetch"——与 QLoRA 线的未来交点已在上游。

**后端章节全线终态**:standard(双拓扑)/deepep(NVLink 首验)/
minimal_async_ep(跑通+上游 bug 裁决+PR30)/moonep(核心目标,全链闭合)。

## review 三条的终局:一个消融推翻了我们自己的注释(be477938d)

第三条("why do you need this?")最初用注释回应——"decoder 级分布使 MoE
边界激活可被再分布,没有它 EP 无物可作用"。消融证明这句话是**错的**:
本地 PCIe 盒 ep2×fsdp2、确定性同 seed、十步,删掉
`set_decoder_sharding_config` 前后 **s10 逐位同值(3.60159)**——MoE 模块
的 in_src 边界自己就会提升 plain 输入,decoder 级声明在 EP-only 路径是
no-op。处置:删行 + import/docstring 同步清理;更小的 diff 即最好的回答。
Reviewer 的直觉正确。

同笔:`enable_sp` 从调用点字面量升为真参数(默认 False,本 PR 唯一取值,
透传至 set_moe_sharding_config)——第二条"remove the hardcoding"按字面
兑现;SP 本体保持与 EP PR 零耦合,归 TP PR 之后的独立后续。

方法学备注:消融首轮两腿同值实为**假阳性**——脚本第 2 腿的字符串替换被
并行进行的穿线编辑扑空,两腿实际都带调用;靠"同值可疑"察觉,手动重做
第 2 腿后结论才成立。教训:同一工作树上不要让脚本编辑与人工编辑并行。

`ep_review1` 现头:be477938d。`k3_ep`(PR 头)保持 18d1f182a,同步待用户。

## 回流:ep_review1 的后端工作全部进集成树(收官)

五笔 cherry-pick 到 `k3_on_4025`(宽度按 latent 配置化、工厂 maep 分支修复、
deepep/maep flavor、decoder 调用删除+enable_sp 参数化;与 MoonEP 接线在
集成树合流——现在四后端 spec 全建:standard/deepep/minimal_async_ep/moonep,
buffer 一律按 latent_dim 配置)。59+186 测试全绿,ep2 十步 rc=0。
**新树 EP 后端工作至此全部结束**;H100 盒证据已归档,可释放。
