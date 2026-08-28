# K3 EP comm-backend 实验(2026-08-28,2×H100 NVL / PCIe)

EP PR(上游 4314)的 review 问 "did you try other backends?" / "make this general instead of
hardcoding"。`ep_review1` 已把 `moe_comm_backend` 做成 spec 参数(`286d139`);本文件记录在
真机上逐后端跑 ep2 的全过程:环境怎么搭、每个后端撞上什么、改了什么代码、数字是多少。
判据同 `EVIDENCE_METHOD_2026-08-25.md`:一个 seed checkpoint、每格预热一趟、每格断言
`Loading the checkpoint from`、报 step 1 / 3 / 10;判差异只看 step-2(HANDOFF_2026-08-26 §七)。

## 〇、一句话结论(截至 15:10,矩阵 5 格跑完 2 格)

| 后端 | 本机状态 | 备注 |
|---|---|---|
| `standard`(PyTorch a2a) | ✅ ep2 跑通,**step-1 与 dp2 逐位相同**(12.59885) | 矩阵主格 |
| `minimal_async_ep` | ✅ ep2 烟测 3 步跑通(12.60522 / 10.28942 / 7.66797) | 需 full AC;需 core 修 buffer 宽度(见 §四.3);矩阵格在跑 |
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

## 五、代码改动(fork `ep_review1`,本地 head `4f6462c`,尚未推)

| commit | 文件 | 内容 |
|---|---|---|
| `286d139`(已有) | `kimi_k3/__init__.py` | `moe_comm_backend` 成为 `model_registry` 参数,穿到 `make_token_dispatcher_config` |
| `20b48f5` | `models/common/token_dispatcher.py` +7/−3 | dispatcher buffer 宽度取 `routed_experts.inner_experts.dim` |
| `4f6462c` | `kimi_k3/__init__.py` +1/−1, `kimi_k3/config_registry.py` +18/−1 | `_kimi_k3_config(..., moe_comm_backend="standard")` 默认;新增 `kimi_k3_debugmodel_deepep` / `kimi_k3_debugmodel_minimal_async_ep`(后者 FullAC) |

`k3_ep`(PR head `c117ce1`)→ `ep_review1` 可 fast-forward,推法 `git push origin ep_review1:k3_ep`。

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

## 八、矩阵 `mx3_ep_backends_0828_0828_145737`(跑到一半,每格落地即补)

树 `ep_review1` @ `4f6462c`,flavor `kimi_k3_debugmodel` 系,全局 batch 8192、微批 256/dp rank,
一个 seed checkpoint,每格 warm + measure 两趟各 10 步,每格 ≈ 9 分钟。

| cell | world | backend | AC | step 1 | step 3 | step 10 | 状态 |
|---|---|---|---|---|---|---|---|
| dp2 | 2 | - | selective | 12.59885 | 7.58868 | 3.30412 | seed-ok rc=0 |
| ep2 × fsdp2 | 2 | standard | selective | **12.59885** | 7.55794 | 3.32943 | seed-ok rc=0,**step-1 与 dp2 逐位相同** |
| ep2 × fsdp2 | 2 | standard | full | | | | 跑中 |
| ep2 × fsdp2 | 2 | minimal_async_ep | full | | | | 排队 |
| ep2 × fsdp2 | 2 | deepep | selective | | | | 排队(预期 rc≠0,见 §三) |

## 九、MoonEP 准备(只读 + CPU,未动 venv,未上 GPU)

* `MoonshotAI/MoonEP` master @ `2bd860b` 已克隆;`setup.py` 是一个 `moonep._C` CUDA 扩展
  (`-std=c++20`,链 `libcuda`),`install_requires` 只有 `nvidia-cutlass-dsl==4.4.2`;构建脚本已写
  (`build_moonep.sh`,矩阵结束后再跑,因为它往 venv 加包)。
* `k3_on_4025` worktree @ `04f73f4`:`torchtitan/models/kimi_k3/tests/test_moon_ep_dispatcher.py`
  **4 passed**(MOONEP_DRAFT 定的上机前门禁,CPU 假替身)。
* 预备了 flavor `kimi_k3_debugmodel_moonep`(`model_registry("debugmodel", moe_comm_backend="moonep")`,
  未提交);`check_moonep_mesh` 要求 `dp_shard == ep` 且无 dp_replicate → 2 卡就是
  `--parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2`。
* VMM 句柄:`moonep.buffer._use_fabric_for_group` 默认 `auto`,所有 rank 都报
  `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` 时走 fabric;容器里没有 IMEX 的话 fabric
  import 会失败,备用 `MOONEP_MEM_HANDLE_TYPE=fd`(POSIX fd,单机够用)。远端读走 P2P,本机 P2P OK,
  但 MoonEP README 的所有带宽数字都是 NVLink 的;PCIe 上只能验对价,不能验吞吐。
* MoonEP 自带测试写死 `torchrun --nproc_per_node=8`,`dist_env` fixture 从 world size 取 R,
  先试 2 rank 能否直接过。

## 十、待办

1. 矩阵剩余三格 → 补进 §八 与 PR body 新 section("EP backend verify result",
   `Raising_PRs/PR_K3_PARALLELISM/PR_BODY_EP.md`;草稿已写好,只差数字)。
2. 推 `ep_review1` 与 `ep_review1:k3_ep`;PR 描述由用户粘贴(本机无 gh/token)。
3. MoonEP 上机:构建 → MoonEP 自测 2 rank → `kimi_k3_debugmodel_moonep` ep2 → token 守恒 /
   反向到达 / 与 standard ep2 数值对照(顺序不可换,MOONEP_DRAFT §上机调试清单)。

## 十一、复现

    matrix_scripts/ep_backend_matrix.sh <tag>   # TITAN=<ep_review1 worktree>;5 格清单与每格 env 都在里面
    matrix_scripts/mx3_backend.sh               # mx3.sh 的"按格选 flavor + env 前缀"版本;mx3.sh 本身未动
