# Further fix 两件:AttnRes AC 复用 + adapter cache offload(2026-08-30)

用户手读报告三项 further-fix 的前两项,按指示直接落在 k3_on_4025。
最终入树两个提交:`2be96d23e`(ac_reuse_attention)、`b85c5b605`
(attn_res_cache_offload),经真尖(fdbf0e71c 系)重排与重验。

## 一、ac_reuse_attention:+29% 吞吐换 +11% 显存,loss 逐位

机制:fla 的 KDA 内核是 opaque 的 autograd.Function(torch.ops.fla 无注册,
per-op SAC policy 存不了它的输出),整层包 selective AC 时反向把内核重跑
~2x。旋钮开启后 AC 只包每层 MoE/FF(mm 隔一存一的收益都在参数大头),
attention 与 residual 数学留区外、激活直接复用。默认关。

真尖 A/B(debugmodel,dp1,seed 42 deterministic,10 步):

| 臂 | s1/s3/s10 | 稳态显存 | 稳态 tps |
|---|---|---|---|
| off | 12.55234 / 6.37338 / 3.46714 | 12.48 GiB | 1066 |
| reuse | **逐位相同** | 13.88 GiB | **1379(+29%)** |

## 二、attn_res_cache_offload:数值逐位,debug 规模显存零收益(如实)

机制:own-rank commits 沉积时异步 D2H 进 pinned host、消费时 H2D 回迁。
安全性:own-commit 的消费链走 Capture/Augment slot 桥(梯度按槽记账),
与共享存储无关,host 往返值等价;relayed 块不动(SEND_B 排水依赖
attached)。默认关。

真尖 A/B(debugmodel_32l,pp4×vp2 Interleaved1F1B,cache 默认开,4 卡):

| 臂 | s1/s3/s10 | 稳态显存 | tps |
|---|---|---|---|
| off | 12.36597 / 6.12694 / 3.32098 | 9.02 GiB | 223 |
| offload | **逐位相同** | 9.02 GiB | 222 |

**为什么没省显存**:被 offload 的 detach 副本与 attached 孪生共享存储,而
attached 侧作为 stage 输出被 PP 的输出暂存(为 SEND_B)持有到该 mb 反向——
cache 引用不是最后一个持有者,挪走它释放不了底层存储。真正的赢面在
**stage 输出暂存本身的 offload**(schedule 级,v2 方向);v1 铺好了安全的
host 往返通道并证明逐位无害。

## 过程事故与修正(全记)

1. **过期检出**:本地 tt_4025 worktree 落后 origin/k3_on_4025 **116 个提交**
   (muon/mtp/LoRA/QAT/QB/MoonEP 均已合树,adapter 已带位置 carrier 与
   config 布局探测,DEP 与 delta transport 已默认开,dyncp flavor 已删)。
   当天早些基于旧检出宣布的"delta transport 在树上从未跑通/探测回退"等
   树债系旧基座幻影,予以收回;仍为真债的只有 doc-mask 未回灌与 KCP 仍指
   裸 fla(归 #27 rebase)。教训:worktree 使用前必须 fetch 对时。
2. 旧基座上的首轮 offload A/B 连环揭露(marker 探测回退 → blocks= kwarg
   TypeError)对旧基座为真,对真尖不适用;旧基座上的 adapter 回灌提交在
   rebase 时被正确判冗余丢弃。
3. 一次 `git reset --hard` 误操作抹掉四个提交,reflog 完整找回。教训:
   reset 前先 log 目标。
4. 32l 文本配置存在与 cp4 同源的启动级非确定(同码两启动 s1 不同,triton
   缓存热后消失)——autotune 吸引子,量数值前先热缓存(与
   gate-inductor-cache 教训同族)。

## 复现

`torchtitan_recipes/k3_further_fix.py`(未入库,备份于 scratchpad
recipes_only/;四个入口 k3_ac_off/k3_ac_reuse/k3_cache_off/
k3_cache_offload),logs:/workspace/tip_revalidate/。真尖 loader 只认
可导入模块路径,scratch 目录模块不再可用;另 scratchpad 根部的
sitecustomize.py 旧探针会污染带该目录进 PYTHONPATH 的进程,勿再把
scratchpad 根加进 PYTHONPATH。

## 三、用户校正后的重做(报告原文到手,2026-08-30 深夜)

### 3.1 报告条目 1 的正确语义:AttnRes 计算包 checkpoint(显存中性),已实现

早间的 `ac_reuse_attention` 是"多存换算力"的性能旋钮,不是报告条目;报告
要的是反向:**residual 数学整体包 checkpointing,每层保存集与标准残差架构
一致**(block 表示边界层生成一次、栈内共享)。已实现为 `c67f5b2e1`:
`_apply_attention_residual` 变 checkpoint 入口(use_reentrant=False),
fp32 全栈上抛的中间量((N+1)×T×D ×2 处/层)反向重算。

no-AC 配置(wrap 收益可见的路径)HEAD~ vs HEAD,debugmodel dp1:

| 臂 | step 1 | 结局 |
|---|---|---|
| 无 wrap | loss 12.55234,11.84 GiB | **step 2 CUDA OOM**(16GiB 卡) |
| 有 wrap | loss **逐位相同**,11.01 GiB(-0.83) | 10 步全程,稳态 14.24 GiB |

同一配置无 wrap 直接 OOM、有 wrap 跑完——判定封顶。selective AC 下两者
数值也逐位(中间量本就不存,wrap 是嵌套 no-op 语义)。

**PR 结构(用户定)**:此 wrap 为独立 PR;`ac_reuse_attention` 旋钮不属于
报告条目,是否随行由用户在切分支时定夺。

### 3.2 报告条目 2 的正确语义:PP rank 间激活均衡(Mooncake 卸到 peer 显存)

判据 = 各 rank 峰值趋平、最大 rank 下降。干净的四 rank 基线
(debugmodel_32l,pp4×vp2,10 步):

| rank | 峰值 reserved |
|---|---|
| 0 | 6.51 GiB |
| 1 | 2.55 GiB |
| 2 | 4.09 GiB |
| **3** | **9.02 GiB** |

3.5 倍散布确认,但 debug 尺度下最重的是**末段 rank3**(vocab 163840 的
logits/loss 主导),而非报告暖机故事里的 rank0——暖机不均衡要在激活主导的
大形状下才是主项。v2 设计(stacked on PP PR):stage 输出暂存级跨 rank
卸载,本机无 Mooncake 用 NCCL P2P/host 反弹作 reference;卸载策略按实测
per-rank 峰值(而非固定"early→late")选源与宿。v1 的 host 通道保留为基建。
探针:torchtitan_recipes/k3_further_fix.py 的 atexit [PEAK](metrics 只打
rank0,曾因此漏看分布)。

## 四、v2:PP rank 间激活均衡,Mooncake 本尊接入(2026-08-31,c111126b8)

不造轮子:`mooncake-transfer-engine` pip 包直接可用(cu12 runtime 由守卫
ctypes 预载,零环境变量);`get_local_topology()` 判 HCA——集群 RDMA 直达
peer GPU(报告语义),本机无 HCA 自动落 TCP + pinned host 缓冲。两个引擎
坑:RPC 端口无视请求自选(段名必须回读 `get_rpc_port()`);TCP 传输只吃
host 内存。机制:`saved_tensors_hooks` 包 stage forward,pack 停放+释放、
unpack 取回;first-fit 合并空闲表管池;旋钮 `pp_balance_source_ranks/
dest_rank/pool_gib/staging_mib/min_tensor_mib`。

### 数值判定链(五组实验,机制终审通过)

| 臂 | s1/s3/s10 | 结论 |
|---|---|---|
| 基线 ×2 | 12.36597/6.15894/3.34661 | 自逐位 |
| ppbal ×2 | 12.36597/6.15438/3.35641 | 自逐位、对基线 s3 起分叉 |
| only2d / only3d 二分 | s3=6.13528 / 6.16063 | 任何停放子集都分叉 → 非类别问题 |
| 哑分配控制 | 与基线三步逐位 | 单次预分配不动 → 泛分配敏感排除 |
| **park-and-keep** | **与基线三步逐位** | 全量传输、不提前释放 → **传输机件数值惰性** |

**定性(独立发现)**:分叉唯一来源是"提前释放"改变步内分配器布局,而
KDA triton 反向用原子累加、按地址序归约——**任何改变激活生命周期的显存
优化在 KDA 模型上都不可能对基线逐位**(与 cp4 双吸引子、32l 启动抖动同
族)。验收因此改为:自复现逐位(ppbal 具备)+ 曲线在模型自身包络内
(s3 偏差 ~2e-3,与 grad-norm 精度实验同量级)+ 峰值目标。
`K3_PPBAL_KEEP_LOCAL=1` 保留为数值隔离开关,一条命令向 reviewer 重演
"传输精确、释放才移位"。

### 显存现状与 v3 方向

debug 形状下源 rank 峰值仅 −0.08 GiB(rank3 9.60→9.60@mb8):被停放的
保存张量与 PP send/输出暂存别名,叠加末段瞬态 logits 主导峰值。真收益
需要激活主导的形状(48B/长序列)或叠加暂存级释放;传输层与数值判定已
就位,形状扩展属 #27 rebase 后的验证项。
