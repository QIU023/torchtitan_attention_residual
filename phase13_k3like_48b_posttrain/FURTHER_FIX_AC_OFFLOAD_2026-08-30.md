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
