# 内部分析:PP8 惰性 communicator 挂起(不随 PR 外发)

对应 GB200 报告的 "PP8 direct-Slurm startup" 项。`PR_BODY.md` 是可贴正文;
本文件是完整机制、单节点为何不复现、以及 fix 位置的论证。

## 一、机制,三层

1. **NCCL communicator 按边惰性创建**。每条流水线边(rank r <-> r+1)的
   communicator 在第一个 P2P op 碰到它时才 bootstrap,创建本身是那对 rank
   之间的集合操作 —— 两端都到才完成。
2. **首次触碰的位置决定安全性**。DYNAMIC 模式下 `_send_meta`/`_recv_meta`
   在调度开始前按依赖顺序逐边触碰 —— 无意中充当了安全的 warmup(每次只有
   一条边,两端同时到)。STATIC 模式跳过元数据推断,首次触碰落进 1F1B 稳态的
   混合 `_batch_p2p` —— 此时各 rank 已经互相阻塞:rank r 必须先完成更早的
   recv(需要更早的边)才能到达创建边 (r, r+1) 的 op。bootstrap 沿依赖链
   **串行**。
3. **300 秒是超时不是死锁**。挂在链尾的边(5->6、6->7)等的是整条链逐边
   完成的累计时间。

## 二、为什么单节点不复现(用户问题)

同样 8 rank、同样的调度、同样的惰性链 —— 差别只在每一环的成本:

| | 单节点 | 跨节点 |
|---|---|---|
| bootstrap 介质 | shm / NVLink 握手 | socket rendezvous + NIC/IB 注册 |
| 每边成本 | 毫秒级 | 秒级;launcher/网卡选择不利时更差 |
| 链尾累计 | << 300 s,三个数量级余量 | 可超 300 s |

所以单节点上这条惰性链**每次都在走,只是快到看不见** —— 不是路径不同,是
时钟不同。这也解释了 Elfie 归类为 "launcher/runtime issue":direct-Slurm 的
接口/地址配置会放大每环成本,但根子(稳态里串行 bootstrap)在库。

推论:**单节点永远无法复现这个挂起**,复现下限是 2 节点。我们能做的单节点
验证只有"补丁无回归 + 数值逐位不变",已做。

## 三、fix 为什么放 torch 而不是 titan(用户问题)

* 触发面 = `torch.distributed.pipelining` 用户 x STATIC 模式 x 多节点。
  **Megatron-LM / DeepSpeed 自带调度实现**,直接用 `torch.distributed` 原语
  (P2POp / batch_isend_irecv)并自行规定收发顺序,不经过这段代码 ——
  在 torch.pipelining 里修,他们一行都不会执行到。影响面精确等于受害面。
* 缺口在库自己的协议层:vote 协议 warm 了 2-rank 子 communicator,漏了
  group communicator,TODO 是库作者自己写的,连修法都指定了。
* titan 侧的 warmup(两棵树已落)是**过渡**:torch fix 要走 nightly ->
  release 的传播周期。两者共存幂等无害。

## 四、私有 API 与守卫(用户问题的结论)

* fix 进 torch 本体:torch 调自己的内部符号,无私有 API 问题。
* titan 侧版本:硬导入,**不加 try/except** —— 守卫式降级会把
  "响亮的立即失败"换成"安静地退回 300 秒挂起",恰好把修复静默注销。
  符号漂移应该在 CI 里大声断掉。

## 五、本地验证方式

fix 是纯 Python(`schedules.py` 一个文件),不需要重编 torch。本机做法:
备份 `site-packages/.../schedules.py` 为 `.orig_20260802`,原地应用
`schedules_warmup.diff`。回滚:

    cp schedules.py.orig_20260802 schedules.py

验证记录(单节点 8 卡,老树 `kimi_k3_debugmodel_report_arch`,多模态
dataloader,PP8,确定性模式,2 步):

* torch fix 单独(titan 无 warmup 的 commit `e4d8118e0`):rc=0,16 条 loss,step-2 = 11.98902
* torch fix + titan warmup 共存(`gb200_fixes` tip):rc=0,step-2 = 11.98902

三种组合(无补丁 / 仅 torch fix / 双 fix 共存)step-2 全部 11.98902,逐位相同。

## 六、给 Elfie 的话术要点

她的诊断步骤("先小型 direct-Slurm 复现,确认后加 in-process eager warmup")
与 TODO 的方案一致。如果她已在 torch 侧修好,比对她的 diff 与
`schedules_warmup.diff`;如果她修在启动器/环境侧(如 NCCL_* 调参),
那只是抬成本不是消依赖,建议仍然合入 eager warmup。
