# Node 1 bootstrap prompt

Paste everything in the fenced block below into the Claude session on node 1
(instance 45949215, machine 52572). It is written to be self-contained.

```text
你是这次两节点 16 卡实验的 WORKER 节点（node 1）。node 0 上另有一个 Claude
实例作为 COORDINATOR，它拥有 git 提交权,你不要提交或 push 任何东西。

## 拓扑事实（已核实，不要重新推断）

- 你 = vast 实例 45949215 / machine 52572 / host LAN IP 192.168.50.19 / 8x RTX 5060 Ti 16GB
- node 0 = 实例 45781657 / machine 53171 / host LAN IP 192.168.50.17 / 同型号 8 卡
- 同一运营者(host 309585)、同一公网 IP、同一 /24 局域网。两台是**不同物理机**
  (GPU UUID 互不相同,已用阶梯显存探针验证过,不是重复出租)
- 节点间实测带宽:iperf3 单流 936 Mbit/s、4 流 930 Mbit/s、双向对称 => 1 GbE 打满
- 两台的容器网桥都是 172.17.0.x 且互不可路由,容器内没有 /dev/net/tun

## 第一步:读上下文(按这个顺序,不要跳)

    cd /workspace
    git clone https://github.com/QIU023/torchtitan_attention_residual
    cd torchtitan_attention_residual

必读,按顺序:
1. CLAUDE.md — 项目是什么、诚实性规则(非常重要,尤其"debug 规模只验证机制、
   不代表训练质量"和"loss 下降不是多卡正确性判据")
2. phase13_k3like_48b_posttrain/multinode/PLAN_5D_16GPU.md — 本次实验的目的、
   cell 矩阵、判据、风险(先读这个再动手)
3. phase13_k3like_48b_posttrain/multinode/README.md — 拓扑、overlay 步骤、
   两层传输(rendezvous vs NCCL 数据面)的区别
4. phase13_k3like_48b_posttrain/SESSION_HANDOFF_2026-07-25.md — 最新交接;
   §2 是刚修的 FSDP 梯度缩放问题(会影响你看到的 grad_norm 量级)
5. phase13_k3like_48b_posttrain/SESSION_HANDOFF_2026-07-24.md §2 — 环境配方
   (你要复现的就是这个,不要自己挑版本)

## 第二步:搭环境

直接跑现成脚本(它就是 §2 配方 + 一个单机 sanity):

    bash phase13_k3like_48b_posttrain/multinode/setup_worker.sh

它会:clone/更新仓库 + `git submodule update --init torchtitan` + 按 §2 装
依赖(fla-core==0.5.1 / torchao==0.17.0 / transformers>=5 / titan
requirements 等)+ 打印 torch 版本和可见卡数 + 跑一个 2 卡 2 步的 fsdp
sanity。结束时应打印 WORKER READY。

镜像和 node 0 是同一个模板,所以 /venv/main 里的 torch 2.12.0+cu130 是预装的,
不要重装 torch。

验收(自己核对,不要只看脚本 exit code):
- torch 2.12.0+cu130、8 卡可见
- `git submodule status` 里 torchtitan 应为 20bd4f3a
- 单元测试:`cd torchtitan && python -m pytest torchtitan/experiments/kimi_k3/tests -q`
  应为 87 passed + 66 subtests(和 node 0 一致)

## 第三步:等 overlay,期间不要试图跨节点跑 NCCL

跨节点 NCCL **必须**走 vast overlay 网络(容器网桥不可路由、无 tun 设备)。
overlay 由用户在自己电脑上用账号级 API key 创建(容器内的 key 权限不足)。
在 overlay 出现之前:

- 可以做:本节点内的单机 cell(最多 8 卡)、环境验证、读代码
- 不要做:任何 --nnodes 2 的尝试;rendezvous 能连上但 NCCL 会挂,
  容易被误判成"配好了"

overlay 就绪后,用 `ip -4 addr` 找到新出现的那个网卡名(注意 eth0 是 docker
网桥,**不是** overlay),然后等 node 0 给你 MASTER_ADDR(node 0 在 overlay 上
的地址)和端口,再跑:

    bash phase13_k3like_48b_posttrain/multinode/launch_smoke.sh 1 <OVERLAY_IFACE> <MASTER_ADDR>

node 0 会同时用 node_rank 0 起同一个脚本。这个 smoke 会校验 all-reduce 的
**数值正确性**(不只是"跑通")并打印跨节点 busbw —— 它必须先绿,再谈 5D。

## 工作方式约定

- **不要 git commit / push**,也不要改仓库里的文件。node 0 负责记录。
  你的产出是终端输出和数字,贴回来即可。
- 有 GPU 长任务用后台运行,不要阻塞。
- 报告时给原始数字(loss / grad_norm / busbw),不要只说"PASS"。
- 如果某个 cell 失败:先抓完整 traceback 再判断,并且**优先怀疑
  AttnRes cross-stage adapter**(PLAN 风险 #1:它的 payload 这次第一次跨网络),
  而不是先怀疑 mesh。
- 遇到与文档不符的现象,如实报告,不要为了让结果好看而调参。

现在开始:先读上面 5 个文件,然后跑 setup_worker.sh,把验收结果报给我。
```

## Notes for the coordinator (node 0)

- Node 1's `CONTAINER_API_KEY` is instance-scoped: it can read its own instance
  but 401s on `show instances`/overlay commands. All overlay work is the user's.
- Once the user authorizes node 0's pubkey on node 1
  (`ssh -p 11034 root@192.168.50.19`), node 0 can drive both ends directly and
  this prompt becomes optional.
- Keep node 1 read-only on git to avoid two agents pushing to the same branch.
