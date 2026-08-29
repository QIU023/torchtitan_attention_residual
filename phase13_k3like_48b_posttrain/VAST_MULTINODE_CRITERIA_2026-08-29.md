# Vast.ai 多节点筛选条件与市场快照(2026-08-29)

## 硬性条件(用户定)

1. **同构硬件**:同一 gpu_name 的机器 ≥2 台且在同一 `cluster_id` 内——混搭
   (如 5060Ti+4060Ti)不接受;
2. 40/50 系消费卡优先(5060 Ti 最理想);
3. 单节点 2-8 卡;
4. 仅 Docker 模板(overlay 私网不支持 VM 模板)。

## 扫描命令

```sh
# 全 cluster 报价,按 (cluster, gpu_name) 聚合看同型机数
vastai search offers 'cluster_id!=None' --raw --limit 200 \
  | grep -oE '"(cluster_id|machine_id|gpu_name|num_gpus)": ?[^,}]+' \
  | paste - - - - | sort -u
# 锁定某 cluster 后看切片与价格
vastai search offers 'cluster_id=<ID> gpu_name="RTX 5090"' --raw
```

## 租用流程(三步)

```sh
vastai create overlay <CLUSTER_ID> <net_name>
vastai create instance <offer_id_A> --image <img> ... --env "-n <net_name>"
vastai create instance <offer_id_B> --image <img> ... --env "-n <net_name>"
# 容器内:NCCL_SOCKET_IFNAME=eth0;rendezvous 用 overlay eth0 的 inet 地址
```

## 快照(2026-08-29,全市场仅 4 个 cluster)

| cluster | 机器 | 判定 |
|---|---|---|
| 80(加 Alberta) | 5090×3 台(1569:1卡 $0.87/h;8133:8卡 $6.93/h;9111:8卡 $10.67/h)+ RTX PRO 6000 WS×1 | **唯一同构 40/50 系多节点解**:验证型 1569+8133 ≈$7.8/h;全尺寸 8133+9111=16×5090 ≈$17.6/h |
| 81(英) | 5060Ti×1 台 + 4060Ti×1 台 | 同构不成立(单台 5060Ti) |
| 90(冰岛) | H200×1 台 | 单机,且超范围 |
| 91 | 3090 + RTX PRO 4000 | 异构,排除 |

结论:纯 5060Ti 同构多节点市场暂无;需要时按 cluster 80 的 5090 起租。
文档注明的 "A100/H100/H200 only" 限制已过时(live API 有消费卡 cluster)。
用途:MoonEP NVLink 复测之外的跨节点基建验证(NCCL/EP 跨节点/权重同步/
Elfie 类跨节点问题复现)。
