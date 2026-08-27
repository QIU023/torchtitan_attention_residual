# TP 全混合矩阵与边界修复链(2026-08-27)

判据见 `EVIDENCE_METHOD_2026-08-25.md`。

树:`k3_on_4025` / `main` @ `69eae594b`(集成树,TP 去门之后)。
flavor `kimi_k3_debugmodel`(text flavor 已废除,默认即多模态)。全局 batch 8192,
微批 512/dp rank,seq 512。单一 seed,每格断言 `Loading the checkpoint from`,
每格预热后读第二遍。矩阵目录 `mx3_tp_full_0827_194548`。

## 一、这张表回答的问题

TP 声明式在 a4079c014 去门之后,只单独对着 dp 测过(tp2 单轴 5.2e-4)。这张表按
用户给的形状把**全部并行同时打开**:去掉 dp&ep 的一组(tp cp pp 三轴齐开),加上
dp&ep 的组里 pp/cp 二选一(8 卡装不下五轴齐开)。TP-only 与 TP×DP 的表**不在这里**
—— 那两张属于切出去的 TP 分支,PR 的证据不能带别的轴。

## 二、结果

| cell | world | 判定 | step 1 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | 1 | 基线 | 12.46779 | 6.49181 | 3.46779 |
| tp2 | 2 | 通过 | 12.46134 | 6.41197 | 3.36239 |
| dp2 x ep2 | 2 | 通过(含 clip 收编回归,逐位一致) | 12.47407 | 6.59332 | 3.40438 |
| tp2 x cp2 x pp2vp2 | 8 | 设计性守卫(跨段塔 + CP,见第五节) | - | - | - |
| tp2 x pp2vp2 x fsdp2 x ep2 | 8 | 通过 | 12.49181 | 6.60328 | 3.43105 |
| tp2 x cp2 x fsdp2 x ep2 | 8 | 通过 | 12.48765 | 6.69607 | 3.42631 |

两个八卡组合格都是在修复链末端**同 seed 重跑**出的数字,不是首跑侥幸。

## 三、修复链:TP 让主流变成 DTensor 之后,谁没准备好

TP 声明式的副作用是解码器主流(和塔的输出)从 plain tensor 变成
DTensor(Replicate@tp)。aten 算子靠隐式复制全程容忍这件事 —— **这正是 tp2 单轴
从不炸的原因** —— 而三类"内核毗邻"代码不容忍,每一类都用 flat KDA 内核调用点
已验证的同一副平衡 unwrap/rewrap 修掉:

1. **KDA 的 KCP 路径**(`kda.py`):conv halo 交换和 chunk kernel 的状态交换是裸
   c10d 集合通信(`_allgather_base_` 拒绝混型),triton 内核也不认 DTensor。
   `conv_with_halo` 与 KCP 内核调用点各自落地/回包。第一版把 unwrap 放在模块
   入口,被投影的 DTensor 权重否决(plain x 撞 DTensor weight 的 mm 混型),
   回退成调用点配对 —— 流保持 DTensor 走投影,只在内核紧邻处落地。
2. **MLA 的 Ulysses 路径**(`sharding.py`):`all_to_all_single` 没有 DTensor
   sharding 策略。进程在形状读取**之前** unwrap(DTensor 报全局头数,会把
   `h_cp` 算大);回程的 all-to-all 还要再落一次 —— inner_attention 的 local_map
   会把注意力输出重新包成 DTensor。模块出口回包到主流 placements。
3. **CP 特征 splice**(`model.py`):选片在 locals 上消费,所以选片就得在
   localized 视图上做 —— 在 DTensor 上切片会让 SliceBackward 把 plain 梯度加进
   DTensor 零垫。`dep_vision_stage` 的第二处 splice 同构补齐。

## 四、两处模型无关修复(`distributed/`,5f509354e)

- **`clip_grad_norm_` 的 EP 旁路收编**:`ep_enabled` 曾短路进
  `_clip_grad_norm_with_ep`,只按 ep/非 ep 二分。TP 下非 EP 侧自身就有两个
  mesh(声明过的解码器参数在 (fsdp, tp),复制的塔在 (fsdp,)),
  `get_total_norm` 的 stack 拒绝混跨。mesh 分组是同一副二组合并代数的一般化,
  删除专用函数,所有组合走同一条路。**回归**:dp2_ep2 改后重跑,三步全部与
  改前逐位一致。
- **`add_zero_valued_dependency` 跟随 output 的域**:FSDP 保活边把 plain 标量
  加到 DTensor 输出上,反向把 DTensor 梯度灌进 plain 生产者(塔),在
  SliceBackward 处混型。依赖边的值本来就乘零,只有图边需要旅行 —— 现在按
  output 的域 wrap/unwrap。定位靠 `--debug.detect_anomaly` 两次指认同一个
  SliceBackward:第二次输入侧已落地仍然指它,才看出混型来自**下游**的保活边。

## 五、ep+tp 的 MoE 区与设计性守卫

- **ep+tp 组合声明**(`model.py`):区内按 SP 跑(不开 SP 就复活
  S(0)→P(sum) 的拒绝,老注释的警告是真的),边界 override 声明进出都是
  Replicate 主流(in_src/out_dst),latent 三件套的 norm 跟随区内 SP 布局
  (`enable_sp=enable_ep`)而不是主流。
- **tp2 x cp2 x pp2vp2 是守卫不是失败**:塔跨 PP 段时,shard 决策和动态 CP
  patch 计划在 `_encode_images` 内部做出,每个 share 需要完全一致地重算 ——
  尚未支持,`NotImplementedError` 按设计触发。单段塔 + CP 可用(右邻格即证)。

## 六、方法论沉淀

- anomaly mode 的正确读法:同一个反向节点被指认两次、而输入侧已修,结论是
  混型从 grad_output 方向来 —— 定位手段本身要服从"根因必须解释未炸的格"。
- 改已验证路径(clip 收编)必须带同 seed 回归格,逐位一致才算代数等价。
- `pkill -f` 与其他命令合写会匹配自身命令串自杀(exit 144,第二次踩中),
  已入长期记忆:按 PID 或方括号模式杀。
