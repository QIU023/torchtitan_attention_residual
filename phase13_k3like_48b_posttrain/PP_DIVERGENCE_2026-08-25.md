# PP 与 dp1 的逐步分叉:一个被证伪的假说,和一个真结论(2026-08-25)

起因:用户要求 evidence matrix 报 step 1 / 3 / 10 而不只是 step 1。加上 step 3 和
step 10 之后,`k3_pp_text` 的 pp2 与 dp1 的差立刻显形 —— step-1 逐位相同,
step-10 差 6.5%。之前所有 PP 结果都只报 step-1,这个模式一直在,从没进过任何一张表。

## 一、假说:上游 4135(bf16 grad-norm 的分区依赖)—— 证伪

机制叙述当时对得上每一个观察,而且前提可查证:

* `TrainingConfig.dtype` 默认 `float32`,但 **4025 自己**给 `kimi_k3_debugmodel`
  写了 `dtype="bfloat16"`(`upstream/main:config_registry.py:88`),docstring:
  "put all parameters, **gradients**, and optimizer states in bfloat16"。
  全树 9 个核心模型只有 K3 这么设。
* `max_norm` 默认 1.0,实测 grad_norm 13~20,**每步都在裁剪**,
  裁剪因子 `1.0/total_norm`,所以范数误差 = 每步的有效学习率误差。
* 上游 CI 用 `loss_compare.py` 拿每个 config 对**它自己**的 golden 比,
  从不做 config 互比,所以分区依赖的偏移被烘进 golden,永不显形。

打上 4135 之后:

| | 4135 前 | 4135 后 |
|---|---|---|
| dp1 s3 / s10 | 7.56594 / 3.59412 | 7.56975 / 3.60945 |
| pp2 s3 / s10 | 7.63142 / 3.36165 | 7.62965 / 3.32162 |
| **跨格差 s3** | **6.55e-2** | **5.99e-2** |
| **跨格差 s10** | **-2.32e-1** | **-2.88e-1** |

两格各自都动了(补丁确实在执行),但 s3 差只收窄 8.5%,s10 差反而扩大。
**bf16 grad-norm 不是 PP↔dp1 分叉的驱动因素。**

教训与 `root-cause-must-explain-non-failures` 同类:一个能解释全部已观察现象、
前提也成立的机制,仍然可以不是原因。判据只能是实验。

## 二、4135 真正修好的东西:PP 数值不再依赖怎么切

它修的是另一件事,而且修得很干净:

| s10 | 4135 前 | 4135 后 |
|---|---|---|
| pp2(2 段) | 3.36165 | **3.32162** |
| pp4(4 段) | 3.36148 | **3.32162** |
| pp2 x vp2 | 3.36262 | **3.32162** |
| pp2 x vp4 | - | **3.32162** |

四种切法在 step 1 / 3 / 10 三个点全部逐位相同。这正是 4135 commit message
写的 "the value depends on **where the model was cut**"。

对那条 review blocked 的 PR,这是比原证据强得多的一句:原证据是给 llama3 硬加
`--training.dtype bfloat16`(非默认)造出来的,reviewer 自然会问"谁这么跑";
现在树里有一个**默认就这么跑**的模型,而且能给出"分区改变了数值 / 打上后不再改变"
的直接对照。

副产品:4135 把 total_norm 换成 fp32 之后,这个问题**第一次可测**。
打之前 dp1/pp2/pp4 的 step-1 grad_norm 显示全是 `20.5000`(bf16 在 20 附近间距
0.125),打之后是 dp1 `20.4873` vs pp2/pp4 `20.4876`。

## 三、分叉的真实位置:反向

step-1 grad_norm 在 optimizer step 之前算,dp1 与 pp2 差 3e-4 绝对 /
**1.5e-5 相对**。所以差异源头是**反向**,不是裁剪也不是优化器。

排掉的候选(读代码,未耗 GPU):

* **数据量不同** —— 不是。`trainer.py:409-427`,PP 开启时 `num_tokens_per_dp_rank`
  已乘进 `num_pp_microbatches`,dp1 = 8 次累积 x 1 微批,pp2 = 1 次累积 x 8 微批,
  总微批数都是 8。
* **loss 归一化不同** —— 不是。`trainer.py:855-874`,`global_valid_tokens` 把全部
  微批组的有效 token 先数完再算一次,两条路径共用同一个分母。这也是 step-1 loss
  能逐位相同的原因。
* **微批反向顺序不同** —— 不是驱动因素。1F1B 与 Interleaved1F1B 每个 stage 的
  backward 都按微批序 0..7 完成,与 dp1 的累积顺序相同。
* **CUDA graph** —— 不是。三个 flavor 都写死 `disable_cuda_graphs=True`。

剩下的头号嫌疑是 AttnRes 跨 stage 载体本身:所有 PP 格都过它,dp1 不过。
载体里 block 是 detach 存、读时重新挂 autograd,比无 PP 多了实在的算术,在 bf16
下改变舍入 —— 且这件事对所有切法是同一件事,解释了为什么各 PP 格彼此逐位相同
却共同偏离 dp1。

## 四、按老树标准判:过

老树 58 格相对 dp1 的 **step-2 相对差**(离共同起点一个 optimizer step,未被曲线放大):

| cell | s2 相对差 |
|---|---|
| text/pp2 | 7.19e-4 |
| text/pp8 | 3.84e-4 |
| text/tp2 | 2.53e-3 |
| text/fsdp2 | 1.09e-3 |
| mm_full/fsdp2_pp2_cp2 | 3.13e-3 |
| mm_full/tp2_pp2_cp2 | 3.17e-3 |
| **37 格全范围** | **1.2e-4 ~ 3.2e-3** |

**我们 pp2 vs dp1 的 step-2 相对差是 1.63e-3**,落在这个范围正中间。

step-10 看起来差很多是放大倍数不同:

| | s2 相对 | s10 相对 | 放大 |
|---|---|---|---|
| 我们 text_32l pp2 | 1.63e-3 | 7.97e-2 | 49x |
| 老树 text pp2 | 7.19e-4 | 1.24e-2 | 17x |
| 老树 text pp8 | 3.84e-4 | 1.08e-2 | 28x |
| 老树 text tp2 | 2.53e-3 | 1.24e-2 | 4.9x |
| 老树 mm_full fsdp2_pp2_cp2 | 3.13e-3 | 1.57e-2 | 5.0x |

我们的 flavor 10 步降 71%(12.48 -> 3.59),老树 text 降 37%,mm_full 降 18%。
曲线越陡,同样的初始扰动张得越开。

**还欠的对照**:`text_32l` 是 32 层 + 大词表(step-1 loss 12.48),老树 text arm
的 step-1 loss 是 7.70,说明词表小得多,不是同一个模型。已备好
`flat_control.sh`,在我们自己更平的 24 层 `kimi_k3_debugmodel_text` 上跑
dp1 / pp2 / pp4,让 step-2 相对差与老树 text/pp2 的 7.19e-4 直接可比。

## 五、待定

4135 目前 cherry-pick 在 `k3_pp_text`(`7a0a3ca01`)与 `k3_cp_text`
(`6a9221100`)本地,**未推**。证据矩阵是带着它跑的。要不要让三个 PR 依赖一个
review blocked 的上游 PR,未定。
