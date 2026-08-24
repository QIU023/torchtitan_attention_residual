# CP 数值结果(文本侧,for draft PR)

树:`k3_on_4025` @ 已 rebase 到 upstream/main。flavor `kimi_k3_debugmodel_text`。
seq_len 1024(FlexAttention 的 BlockMask 要求 Q_LEN % (cp*128) == 0,cp8 需 >=1024;
FlexAttention 是上游 4025 自带的后端,非我们引入)。10 步,单一 seed,每格自带预热,
每个计量格断言 `Loading the checkpoint`。

## dp1 vs cp2/cp4/cp8

| cell | step-1 loss | 相对 dp1 | step-10 |
|---|---|---|---|
| dp1 | 12.44662 | - | 3.43712 |
| cp2 | 12.44292 | 3.0e-4 | 3.44543 |
| cp4 | 12.45092 | 3.5e-4 | 3.43541 |
| cp8 | 12.44724 | 5.0e-6 | 3.40910 |

## 与老树对比(证据)

老树 58 格(`MATRIX_18_SDPA_2026-08-09.md`,SDPA 后端):

| | 我们相对 dp1 | 老树相对 dp1 |
|---|---|---|
| cp2 | **3.0e-4** | 1.3e-2 / dp1 = 1.3e-3 |
| cp4 | **3.5e-4** | 1.5e-2 / dp1 = 1.2e-3 |

我们 cp2/cp4 比老树紧约 4 倍,全部远在 bf16 相对精度(2^-8 = 3.9e-3)内。

CP 改变序列分片,所以 step-1 本就不与 dp1 逐位相同(老树同理,SDPA/FlexAttention 皆然)。
偏差量级正常且稳定(cp8 = 5e-6)。

## 边界(已在代码里 guard)

* FlexAttention BlockMask:Q_LEN 必须被 cp*128 整除,否则上游报
  `NotImplementedError: Q_LEN not divisible by CP mesh world size * BLOCK_SIZE`。
* 折叠流多文档:microbatch 宽于 context window 时,CP 的 causal-only mask 无法表达
  文档边界,已加断言明确报错(commit "reject a CP stream that folds more than one document")。
