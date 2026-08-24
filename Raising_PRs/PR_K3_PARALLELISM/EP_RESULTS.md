# EP 数值结果(文本 + 多模态,for draft PR)

树:`k3_on_4025` @ rebase 到 upstream/main。上游 `AllToAllTokenDispatcher`,
`comm_backend="standard"`,**不含 MoonEP**。2 步,单一 seed,每格自带同配置预热,
每个计量格断言 `Loading the checkpoint`(全部通过)。

EP 在 data 轴内分片专家(ep_degree 整除 dp),所以与**同 world-size 的纯 dp 基线**比。

## 文本(kimi_k3_debugmodel_text)

| cell | step-1 | 基线 | 相对基线 |
|---|---|---|---|
| dp2 | 12.42251 | - | - |
| ep2_fsdp2 | 12.42327 | dp2 | 7.6e-4 |
| dp4 | 12.40849 | - | - |
| ep4_fsdp4 | 12.40863 | dp4 | **1.4e-4** |
| dp8 | 12.41676 | - | - |
| ep8_fsdp8 | 12.41700 | dp8 | 2.4e-4 |

## 多模态(kimi_k3_debugmodel,含视觉塔)

| cell | step-1 | 基线 | 相对基线 |
|---|---|---|---|
| dp2 | 12.46440 | - | - |
| ep2_fsdp2 | 12.46579 | dp2 | 1.4e-3 |
| dp4 | 12.42986 | - | - |
| ep4_fsdp4 | 12.43112 | dp4 | 1.3e-3 |
| dp8 | 12.45809 | - | - |
| ep8_fsdp8 | 12.45933 | dp8 | 1.2e-3 |

六格全部远在 bf16 相对精度(2^-8 = 3.9e-3)内。多模态偏差略大于文本,与视觉塔参与一致。

## 与老树对比

老树 58 格 `ep2_fsdp2` step-1 与 dp2 五位全同(换 AllToAllTokenDispatcher 之后)。
我们 ep2 文本 7.6e-4 —— 同量级,且 ep4/ep8 一致地更紧或相当。

## MoonEP:明确排除

MoonEP(report 的完美负载均衡 EP + 冗余专家在线规划/迁移)超出本 PR 范围:
需要独立 dispatcher 和专家规划后端,本树没有。dispatcher 的 `comm_backend` pin 成
`"standard"`(非配置项),确保没有 run 误以为在跑 MoonEP。
