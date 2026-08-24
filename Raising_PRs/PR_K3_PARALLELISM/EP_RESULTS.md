# EP 数值结果(文本侧,for draft PR)

树:`k3_on_4025` @ 已 rebase 到 upstream/main。flavor `kimi_k3_debugmodel_text`。
上游 `AllToAllTokenDispatcher`,`comm_backend="standard"`;**不含 MoonEP**(见下)。
10 步,单一 seed,每格自带预热,seed 断言通过。

EP 在 data 轴内分片专家(ep_degree 整除 dp),所以与同 world-size 的纯 dp 基线比,
不与 dp1 比。

| cell | step-1 | 基线 | 相对 |
|---|---|---|---|
| dp2 | 12.43537 | - | - |
| ep2_fsdp2 | 12.43537 | dp2 12.43537 | **逐位相同** |
| dp8 | 12.44491 | - | - |
| ep8_fsdp8 | 12.44615 | dp8 12.44491 | 1.24e-3 (相对 1e-4) |

ep2 与 dp2 step-1 逐位相同。ep8 与 dp8 差 1.24e-3,涉及 8 路专家分片 + 独立 edp mesh,
远在 bf16 相对精度(3.9e-3)内。

## 与老树对比

老树 58 格 ep2_fsdp2 step-1 与 dp2 五位全同(换 AllToAllTokenDispatcher 后)——
我们 ep2_fsdp2 逐位相同,一致。

## MoonEP:明确排除

MoonEP(report 的完美负载均衡 EP + 冗余专家在线规划/迁移)超出本 PR 范围:
它需要独立 dispatcher 和一整套专家规划后端,新树没有。dispatcher 的 `comm_backend`
pin 成 `"standard"`(非配置选项),确保没有 run 误以为在跑 MoonEP。代码注释已注明。
