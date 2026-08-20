# 一致性扫描

两个可复跑的脚本,存在这里的原因是:这一整轮 DEP 分析连着四次给出错误结论,每一次的根因都不是
测量做错了,而是**同一个量在树里有多个互相冲突的值,而没有任何东西会因此变红**。

    python3 consistency/xcheck_claims.py        # 同一量出现多个不同值 -> 报冲突
    python3 consistency/factcheck_table1.py     # 已知错误数字的残留

从仓库根目录跑。两个都只报告,不改文件。

## `xcheck_claims.py`:同一量的多值冲突

按"量名 -> 出现的所有值"分组,只打印有两个以上不同值的量。它抓到过的真问题:

* **`parallelize.py` 行数**在文档里有 1656 / 1768 / 1077 三种说法,真实是 1786。三个都是各自
  时点的快照,没人回头更新。已改成不写死数字 —— 行数快照注定腐烂,要就用 `wc -l` 现取。
* **ViT 参数 447M vs 401M**。两个都对,量的是不同东西:401M 是报告 Table 1 的编码器 + 位置嵌入,
  447.4M 含 46.1M 的 mm projector(`SCALE_AUDIT_2p8t_2026-08-04.md` 早已核对)。但树里三处写
  "447M tower" 而不带限定,任何拿报告对照的人都会认为我们的数字错了。已补上限定。
* **DEP 的 tps / placed / exhausted** 在 8-16、8-17 与 8-19 三份文档里数值不同 —— 那些是不同
  配置下的测量,不是矛盾,但当时没有一处说明配置差异。`DEP_MEASUREMENT_RETRACTION_2026-08-19.md`
  是为此写的。

## `factcheck_table1.py`:与报告 Table 1 对表

报告的权威数字(Table 1):

    #Layers 93        Total 2.78T        Activated 104.2B      Hidden 7,168
    Routed experts 896    Active/token 16    Shared 2     Attn heads 96
    Composition 69 KDA + 24 MLA    MoE hidden/expert 3,072    Latent MoE 3584
    Context 1M        ViT 401M / 27 layers / patch 14 / 12 heads

抽查过的都与树里一致。曾经唯一的差 —— activated 报告 104.2B 而我们算 105.8B —— **2026-08-20
查清了**:报告把 `embed_tokens` 与 `lm_head` 算作一份绑定权重(各 1.17B),我们算两份。
105.42 − 1.17 = 104.25,残差 0.05B。那份 audit 里同时存在 105.4 和 105.8 两个值,105.42 才是
能复现的那个。拆解见 `SCALE_AUDIT_2p8t_2026-08-04.md`。

## 一条使用纪律

**行数、参数量、tps、隐藏率这类会变的量,写进文档时必须带上"什么配置下测的"。** 不带配置的
数字在三个月后无法判真假,而它会被当成事实继续推理 —— 这正是 DEP 那条线上发生的事。

## 第三类:验证基线本身不可信

2026-08-20 抓到一个存在时间不明的 flaky:`test_moonvit_stage_split.py` 构造 `MoonViT` 后
从不调 `init_weights()`,而 `patch_embed.pos_emb.weight` 在那之前只是 `torch.empty`。读到的是
分配器给回的任意内存 —— 多数时候有限,偶尔含 NaN,表现为全量套件**约五次出一次**失败。

**这条比它修的 bug 本身更值得记**:在此之前,这个仓库的每一次"420 passed"都只是抽样没抽到。
本轮我反复用它当门槛,包括为 DEP、注释精简、上游合并适配背书。

定位过程里我错了两次,两次都是同一个毛病 —— **在"失败可复现"这个未经验证的前提上推理**:

* 二分定位到 `test_gated_mla.py`。但同一文件集同一顺序重复三次全过,二分链作废;
* TF32 假设(`parallelize.py` 里有 `set_float32_matmul_precision("high")`,而测试用
  `atol=1e-6`,看起来很合)。实测强制 `high` 跑 6 passed,而且这条路径在 CPU 上,TF32 只影响
  CUDA matmul。

真正定位它的办法是**在全量套件内插桩**:单独跑 40 次不复现,带前 14 个文件预热 20 次也不复现。
插桩一次 dump 就指名了参数。

### 纪律

* 概率性失败不能用二分定位,先确认可复现性再谈定位;
* 断言"这套测试是绿的"之前,至少连跑三轮 —— 单次绿只是一次抽样;
* `torch.empty` 建的参数必须有对应的 `init_weights` 覆盖,而测试必须调用它。这类"没初始化但
  大多数时候看不出来"的缺陷,只有在数值断言足够严格时才会暴露。
