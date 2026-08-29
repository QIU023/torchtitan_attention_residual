# veRL×torchtitan Stage 0:GRPO 冒烟通过(2026-08-29,4×5060Ti)

目标(用户定):不碰 48B;从新树推进 veRL 后训练,smoke 级;titan 后端开
CP+FSDP+EP;不碰 titan 内部 RL(有 owner)。三段计划之 Stage 0 = 上游
e2e 原样跑通(Qwen3-0.6B + GSM8K),验证工具链。

## 一、结论

**rc=0,GRPO 四步端到端**:vLLM rollout → titan engine(FSDP4)log-prob →
GRPO adv → update_actor → 权重同步,每步全链。质量信号:
`rollout_actor_probs_pearson_corr=0.9991`、KL≈8e-4(引擎数值与生成侧吻合,
权重同步正确);稳态 27s/步、~500 tok/s、峰值显存 9.2GB/16GB。
log:`/workspace/verl_stage0.log`;脚本:scratchpad `verl_stage0.sh`。

## 二、可复现环境(/workspace/venv_verl,主 venv 零污染)

doc 的 6 月 known-good 四件套已过保质期(nightly 会下架)。实测可用组合:

```
vllm        1.0.0.dev20260705+cu130   # 0704 只有 aarch64 轮子
torch       2.14.0.dev20260705+cu130  # 与 vllm 同日构建,0-day ABI 差
torchtitan  0.1.0.dev20260701+cu130
torchvision 0.29.0.dev20260705+cu130  # --no-deps
verl        main @ea53291 可编辑安装 + TransferQueue(缺装依赖)
```

坑:uv 需 `--index-strategy unsafe-best-match`(outlines-core 被
first-index-wins 卡);vLLM 需 `rollout.max_model_len=1024`(默认吃模型
40k 上下文撞 max_num_batched_tokens)与 `gpu_memory_utilization=0.35`
(16GB 卡与 trainer 同卡,0.8 在 wake_up 时 OOM)。

## 三、CP 专项判定(verl 侧从未通过,三连阻断;titan 自身 CP 不背锅)

verl 的 e2e 脚本有 FSDP/TP/EP 旋钮、唯独没有 CP——首次实跑即证该路未完工:

1. `prepare_model_inputs` 把 positions 放 extra_inputs,而 titan 的
   `prepare_context_parallel_input` 要求 `extra_kwargs["positions"]`
   (KeyError;本地已补:交接进去、切片后取回);
2. headtail load balancer 要求 `seq % (2·cp) == 0`,与 NO_PADDING 的
   packed 变长流冲突(本地已补:传 None 连续切分);
3. flex-CP 的 block-mask 要求 Q_LEN 整除 `cp·2·128`,且 titan 明拒
   varlen×CP——硬墙。正解 = packed 流 pad 到该倍数 + logits 侧 unpad,
   属真实工程。

三条 + 设计草案 = 现成的 verl issue/PR 素材(K3 线在 verl 的第一块阵地)。

## 四、后续

- Stage 1:K3 SFT smoke(FSDP+EP;`_HF_MODEL_TYPE_TO_TORCHTITAN_NAME`
  加 K3 行,titan 安装指向我们的树;EP 用 K3 debugmodel 自身 MoE);
- Stage 2:GRPO 全环(K3 `sd_adapter.to_hf` 升格 + debugmodel 形状 HF
  config,vLLM 有 K3 day-0 类);
- 多节点缺口照旧(Elfie 类跨节点问题本地不可复现)。
