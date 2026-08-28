# PR30: MinimalAsyncEP dispatch output must be owned under autograd

* 分支:fork `maep_dispatch_owned`(本地 worktree `/workspace/tt_maep_fix`,基点 upstream/main
  `b953a3f`,commit `74a89f8`),**未推**,推不推由用户定。
* 发现过程与全部证据:`phase13_k3like_48b_posttrain/EP_BACKEND_EVIDENCE_2026-08-28.md` §八.1;
  探针工具 `matrix_scripts/ep_backend_probe/`。
* body 数字来源:`<scratch>/run_maepfix_verify.sh`(探针,after)与 `<scratch>/run_dsv3_10step.sh`
  (10 步前后)。占位符在 body 里,跑完填。
* 零 K3 依赖:deepseek_v3_debugmodel 普通专家即可复现,body 里不提 K3 之外的树。

## 数字(2026-08-28)

* deepseek 普通专家探针:修前 experts 组 max rel 1.0(w1/w3 精确 0),修后 3.1e-4;fused 路径修前 w13 精确 0。
* deepseek 10 步:standard 8.00752 / 5.07747 / 3.95123;maep 修前 8.00751 / 5.07892 / 3.95201;修后 8.00751 / 5.07750 / 3.95003。
* K3(ep_review1 + 修复):见 `EP_BACKEND_EVIDENCE_2026-08-28.md` §八.0。
