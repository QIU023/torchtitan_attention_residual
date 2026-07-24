# Session handoff -- 8x5060Ti session, all changes + next-box plan (2026-07-24)

Authoritative pull-target for the 8x RTX 5060 Ti (16GB) session.
Supersedes SESSION_HANDOFF_2026-07-23 as the latest handoff; that doc
remains the context map for the 5090 session's work.

## 0. Repos to pull (all clean + pushed; logbook pins the submodules)

| repo | branch | HEAD | this session |
|---|---|---|---|
| logbook | main | (this commit) | phase13 docs/scripts below |
| torchtitan | `attention_residual_dev` | `a42be25f` (7 commits from `7d8acabe`) | CP guards, Ulysses CP, CP+TP, LoRA dtype, packed-MXFP4 QLoRA, 8h flavor, FSDP-gate grad-sync fix, tests |
| verl | `kimi_k3_integration` | `60b185fe` | engine CP working (5 fixes) |
| sglang | `attention_residual_inference` | `e45f675` | unchanged |

## 1. What changed (one line each; full detail in CP_TP_3D_VERIFICATION_2026-07-24.md)

**Three silent-correctness bugs found and killed in the landed CP:**
1. headtail load-balancer permuted the sequence under kimi CP
   (future-token leakage; step-1-parity gates cannot catch it).
2. CP+TP was split-brain (KDA applied CP, MLA skipped it).
3. FSDP was skipped at dp_shard=1+cp>1 -> cp replicas trained UNSYNCED
   (grads never reduced over cp; per-rank grad_norm was the symptom).

**Capabilities landed:**
- Real Ulysses CP (a2a seq<->head) for KDA+MLA; -19..22% peak mem vs
  the all-gather CP at seq 8k-32k; memory now scales with cp.
- CP x TP x PP 3D verified (tp2cp2pp2), plus CP x {FSDP, EP, AC, DCP,
  LoRA, QLoRA, Muon} and tp2cp4/tp4cp2 via the new debugmodel8h flavor.
- QLoRA through torchtitan.train: meta-built packed-MXFP4 layout +
  offline streaming quantize (stream_quantize_mxfp4_dcp.py); packed
  bytes bit-exact vs on-device quantization. TP x packed-base not wired.
- verl torchtitan engine CP works (kimi): SFT dp1cp2 full epoch,
  step-1 parity 6e-3 vs single GPU, end-loss matches baseline.
- gated-MLA x TP fixed (attn_gate_proj TP plan entry).

## 2. Environment recipe (this box; reproduce on the next)

- `/venv/main`: torch 2.12.0+cu130 (preinstalled), `uv pip install
  fla-core==0.5.1 torchao==0.17.0 "transformers>=5" datasets tiktoken
  blobfile safetensors pandas pyarrow pytest` + `-r torchtitan/requirements.txt`.
- `/venv/verl`: python -m venv; torch 2.12.0 cu130; `uv pip install -e
  ./verl --no-build-isolation` + fla-core + torchao + `-r
  torchtitan/requirements.txt` + `uvicorn fastapi ray hydra-core
  tensordict codetiming pylatexenc torchdata peft datasets tiktoken
  blobfile`. Run with `PYTHONPATH=<fork>/torchtitan`.
- veRL fixture: `python phase13_k3like_48b_posttrain/make_fake_hf_fixture.py
  --out /workspace/fake_hf` (downloads official 48B aux files from HF,
  exports random-init 194m via to_hf; also writes sft_tiny.parquet).
- sglang venv: not rebuilt this session (no rollout work);
  install_sglang_isolated.sh unchanged.

## 3. Verification snapshot (final HEAD, deterministic seed 42)

- unit suite: 84 passed + 66 subtests (includes new test_cp_qlora_fixes).
- matrix (5-step cells, rank-identical grad_norm everywhere): cp4
  7.573->6.073; tp2cp2 7.639->6.130; 3D tp2cp2pp2 7.626->6.136;
  fsdp2tp2cp2 7.649->6.019; LoRA x 3D trains; tp2cp4(8h) 7.684->4.707;
  fsdp8 regression bit-identical pre/post fixes (7.58611->5.80952).
- QLoRA(mxfp4-packed) fsdp2 + fsdp2cp2: 7.5695 step-1, trains.
- Muon x FSDP2 x TP2 x CP2 capstone: PASS.
- verl SFT dp1cp2: 12.148 -> 3.813 (1 epoch, tiny fixture).

## 4. Next-box work plan (in priority order)

**On 2xH200 / 8x5090-class (48B feasible):**
1. **48B QLoRA SFT via the packed-MXFP4 path** (the new capability):
   download official 48B -> hf-to-dcp -> stream_quantize_mxfp4_dcp.py
   -> `kimi_linear_48b_...` QLoRA flavor (add the 48B twin of
   debugmodel_gated_qlora_mxfp4) -> FSDP8(+CP) SFT on GSM8K. All infra
   verified at debug scale; this is execution + one config.
2. bf16-base 48B LoRA SFT re-run (recipe in run_48b_lora_sft.sh) to
   longer horizons -- unchanged from 07-23 plan.
3. **Real long-context CP run** (32k-128k+): CP x PP both available
   now; needs long-context data path. Also do the verl sharded-loss
   variant there (avoid gathering [B,T,V] logits; engine TODO noted in
   verl 60b185fe).

**On 8xH200 (full-param):**
4. Full-param 48B SFT + MXFP4-QAT (fp32 masters ~576GB) -- H200_HANDOFF
   Exp A; Muon variant now has the FSDPxTPxCP capstone behind it.

**On >=16 ranks:**
5. Full 5D (FSDPxTPxCPxEPxPP all >1) single run with debugmodel8h
   (dp2*tp2*cp2*pp2 = 16). Every <=8-rank projection is already green
   (Part 4 of the verification doc): all CP 2/3-axis combos, the two
   4-axis-with-EP-folded combos, HSDP x CP, cp8, compile x CP,
   validation x CP, and Interleaved1F1B (vp2) both alone and x CP
   (requires the phase3 recipe's pipeline_parallel_layers_per_stage
   flags; without them the default splitter emits non-contiguous P2P
   buffers -- config papercut, documented in the verification doc's
   CORRECTION note).

**Date-gated (7.27 official release):**
6. Reconciliation checklist (K3_RELEASE_IMPACT sec 4) + weight-sync
   tensor-name freeze + GRPO at scale on official vLLM.

**Anywhere / small:**
7. TP x packed-MXFP4 base (redirect Colwise/Rowwise to qdata/scale
   split storage) -- only when 48B QLoRA needs TP.
8. Upstream PR extractions still pending: moe.py TP+EP scatter fix
   (torchtitan), checkpoint interval=1 + the engine CP fixes (verl).

## 5. Honesty carries (unchanged, from CLAUDE.md + 07-23 handoff)

- Never claim 2.8T personally validated; 48B real weights + K3-faithful
  topology is the claim ceiling.
- Debug-scale cells verify MECHANISMS, not training quality; no
  under-trained result is presented as competitive.
- New this session: "loss descends within a band" is not a multi-rank
  correctness gate -- require rank-identical grad_norm in new-axis cells.
