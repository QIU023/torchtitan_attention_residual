# seq-KD on 2×H200 — Setup & Runbook (zero-spin host bring-up)

**Purpose:** bring a fresh host to a working seq-KD / eval / OPD / GRPO state WITHOUT
re-discovering the environment traps that cost hours on 2026-05-30. Read this first.

Origin host: 2×H200 144GB, **sm_90 (Hopper)**, driver 580.126.20, CUDA 13, spot VM.
Everything persistent lives under **`/home`** (500G rbd). `/workspace` is ephemeral — never use it.

---

## 0. TL;DR — what's already done (state as of this commit)

- seq-KD pipeline works end-to-end on H200. The **only** real blockers were environment,
  all now fixed + documented below.
- 30k-subset seq-KD: DONE, eval val_loss base(sft_5200)=1.1797 → seq-KD=~1.06 (Δ≈−0.12).
- **Full 665k teacher distillation: RUNNING** (Qwen3-VL-30B-AWQ, DP=2). Resume-by-position.
- sentinel-collision training crash: ROOT-CAUSED + FIXED (see §5).
- sglang built + import-verified for OPD/GRPO (see §4).
- Code pushed: main `torchtitan_attention_residual@main`, fork `torchtitan@attention_residual_dev`,
  `sglang@attention_residual_inference`. All remotes are **ssh** (key id_ed25519, must be on GitHub).

---

## 1. Three Python environments (DO NOT MIX — torch version conflicts)

| Env | Python | Purpose | Key pkgs |
|---|---|---|---|
| conda `py3.10` | `/root/miniconda3/envs/py3.10/bin/python` | **training** (torchtitan + fla) | torch 2.11.0+cu130, transformers 5.7.0, fla 0.5.0, sentencepiece 0.2.1, torchao 0.17.0, datasets, pyarrow |
| `/home/venv/vllm` | `/home/venv/vllm/bin/python` | **teacher generation** (vLLM) | vllm 0.22.0, torch 2.11.0+cu130 |
| `/home/venv/sglang` | `/home/venv/sglang/bin/python` | **OPD/GRPO rollout** (sglang) | sglang 0.0.0.dev12321 (editable fork), sgl-kernel 0.4.2, flashinfer 0.6.8.post1, torch 2.11 |

- **torchtitan is NOT pip-installed.** It runs via `PYTHONPATH=/home/torchtitan_attention_residual:/home/torchtitan_attention_residual/torchtitan`. Do not `pip install -e` it (an early agent did; uninstall it — it leaves a dep pin that can clobber torch).
- vLLM and sglang each need their own torch; keep them isolated from the conda training env.

### Env traps fixed this session (each cost real time — don't re-hit):
1. **sentencepiece missing** → SigLIP tokenizer `ImportError`. `pip install sentencepiece` into conda env.
2. **torchao missing** → eval (`..._fp8` configs / float8 linear) `ModuleNotFoundError`. `pip install torchao` into conda env.
3. **Python.h missing** → `torch.compile` inductor C++ build fails (`CppCompileError`). Always `export TORCHDYNAMO_DISABLE=1` for training/eval (eager is fine for 447M). Baked into `/home/seqkd_overnight/lib.sh`.
4. **torchrun not on PATH** at `/usr/local/bin/torchrun` (launch scripts hardcode it). `ln -sf /root/miniconda3/envs/py3.10/bin/torchrun /usr/local/bin/torchrun`.
5. **hf_assets tokenizer absent**: launch scripts want `torchtitan/assets/hf/Llama-3.1-8B/{tokenizer.json,tokenizer_config.json,config.json}`. Download: `HF_HOME=/home/.hf_home hf download NousResearch/Meta-Llama-3.1-8B tokenizer.json tokenizer_config.json config.json special_tokens_map.json --local-dir torchtitan/assets/hf/Llama-3.1-8B`.
6. **All launch scripts hardcode `/workspace`** → remap to `/home` (cd line + INSTRUCT_DIR + CACHE_DIR). Already patched in `phase5_vlm_multimodal_sft/launch_stage{1,2}.sh`, `eval_stage2_ckpt.sh`.

---

## 2. Data layout (all under /home/.hf_home)

- Teacher model: `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ` at `/home/.hf_home/hub/models--QuantTrio--Qwen3-VL-30B-A3B-Instruct-AWQ/` (~17G). `hf download` it.
- LLaVA mix665k: `/home/.hf_home/LLaVA-Instruct/llava_v1_5_mix665k.json` (665298 convs; 40688 text-only).
  Images under `images/{coco,gqa,ocr_vqa,textvqa,vg}`. Download via patched `/home/dl_llava.sh`
  (conda-python version of `phase5.../download_instruct_665k.sh`, DEST=/home).
  - **ocr_vqa is special**: the HF repo `howard-hou/OCR-VQA` ships images embedded in **parquet**,
    NOT as `images/*.jpg`. Use `phase5_vlm_multimodal_sft/fix_ocrvqa_from_parquet.py` with
    `HF_HOME=/home/.hf_home DEST=/home/.hf_home/LLaVA-Instruct` (conda python). Extracts ~207k jpgs.
- Eval benchmarks: `/home/.hf_home/eval_data/` (POPE/TextVQA/ScienceQA/MMBench downloaded via
  patched `/home/dl_evaldata.sh`). GQA images reuse `LLaVA-Instruct/images/gqa`.
- Student init ckpt (seq-KD start): `phase5_vlm_multimodal_sft/runs/sft_5200_base/checkpoint/step-5200`
  (17G DCP, 8 shards, has projector+LM+train_state). This is the user-provided SFT-5200.

---

## 3. Pipeline stages + exact commands

Orchestration scripts live in `/home/seqkd_overnight/`. `lib.sh` exports HF_HOME + TORCHDYNAMO_DISABLE.

### S1. Teacher distillation (gen) — DP=2, ~5-9h for full 665k
```bash
cd /home/torchtitan_attention_residual/phase11_rlhf_grpo_infra/seq_kd
MODE=full NUM_SHARDS=2 GPUS=0,1 SUBSET=0 GPUMEM=0.85 CHUNK=512 MAXSEQS=0 \
  bash /home/seqkd_overnight/01_teacher_gen.sh
```
- Output `out_full/shard{0,1}.jsonl` → merged+image-filtered → `distilled_mix665k_full.json`.
- **Resume-by-position**: re-run and it skips already-written lines. Safe across spot preemption.
- Throughput ~10-15 conv/s/GPU (varies with answer length; teacher answers p50=106 tok).
- TUNING NOTE: raising chunk/max_num_seqs was a NEGATIVE optimization here (10→8.5 conv/s). Leave defaults.

### S2. seq-KD student SFT — FSDP=2, fast (~1-2h)
```bash
DISTILLED=.../distilled_mix665k_full.json \
INIT_CKPT=.../runs/sft_5200_base/checkpoint/step-5200 \
OUT_DIR=.../runs/seqkd_sft_447m \
NGPU=2 GLOBAL_BS=128 LOCAL_BS=16 SEQ_LEN=1024 TEXT_LEN=828 LR=2e-5 \
STEPS=<≈rows/128> SAVE_FREQ=50 KEEP_K=3 MAX_ATTEMPTS=6 \
  bash /home/seqkd_overnight/02_seqkd_sft.sh
```
- **LOCAL_BS=16 ≈ 60% mem (safe). LOCAL_BS=32 OOMs** (VLM activations are non-linear, not 3.6GB/sample).
- Use `/home/seqkd_overnight/supervisor.sh` for self-relaunching autoresume (survives crashes, grinds to STEPS).
- **Cannot resume a 2-GPU ckpt on 1 GPU** — torchtitan dataloader doesn't reshard across dp degree
  (`AssertionError: dp_degree is inconsistent`). Resume with the SAME NGPU it was saved at.

### S3. Eval — held-out val loss (quick) + real VQA accuracy (proper)
```bash
# quick val loss (DCP, NGPU=2): config MUST be non-fp8 to match training
bash /home/seqkd_overnight/03_eval.sh <ckpt_dir> <tag>   # STUDENT_CONFIG=...n4 (not n4_fp8)
# real VQA accuracy (the meaningful metric, NOT loss):
STAGE2_CKPT=<ckpt> NGPU=2 bash phase5_vlm_multimodal_sft/eval_benchmarks/run_all_evals.sh
#   BENCHES="mmmu scienceqa mmbench pope gqa"; consumes DCP; remap /workspace->/home + eval_data path.
```

### S4. OPD (on-policy distillation — NOT RL, it's distillation)
- Teacher = `TIGER-Lab/Mantis-8B-siglip-llama3` (loaded via HF AutoModelForImageTextToText, not vLLM).
- Entry: `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py --task opd` (uses sglang rollout).
- **HISTORY: token-JSD OPD was REFUTED across D2-D7** (18× capacity gap; best student ≈ SFT baseline).
  seq-KD is the distillation that actually worked. Run OPD only as a documented re-confirm.

### S5. GRPO (the only real RL)
- Entry: `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py --task gqa` (verifiable exact-match reward).
- Needs sglang (ready, §4) + hardcodes 8-GPU mesh (`total_gpus=8` lines ~286/495) → edit for 2 GPUs.
- weight-sync = disk; writes NO ckpt by design (wire a final save if you want RL'd weights).

---

## 4. sglang (for OPD/GRPO) — BUILT, import-verified

venv `/home/venv/sglang`. Editable install of on-disk fork `sglang/python` (commit f0589b077).
Verify: `PYTHONPATH=/home/torchtitan_attention_residual:/home/torchtitan_attention_residual/torchtitan /home/venv/sglang/bin/python -c "import sglang; from sglang.srt.models import attn_res_vl_overlay; from sglang.srt.configs.kimi_attn_res_vl import KimiAttnResVLConfig; from sglang.srt.entrypoints.engine import Engine; import sgl_kernel; print('OK')"`

If rebuilding from scratch on a new host:
```bash
/root/miniconda3/envs/py3.10/bin/python -m venv /home/venv/sglang
cd /home/torchtitan_attention_residual/sglang/python
/home/venv/sglang/bin/pip install -e . --no-deps --no-build-isolation   # editable, NO rust stall (key: these flags)
# then iteratively pip install the missing pure-python deps (orjson psutil aiohttp ... ~135 of them)
/home/venv/sglang/bin/pip install sgl-kernel==0.4.2 flashinfer-python==0.6.8.post1   # 0.3.21 is ABI-broken vs torch2.11
apt-get install -y libnuma1                                                          # sgl_kernel .so needs libnuma.so.1
# the fork's attn_res_overlay.py f-strings are already py3.10-fixed (commit f0589b077)
```
- sgl_kernel/flashinfer import on CPU; CUDA kernels JIT-compile at first GPU run (deferred, sm_90).
- rust gRPC ext NOT built (gRPC-serving only, not needed for OPD/GRPO import path).

---

## 5. The sentinel-collision training crash (ROOT-CAUSED + FIXED)

Symptom: multimodal SFT on distilled data hits `CUDA device-side assert` ~every 25 steps
(surfaces async as CUBLAS_FAILED at MoE/KDA — RED HERRING, not MoE).
Root cause: `IMAGE_TOKEN_ID=32000` decodes to the subword **'utility'** in Llama-3.1 (it is NOT a
reserved id). Teacher-distilled answers use "utility"-ish words → ~0.03% of rows tokenize a TEXT
token to 32000 → that row has >196 sentinel positions but only 196 vision embeds → `masked_scatter`
`totalElements <= srcSize` assert.
FIX (committed, fork `torchtitan@attention_residual_dev` 55abb36): clamp scatter destinations per row
to `n_vis_max` in `experiments/kimi_linear/attn_res_model.py`. Bit-identical for well-formed rows.
Deeper fix (deferred): pick a genuinely-reserved sentinel id in the SFT dataset.

---

## 6. Operational rules (spot VM — no state loss)

- **Commit + push after every meaningful artifact.** Spot can preempt anytime. All 3 remotes are ssh.
- Disk watchdog `/home/seqkd_overnight/watchdog.sh` kills all jobs if /home <10G (ckpts are 17G each; KEEP_K small).
- All long jobs run `setsid`/detached + resume-by-position or autoresume, so preemption loses ≤ last interval.
- Recovery state notes: `/home/seqkd_overnight/STATE.md` + `NOW.txt` (human log of what's running + lessons).
- `pkill -f train_mm` MATCHES YOUR OWN SHELL → self-kill. Always use bracket trick `pkill -f '[t]rain_mm'`.
- Verify subagent "SUCCESS" claims independently (two sglang agents falsely reported success before the real build).

---

## 7. STANDARD seq-KD EVAL SPEC (full-N triangle — supersedes any 500-subset smoke)

**Triangle:** student-baseline (pre-seq-KD = SFT-5200) vs student-post-seq-KD vs **teacher (Qwen3-VL-30B-A3B)**.
ALWAYS run full N (`*_LIMIT=0`); the 500-subset numbers from early smokes are NOT standard.

Pipeline (already in repo, eats DCP directly, no HF convert):
`phase5_vlm_multimodal_sft/eval_benchmarks/run_all_evals.sh`  BENCHES="mmmu scienceqa mmbench pope gqa"  *_LIMIT=0
Input: `STAGE2_CKPT=<DCP ckpt>`  (config must be non-fp8 `...n4`).  NGPU=2.

### Core 3 (have baseline anchors — MUST run):
| Benchmark | full N | metric | student baseline | note |
|---|---|---|---|---|
| GQA test-dev-balanced | 12,578 | exact-match acc | **12.3** | data re-downloaded (lmms-lab/GQA parquet) |
| MMBench-EN-dev | 4,377 | 4-way MC acc | **36.4** | data |
| POPE (random+popular+adversarial) | 8,910 (~3K×3) | F1 / acc | **50 (always-no → F1=0)** | check yes_ratio for harness bug |

### Breadth (repo supports):
| Benchmark | full N | metric |
|---|---|---|
| ScienceQA-IMG (test) | 2,017 | acc |
| MMMU (val) | 900 | acc |

### Teacher eval (triangle upper bound) — CRITICAL apples-to-apples:
Feed Qwen3-VL-30B through the SAME pipeline but with the SAME `max_pixels=1003520` (~1280 tok) used during
distillation generation. Full-resolution would inflate teacher scores → not comparable.

### eval datasets on /home/.hf_home/eval_data (scorers read these exact paths):
- gqa/ (testdev_balanced parquet)  — `lmms-lab/GQA` — re-downloaded
- mmbench/en/ — `lmms-lab/MMBench` — re-downloaded
- mmmu/data/ (validation parquet) — needs `MMMU/MMMU` val
- pope/ (+ COCO val2014) ✓ present
- scienceqa/ ✓ present ; textvqa_val/ ✓ present
Download helpers (conda-python, DEST=/home): /home/dl_evaldata.sh (priority_a), /home/dl_evaldata_b.sh (GQA+MMBench).

## 8. seq-KD training data
mix665k full = **665,298 convs** (624,610 with-image + 40,688 text-only); teacher rewrites ALL assistant turns.
(2026-05-30 full distillation run does exactly this — vs the earlier 30k smoke subset.)
