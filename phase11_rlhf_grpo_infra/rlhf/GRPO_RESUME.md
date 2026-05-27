# GRPO resume — stage2 LLaVA ckpt (2026-05-27)

Picked the torchtitan multimodal GRPO back up after the LLaVA SFT ckpt was
scp'd back. This captures everything needed to run it.

## TL;DR launch
```bash
cd /workspace/torchtitan_attention_residual
bash phase11_rlhf_grpo_infra/rlhf/run_grpo_stage2_step5200.sh 500   # 500 = real run; 1 = smoke
```
The launcher (created this session) wires DCP + HF + flavor + env exports. All 6
boot blockers below are already fixed.

## Checkpoint (scp'd back, verified)
- **DCP** (trainer): `phase5_vlm_multimodal_sft/runs/stage2_instruct_sft_447m/checkpoint/step-5200`
  — 17G, 8 FSDP shards + .metadata, 1706 tensors incl `mm_state.projector.*`. Verified readable.
- **HF** (SGLang generator): `phase11_rlhf_grpo_infra/hf/stage2_447m_step5200`
  — converted from the DCP this session (3.0G safetensors, 1823 keys). Plus tokenizer +
  **preprocessor_config.json** (see fix #6).

## Env (/venv/main, no rebuild)
torch 2.11.0+cu130 · torchtitan/monarch editable · torchstore 0.1.2 · sglang 0.5.11 · transformers 5.6.0.
Required exports (in the launcher): `PYTHONPATH=$PWD/torchtitan:$PWD`,
`ATTNRES_MLA_FP32_FALLBACK=1`, `SGLANG_DISABLE_SHM_MM=1`,
`SGLANG_FP8_IGNORED_LAYERS=attn_res_proj,mlp_res_proj,final_attn_res_proj,mlp.experts`.
GPU layout: trainer FSDP=4 (ranks 0-3) · SGLang generator TP=4 (ranks 4-7) · grader CPU.

## 6 boot blockers fixed this session
1. **DCP→HF conversion** — HF ckpt was deleted; regenerated via `dcp_to_hf_kimi_attn_res_vl.py`
   `--config kimi_linear_447m_aligned_block_attn_res_n4 --vision-tower google/siglip-base-patch16-224
   --processor-source phase10_ckpt_dcp_to_hf/hf_step9700_paperalign_C`.
2. **cache_3cam_native style — n/a here.**
3. **flavor / num_blocks mismatch** (the big one): the ckpt was trained `_n4` (num_blocks=4, a
   config_registry override). GRPO's `model_registry` only resolves 447m_aligned block_attn_res→8,
   full_attn_res→16 — NO num_blocks=4. num_blocks is a forward-time grouping (doesn't change tensor
   shapes), so an 8-block skeleton would LOAD but group AttnRes wrong → bad logits/reward. **Fix:**
   patched `run_grpo_llava_kimi.py` so `_n4` flavors source the ModelSpec from `config_registry`
   (num_blocks=4, the exact spec the SFT + converter used). Launcher passes
   `--flavor kimi_linear_447m_aligned_block_attn_res_n4`.
4. **LLaVA-Pretrain-558K data deleted** — built an 8-record **stub** at the default path
   `/workspace/.hf_home/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json` (+ `stub_images/` symlinked to
   real nuScenes jpgs) so the stack boots. **For a real run, scp the true 558K JSON + images to that
   same path (overwrites the stub).** Needs disk cleanup first — only 15-16G free, full set is tens of GB.
5. **(env exports)** already in launcher (MLA fp32 fallback + SHM-MM disable were prior-session fixes).
6. **SGLang processor missing preprocessor_config.json** — converter copied only tokenizer.json +
   tokenizer_config.json, so `AutoProcessor.from_pretrained` returned no `.tokenizer` and the
   AttnRes-VL processor raised "no tokenizer found". **Fix:** copied SigLIP
   `preprocessor_config.json` into the HF dir → AutoProcessor now returns a SiglipProcessor with
   both `.tokenizer` and `.image_processor`. (TODO: have the converter copy this automatically.)

## Smoke status (2026-05-27 ~10:24) — ✅ FULL STACK VALIDATED
1-step smoke completed a full GRPO step end-to-end:
```
step 0  loss=-1.0087  reward_mean=-0.578  dt=204.1s
  [-] reward=-1.000   [+] reward=+0.238
  [-] reward=-1.000   [+] reward=+0.280
```
trainer FSDP=4 + SGLang generator TP=4 + grader + reward + GRPO update all worked with the scp'd
num_blocks=4 ckpt. Reward distribution is HEALTHY (mix of -1.0 length/format violations and positive
BLEU content scores) — NOT the old v16 all-(-1.0) collapse. Process exited cleanly, GPUs freed.
(The `TimeoutError` at the very end is a cosmetic monarch actor-teardown race AFTER step 0 finished.)
Benign warnings only: mistral-regex tokenizer, missing generation_config.json, cutlass.cute, SigLIP
text-tower UNEXPECTED keys (vision-only model). dt=204s/step is the torch_native decode (slow by design).

**NOTE: reward VALUES are meaningless here — they're vs the 8 fabricated stub captions. The PIPELINE is
proven. A converged run needs the real LLaVA-558K data (below).** A long run was deliberately NOT
launched on stub data (would burn GPU on garbage rewards).

## Known benign warnings (ignore)
- `incorrect regex pattern ... fix_mistral_regex=True` — tokenizer warning; the SFT used this exact
  tokenizer so it's consistent. (Could add `generation_config.json` to silence the other warning.)
- `cutlass.cute.experimental` import walk — SGLang/CUTE_DSL, harmless.

## Remaining for a REAL converged run
- Replace stub with the true LLaVA-Pretrain-558K data (scp to the default path; cleanup disk first).
- Bump `--num-steps` (launcher arg) from 1 to the target (e.g. 500); torch_native decode is slow by design.

## 12h overnight run (2026-05-27 10:34) — REAL DATA
- **Data:** real COCO val2017 image-caption (5000 pairs, downloaded ~800MB), built into
  LlavaCaptionTask json at the default path (`/workspace/.hf_home/LLaVA-Pretrain/...`, symlinked
  `val2017/`). The true LLaVA-558K images (24GB) were too big for the 15G disk; COCO is the
  disk-safe real-caption substitute (model is a general captioner, so COCO is in-domain enough
  for BLEU+format reward to give real RL signal).
- **Launcher:** `run_grpo_12h_overnight.sh 220` → wraps `run_grpo_stage2_step5200.sh` with a
  12h hard timeout + a disk-watchdog that kills if /workspace < 8G (vastai daemon eats disk).
  GRPO writes NO ckpts → disk-safe; ~204s/step → ~220 steps in 12h.
- **Improvement metric:** `reward_mean` per step (grep `step N loss=.. reward_mean=..`). Baseline
  step 0-1 ≈ -0.60/-0.65. Substantive RL improvement = reward_mean trending up over the run.
- **Reward values are real now** (vs the earlier stub): BLEU-1(completion, COCO gold) + length + format.

## On interfacing torchtitan PP adapter ↔ veRL (assessment, per request)
- **torchtitan GRPO (this run):** trainer = torchtitan FSDP + `pipeline_kimi_linear_with_cache_adapter`
  (the PP adapter — torchtitan pipeline schedule + AttnRes KV-cache adapter); rollout = SGLang via the
  engine-agnostic Generator monarch actor; weights synced torchstore-RDMA (or DCP→HF).
- **veRL GRPO (other track):** veRL's own FSDP/Megatron trainer + vLLM/SGLang rollout; data = the
  on-box `nusc_planning` parquet (7.5G, waypoint reward) for the Qwen2.5-VL VLA — a DIFFERENT model+task.
- **Feasible interface = shared ROLLOUT + WEIGHT BRIDGE, not a trainer merge.** Both can share the
  SGLang rollout backend (veRL already supports SGLang) and a DCP↔HF weight bridge. The torchtitan
  PP adapter is torchtitan-pipeline-specific and does NOT transplant into veRL (veRL would use
  Megatron-PP). So: keep torchtitan's PP adapter on the trainer side, expose the policy to veRL only
  via weight-sync + the engine-agnostic generator — NOT by running torchtitan PP stages as veRL actors.
- **Recommendation:** torchtitan env is stable here (this run trains), so NO switch to veRL needed.
  veRL stays the fallback for the nusc_planning track if the torchtitan stack regresses.

## 12h run v2 status @ 601 steps (2026-05-27 ~15:00) — HONEST findings
reward_mean windowed avgs (100-step windows): -0.327 / -0.311 / -0.329 / -0.316 / -0.336 /
-0.299 / (-0.288). **Essentially FLAT** (~-0.32 ± 0.03 noise) with only a faint late uptick;
loss stable ~O(0.3) (the reward-shaping + kl_coef 0.1 fixed the v1 instability + sparsity, but
the reward is not climbing substantially). 0 errors, disk 14G stable, dt ~20s/step.

Interpretation (honest): the GRPO infra + recipe are CORRECT and stable on real data — that is the
solid deliverable. But the **reward is near its ceiling** for this setup: a 447M policy + BLEU-1-vs-
COCO-human-caption reward is a low, noisy objective, and kl_coef=0.1 keeps the policy near the SFT
init. So per-step RL gains are marginal, not dramatic.

### TWO limitations the user should weigh in on (NOT auto-fixed to avoid overnight thrash):
1. **No weight persistence.** This GRPO scaffold writes NO checkpoint (disk-safe by design), so the
   RL-updated policy is LOST when the run exits — even a real improvement would be unusable. To make
   the run produce a usable RL'd ckpt, add a final DCP/HF save at the end of the GRPO loop in
   run_grpo_llava_kimi.py (447M ≈ 3-9GB; 14G free fits one save). REQUIRES a relaunch.
2. **Flat reward → levers for a bigger gain (each a relaunch):** (a) lower kl_coef 0.1→0.02-0.03
   (let the policy chase reward — now safe since the reward is dense, unlike v1); (b) stronger/cleaner
   reward (BLEU-1 is weak; reward length-correctness + a better content metric); (c) the data is COCO
   not the model's true LLaVA-558K distribution (closer data would give cleaner signal).

### Recommendation
The current run is the best-conditioned config and is left running (disk-safe, validates 12h stability;
restarting a 3rd time would lose the accumulated steps). Decide in the morning: if you want a USABLE
RL'd model, relaunch with (1) a final-save wired in + (2) kl_coef 0.02. The infra is proven; these are
recipe/scaffold choices, not bugs.
