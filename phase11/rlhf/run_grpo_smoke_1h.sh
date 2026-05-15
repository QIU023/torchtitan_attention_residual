#!/usr/bin/env bash
# 1-hour GRPO smoke on the 447M Kimi AttnRes VLM SFT step-3100 ckpt.
# Goal: verify reward can LEARN now that the flashinfer_mla bf16-NaN
# inference blocker is fixed (decode_attention_backend=torch_native +
# ATTNRES_MLA_FP32_FALLBACK=1). v16 GRPO collapsed to reward=-1.0 because
# the rollout generator emitted all-`!` garbage; that root cause is now
# resolved (see phase11/VISION_INJECTION_BUG_RCA.md).
#
# timeout 3900 = ~65 min wall clock; --num-steps 500 will not be reached
# (torch_native decode has no CUDA graph, so it is slow on purpose) — the
# run is killed by the timeout and we read the reward trajectory.
set -euo pipefail
cd /workspace/torchtitan_attention_residual

export PYTHONPATH="${PWD}/torchtitan:${PWD}"
export ATTNRES_MLA_FP32_FALLBACK=1
# Bypass SGLang's POSIX-SHM bridge for multimodal payloads (UPSTREAM_PR_LIST
# #1). The SHM bridge races against monarch's actor lifecycle (producer
# unlinks /psm_xxx via resource_tracker before the scheduler's
# ShmPointerMMData.__setstate__ can attach) → SharedMemory(name=...) crash.
# Our sglang fork's tokenizer_manager._determine_tensor_transport_mode
# honors this env to fall back to inline pickle, lifecycle-safe.
export SGLANG_DISABLE_SHM_MM=1

exec timeout 3900 /usr/bin/python3 phase11/rlhf/run_grpo_llava_kimi.py \
    --dcp-load-path "${PWD}/phase5/runs/mm_sft_447m_full/checkpoint/step-3100" \
    --hf-model-path "${PWD}/phase5/runs/mm_sft_447m_full/hf_step3100" \
    --num-steps 500 \
    --num-episodes-per-step 4
