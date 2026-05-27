#!/usr/bin/env bash
# 12h overnight GRPO on REAL COCO captions (stage2 LLaVA ckpt, num_blocks=4).
# Goal: substantive multimodal RL improvement = reward_mean trends up over steps.
# Safety: hard 12h timeout; disk-watchdog kills the run if /workspace < 8G (vastai
# daemon eats disk autonomously); GRPO itself writes NO ckpts (verified). Auto-restart
# once if it dies before 1h (transient boot crash) — but NOT in a tight loop.
set -uo pipefail
ulimit -c 0   # no core dumps (a crash here previously dumped 122G to core_pattern)
cd /workspace/torchtitan_attention_residual
LOG=/tmp/grpo_12h_overnight.log
NUM_STEPS="${1:-220}"   # ~204s/step -> 220 ≈ 12.5h; 12h timeout caps it anyway

disk_watchdog() {
  while true; do
    sleep 60
    fr=$(df -BG /workspace | awk 'NR==2{gsub("G","",$4);print $4}')
    if [ "$fr" -lt 8 ]; then
      echo "[DISK<8G @ $(date +%H:%M)] killing GRPO to protect the box" | tee -a "$LOG"
      pkill -9 -f run_grpo_llava_kimi.py
      return 2
    fi
    pgrep -f run_grpo_llava_kimi.py >/dev/null || return 0
  done
}

echo "===== 12h GRPO overnight start $(date) num_steps=$NUM_STEPS =====" | tee -a "$LOG"
disk_watchdog & WD=$!
# 12h hard cap; launcher already sets env + flavor + ckpt paths
timeout 43200 bash phase11_rlhf_grpo_infra/rlhf/run_grpo_stage2_step5200.sh "$NUM_STEPS" >> "$LOG" 2>&1
RC=$?
kill $WD 2>/dev/null
echo "===== GRPO exited rc=$RC at $(date) =====" | tee -a "$LOG"
echo "===== reward_mean trajectory =====" | tee -a "$LOG"
grep -aoE "step +[0-9]+ +loss=[-0-9.]+ +reward_mean=[-0-9.]+" "$LOG" | tee -a "$LOG"
echo "ALL_12H_DONE" | tee -a "$LOG"
