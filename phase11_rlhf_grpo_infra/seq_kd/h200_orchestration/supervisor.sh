#!/usr/bin/env bash
# Self-relaunching supervisor for seq-KD: the inner autoresume wrapper exits after
# MAX_ATTEMPTS; this keeps re-invoking it (resume from latest ckpt) until step-600
# is reached or a hard deadline. Survives the data-driven MoE device-side assert.
set -uo pipefail
CK=/home/torchtitan_attention_residual/phase5_vlm_multimodal_sft/runs/seqkd_sft_447m/checkpoint
LOG=/home/seqkd_overnight/logs/supervisor.log
DEADLINE_TS=$(( $(date +%s) + 6*3600 ))   # 6h cap
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
log "supervisor start"

latest_step(){ ls -d "$CK"/step-* 2>/dev/null | sed 's/.*step-//' | sort -n | tail -1; }

while true; do
    s=$(latest_step); s=${s:-0}
    if (( s >= 600 )); then log "REACHED step-$s — done"; break; fi
    if (( $(date +%s) > DEADLINE_TS )); then log "DEADLINE reached at step-$s"; break; fi
    # nothing running -> (re)launch the autoresume wrapper
    if ! pgrep -f '[0]2_seqkd_sft' >/dev/null; then
        log "relaunch inner wrapper (latest=step-$s)"
        cd /home/torchtitan_attention_residual
        env DISTILLED=/home/torchtitan_attention_residual/phase11_rlhf_grpo_infra/seq_kd/distilled_mix665k_full.json \
            INIT_CKPT=/home/torchtitan_attention_residual/phase5_vlm_multimodal_sft/runs/sft_5200_base/checkpoint/step-5200 \
            OUT_DIR=/home/torchtitan_attention_residual/phase5_vlm_multimodal_sft/runs/seqkd_sft_447m \
            NGPU=2 GLOBAL_BS=128 LOCAL_BS=16 SEQ_LEN=1024 TEXT_LEN=828 LR=2e-5 \
            STEPS=600 STEPS_CAP=600 WARMUP_STEPS=30 SAVE_FREQ=25 KEEP_K=3 MAX_ATTEMPTS=30 LOG_FREQ=1 \
            bash /home/seqkd_overnight/02_seqkd_sft.sh >> /home/seqkd_overnight/logs/s2_super.log 2>&1
        log "inner wrapper exited (latest now step-$(latest_step))"
    fi
    sleep 60
done
echo SUPERVISOR_DONE > /home/seqkd_overnight/SUPERVISOR_DONE
log "supervisor exit"
