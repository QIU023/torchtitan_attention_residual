#!/bin/bash
# Full rerun of the stack verification on k3_mxqat_stack_20261012 = 7f8b1bf4d (user 10-14: "全部重跑"), 8 x 5060.
# CPU jobs in the background; GPU phases one after another, one owner. Fresh output dirs under r7f8b, no deletion.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; V=$S/verify_stack_1013; R=$V/r7f8b
K=$V/kit_more; T=$S/wt_mxqat_fix; LB=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM
note() { echo "$(date -u +%H:%M:%S) $*" >> $R/progress.txt; }
lane() {  # <out> <marker expression> <gpus> <port>
  ( . /workspace/venv_1006/bin/activate; cd $T && MASTER_PORT=$4 CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=$S/main_pins:$T:$S/elfie_ao \
    TORCHINDUCTOR_CACHE_DIR=$1.cache/ic TRITON_CACHE_DIR=$1.cache/tc timeout 5400 python -m pytest -q -p no:cacheprovider -rfE \
    --continue-on-collection-errors -m "$2" --strict-markers --durations=40 tests/unit_tests/gpu > $1 2>&1; echo "rc=$?" >> $1 )
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
( bash $V/cpu_suite2.sh $T $R/cpu_stack; note "CPU suite done" ) &
( . /workspace/venv_1006/bin/activate; cd $T && CUDA_VISIBLE_DEVICES= PYTHONPATH=$S/main_pins:$T:$S/elfie_ao timeout 1800 \
    python $K/k3_full_key_coverage.py /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/official_k3/reference \
    > $R/key_coverage.txt 2>&1; note "key coverage rc=$?" ) &
note "G1 mm18 base"; bash $K/run_cells.sh base $T $R/mm18_base $R/cells_mm18.txt
note "G2 mm18 mxqat"; bash $K/run_cells.sh mxqat $T $R/mm18_mxqat $R/cells_mm18.txt
note "G3 mxqat_muon"; mkdir -p $R/mm18_mxqat_muon && cp -r $R/mm18_mxqat/seed $R/mm18_mxqat_muon/
bash $K/run_cells.sh mxqat_muon $T $R/mm18_mxqat_muon $K/cells_muon10.txt
note "G4 mm18 lora"; ENTRYPOINT=mm18_train_export bash $K/run_cells.sh lora $T $R/mm18_lora $R/cells_mm18.txt
note "G5 mm18 lora_mxqat"; ENTRYPOINT=mm18_train_export bash $K/run_cells.sh lora_mxqat $T $R/mm18_lora_mxqat $R/cells_mm18.txt
note "G6 frozen stages"; ENTRYPOINT=mm18_train_export bash $K/run_cells.sh lora_mla $T $R/frozen $R/cells_frozen.txt
note "G7 resume"; bash $R/run_resume_all.sh $R/mm18_mxqat/seed/step-0 $R/resume
note "G8 CI cells"; bash $V/kit_ci/run_ci_cells.sh
note "G9 Elfie entries"; bash $R/run_elfie_entries.sh; bash $R/run_memory32.sh
note "G10 GPU lanes"; lane $R/lane1_stack.txt "not multi_gpu" 0 29621; lane $R/lanem_stack.txt multi_gpu 0,1,2,3,4,5,6,7 29622
wait
note "all done"; echo done > $R/DONE
