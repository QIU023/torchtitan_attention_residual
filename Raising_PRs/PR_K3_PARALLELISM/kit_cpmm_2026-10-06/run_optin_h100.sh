#!/bin/bash
# #4380 opt-in check (c5fdbcca5) on the H100: the PP collective probe, unit tests, pyrefly, then end to end on one cache:
# main, the new head with dynamic CP off (default) and on (256), the measured head at 256; and the h100 CP recipes with
# the DistMuon fix, new head against the measured head on a copy of the 10-08 fix-smoke cache.
O=/workspace/h100_optin; rm -rf $O; mkdir -p $O; L=$O/summary.txt; K=/workspace/kit_cpmm
. /workspace/kit_setup/env_cu126.sh
git clone -q --depth 4 -b cpmm_review1 https://github.com/QIU023/torchtitan.git /workspace/tt_new_c 2>/dev/null || true
rm -rf /workspace/tt_new && mv /workspace/tt_new_c /workspace/tt_new && echo "tt_new $(git -C /workspace/tt_new rev-parse HEAD)" >> $L
rm -rf /workspace/tt_new_fix && cp -r /workspace/tt_new /workspace/tt_new_fix && cd /workspace/tt_new_fix && git fetch -q --depth 2 origin k3_cp_muon_layout && git -c user.name=probe -c user.email=probe@local cherry-pick FETCH_HEAD > /dev/null && echo "tt_new_fix $(git rev-parse HEAD)" >> $L
cd /workspace/tt_new
PYTHONPATH=/workspace/kit_setup/shims:. CUDA_VISIBLE_DEVICES= timeout 300 torchrun --nproc_per_node=8 --master_port=29720 $K/probe_install_pp.py > $O/pp_probe.log 2>&1; echo "pp probe rc=$? $(grep -c PP_PROBE_OK $O/pp_probe.log) ok" >> $L
PYTHONPATH=/workspace/kit_setup/shims:. CUDA_VISIBLE_DEVICES= timeout 900 python -m pytest -q -p no:cacheprovider tests/unit_tests/cpu/test_kimi_k3_vision_cp_plan.py > $O/cpu_plan.log 2>&1; echo "cpu plan rc=$? $(tail -1 $O/cpu_plan.log)" >> $L
start=$(date +%s); PYTHONPATH=/workspace/kit_setup/shims:. CUDA_VISIBLE_DEVICES=0,1,2,3 DISTRIBUTED_TESTS_DEFAULT_TIMEOUT=3000 TORCHINDUCTOR_CACHE_DIR=$O/ut_ic TRITON_CACHE_DIR=$O/ut_tc timeout 1800 python -m pytest -v -p no:cacheprovider --durations=0 tests/unit_tests/gpu/test_kimi_k3_vision_cp.py > $O/gpu_test.log 2>&1; echo "gpu unit rc=$? wall $(( $(date +%s) - start )) s: $(tail -1 $O/gpu_test.log)" >> $L
( rm -rf /workspace/pyrefly_new && cp -r /workspace/tt_new /workspace/pyrefly_new && cd /workspace/pyrefly_new && timeout 1800 pyrefly check --remove-unused-ignores --summarize-errors > $O/pyrefly_new.log 2>&1; echo "pyrefly new rc=$? $(grep -c '^ERROR' $O/pyrefly_new.log) errors, hook changed $(git status --short | wc -l) files" >> $L ) &
export KIT_ENV=/workspace/kit_setup/env_cu126.sh PP_PRE=/workspace/kit_setup/shims
( WARM_STEPS=5 STEPS=10 $K/run_matrix2.sh $O/e2e main:/workspace/tt_main:0,1:CP_MODE=allgather,WARM=e \
    new_off:/workspace/tt_new:0,1:CP_MODE=allgather,CP_MINP=off,WARM=e old_256:/workspace/tt_cpmm:0,1:CP_MODE=allgather,CP_MINP=256,WARM=e \
    new_256:/workspace/tt_new:0,1:CP_MODE=allgather,CP_MINP=256,WARM=e; echo "e2e done" >> $L ) &
( rm -rf $O/cache_cp && cp -r /workspace/h100_muonfix/cache_cp $O/cache_cp && /workspace/kit_setup/run_muon_cells.sh $O/recipe $O/cache_cp \
    old_fix_ag:/workspace/tt_cpmm_fix:2,3:MP_RECIPE=agcp new_fix_ag:/workspace/tt_new_fix:2,3:MP_RECIPE=agcp new_fix_ul:/workspace/tt_new_fix:2,3:MP_RECIPE=ulcp; echo "recipe done" >> $L ) &
wait
echo OPTIN_DONE >> $L
