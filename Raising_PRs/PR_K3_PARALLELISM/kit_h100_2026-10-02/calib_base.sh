#!/bin/bash
# DEP calibration for the K3 range on one H100: the widened debug text (dim 6144, 17 layers) with the tower not widened
# (DEPV_TOWER=base: 256 wide, 2 layers), encode vs the mean middle stage of the 16 stage split (pp4 x vpp4) at seq 2048
# and 4096, then the planner's cost ratio for every (seq, image side, images-per-sample cap) from the dataset's image-count
# distribution (choose_levels_k3range.py). The 09-30 calibration used the released K3 tower, whose encode is about 21 ms
# whatever the image side (latency bound), against a 9 ms stage: cost ratio 2.3 and up, outside the K3 range.
M=~/mep; K=~/kit/overnight/dep_ratio; KN=~/kit/kit_h100_2026-10-02; W=$M/w/dep; O=$M/results/dep_calib_base; mkdir -p $O
. $M/venv_src/bin/activate
cd $W
for seq in 2048 4096; do
  DEPV_TOWER=base CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$K:. timeout 1800 python $K/microbench_ratio.py --dim 6144 --seq $seq \
    --stages 16 > $O/microbench_base_d6144_s$seq.txt 2>&1
  echo "microbench seq $seq rc=$?"
done
python $KN/choose_levels_k3range.py /root/dep_data/t2i1024_k4/MANIFEST.json $O/microbench_base_d6144_s*.txt > $O/levels.md 2>&1
cat $O/levels.md
echo CALIB_DONE
