#!/bin/bash
# DEP calibration on one H100: encode vs text stage, the widened debug text (dim 6144, 17 layers, seq 2048) with the
# released K3 tower (DEPV_TOWER=k3: 27 layers, dim 1024) and, for reference, the 2-layer debug tower; then the vision
# share of the step for each (seq, images-per-sample cap) from the dataset's image-count distribution.
M=~/mep; K=~/kit/overnight/dep_ratio; W=$M/w/dep; O=$M/results/dep_calib; mkdir -p $O
. $M/venv_src/bin/activate
cd $W
for t in k3 debug; do
  DEPV_TOWER=$t CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$K:. timeout 1200 python $K/microbench_ratio.py --dim 6144 --seq 2048 \
    > $O/microbench_${t}_d6144_s2048.txt 2>&1
  echo "microbench $t rc=$?"
done
python ~/kit/kit_h100_2026-09-30/choose_levels.py $O/microbench_k3_d6144_s2048.txt /root/dep_data/t2i1024_k4/MANIFEST.json \
  > $O/levels.txt 2>&1
cat $O/levels.txt
echo CALIB_DONE
