#!/bin/bash
# K3-faithful parallelism matrix on 8x RTX 5060 Ti (2026-07-27).
#
# Registers NEW baselines. Every previously recorded MoE loss is invalid --
# routed experts were never initialized (EVIDENCE_INVALIDATION_2026-07-27.md),
# so nothing here is comparable to a pre-2026-07-27 number.
#
# The flavor is kimi_k3_mini_block_attn_res: K3's structure at reduced
# extents (SiTU-GLU routed experts, Gated MLA with q-compression and a
# full-rank output gate, KDA with the lower-bounded decay, Stable LatentMoE
# with 2 shared experts, AttnRes block size 12 over 21 layers, head_dim 128).
#
# Judge criteria, in order:
#   1. it runs at all (no mixed Tensor/DTensor, no shape error)
#   2. every rank prints an identical loss (rank-identity)
#   3. grad_norm is finite and in the same range as the 1D baseline
# Loss EQUALITY across degrees is NOT a criterion: FSDP2 meta-init gives each
# rank its own RNG stream, so cross-degree curves only match from a seed
# checkpoint. Do not read the columns as a numerical A/B.
set -u
TITAN=/workspace/torchtitan_attention_residual/torchtitan
OUT=${OUT:-/workspace/out_k3_matrix}
STEPS=${STEPS:-3}
cd "$TITAN"
export PYTHONPATH=$TITAN

# PP legs need microbatches >= stages, and microbatches derive from
# global_batch_size / (dp_degree * local_batch_size). With global 8 and dp 1 the
# local batch defaults to 8, giving ONE microbatch and a hard failure -- so
# every PP leg pins --training.local-batch-size 2.
COMMON="--module kimi_k3 --config kimi_k3_mini_block_attn_res \
 --training.steps $STEPS --training.global-batch-size 8 --training.seq_len 512 \
 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
 --checkpoint.no-enable"
PPB="--training.local-batch-size 2"

# Non-last PP stages report a placeholder loss, not a real one, and the value
# varies with the configuration (-2, -4 and -8 all observed). Filter on the sign
# instead of on the specific values: a cross-entropy is never negative, so any
# negative loss line is a placeholder. Keeping them in makes every PP leg look
# like a rank divergence.
losses() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+ +grad_norm: +[-0-9.]+).*/\1/' \
  | grep -vE 'loss: +-'; }
fails() { sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -iE "traceback|RuntimeError|ValueError|NotImplementedError|AssertionError" \
  | head -2; }

# A leg passes rank-identity when every rank printed the same loss line, i.e.
# the number of DISTINCT loss lines equals the number of steps.
run() {
  local name="$1" ngpu="$2" port="$3"; shift 3
  echo "=== $name (${ngpu} GPU) ==="
  local out uniq
  out=$(CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) torchrun \
        --nproc_per_node=$ngpu --master_port=$port -m torchtitan.train \
        $COMMON "$@" --dump-folder "$OUT/$name" 2>&1)
  uniq=$(echo "$out" | losses | sort -u)
  echo "$uniq"
  local n_uniq n_total
  n_uniq=$(echo "$uniq" | grep -c "step:")
  n_total=$(echo "$out" | losses | grep -c "step:")
  if [ "$n_total" -eq 0 ]; then
    echo "  -> FAIL (no steps)"; echo "$out" | fails
  elif [ "$n_uniq" -eq "$STEPS" ]; then
    echo "  -> PASS rank-identical ($n_total lines over $ngpu ranks)"
  else
    echo "  -> FAIL rank divergence ($n_uniq distinct lines, expected $STEPS)"
  fi
}

echo "########## 1D baselines ##########"
run fsdp8            8 30701 --parallelism.data_parallel_shard_degree 8
run fsdp2            2 30702 --parallelism.data_parallel_shard_degree 2
run tp2              2 30703 --parallelism.data_parallel_shard_degree 1 \
                              --parallelism.tensor_parallel_degree 2
run cp2              2 30704 --parallelism.data_parallel_shard_degree 1 \
                              --parallelism.context_parallel_degree 2
run pp2              2 30705 --parallelism.data_parallel_shard_degree 1 \
                              --parallelism.pipeline_parallel_degree 2 $PPB
# EP is carved out of the data-parallel axes, so dp_shard must cover it.
run ep2              2 30706 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.expert_parallel_degree 2

echo "########## 2D ##########"
run dp2xtp2          4 30711 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.tensor_parallel_degree 2
run dp2xcp2          4 30712 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.context_parallel_degree 2
run dp2xpp2          4 30713 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.pipeline_parallel_degree 2 $PPB
run tp2xcp2          4 30714 --parallelism.data_parallel_shard_degree 1 \
                              --parallelism.tensor_parallel_degree 2 \
                              --parallelism.context_parallel_degree 2
run tp2xpp2          4 30715 --parallelism.data_parallel_shard_degree 1 \
                              --parallelism.tensor_parallel_degree 2 \
                              --parallelism.pipeline_parallel_degree 2 $PPB
run dp2xep2xtp2      8 30716 --parallelism.data_parallel_shard_degree 4 \
                              --parallelism.expert_parallel_degree 2 \
                              --parallelism.tensor_parallel_degree 2

echo "########## 3D ##########"
run dp2xtp2xcp2      8 30721 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.tensor_parallel_degree 2 \
                              --parallelism.context_parallel_degree 2
run dp2xtp2xpp2      8 30722 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.tensor_parallel_degree 2 \
                              --parallelism.pipeline_parallel_degree 2 $PPB
run dp2xcp2xpp2      8 30723 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.context_parallel_degree 2 \
                              --parallelism.pipeline_parallel_degree 2 $PPB
run tp2xcp2xpp2      8 30724 --parallelism.data_parallel_shard_degree 1 \
                              --parallelism.tensor_parallel_degree 2 \
                              --parallelism.context_parallel_degree 2 \
                              --parallelism.pipeline_parallel_degree 2 $PPB

echo "########## QAT / QLoRA under a mesh ##########"
run qat_dp2xtp2      4 30731 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.tensor_parallel_degree 2 \
                              --config kimi_k3_mini_qat_mxfp4
run qlora_dp2xtp2    4 30732 --parallelism.data_parallel_shard_degree 2 \
                              --parallelism.tensor_parallel_degree 2 \
                              --config kimi_k3_mini_qlora
echo "########## done ##########"
